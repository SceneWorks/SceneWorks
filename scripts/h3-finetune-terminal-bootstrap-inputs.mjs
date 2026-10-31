// First-run input resolution for the existing H3 terminal routes. No CLI, model work or code execution.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { open, lstat, mkdir, realpath, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { stripJsoncComments } from './lib/jsonc.mjs';
import { patternToRegExp } from './check-download-patterns.mjs';
import { cacheHub, stageOneFile, sourceUrl } from './h3-finetune-terminal-stage.mjs';
import { FROZEN_PROVENANCE } from './h3-finetune-terminal-artifact-probe.mjs';

const SMALL_FILE_LIMIT = 32 * 1024 * 1024;
// The file extensions the product's H3 component loader admits (`members()`,
// crates/sceneworks-core/src/minimax_h3_load.rs). A catalog glob such as `FL2VA/audio_vae/*` also
// stages loader sources (.py) the product never reads: they stay staged inputs (the product's own
// download set) but are not declared model components (sc-26691).
export const PRODUCT_COMPONENT_EXTENSIONS = Object.freeze(['safetensors', 'gguf', 'json', 'yaml', 'yml', 'model', 'txt', 'tiktoken']);
export const isProductComponentMember = file => PRODUCT_COMPONENT_EXTENSIONS.includes(path.posix.extname(String(file)).slice(1));
const METADATA_LIMIT = 8 * 1024 * 1024;
const REFERENCE_LIMIT = 4 * 1024 * 1024;
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const sha40 = value => assert.match(value ?? '', /^[a-f0-9]{40}$/, 'immutable revision/Git blob SHA1 required');
const sha64 = value => assert.match(value ?? '', /^[a-f0-9]{64}$/, 'content SHA256 required (not Git blob SHA1)');
const positive = value => assert.ok(Number.isSafeInteger(value) && value > 0, 'actual positive file byte size required');
const below = (root, file) => { const r = path.relative(root, file); return r !== '' && !r.startsWith('..') && !path.isAbsolute(r); };
function safeFile(file) {
  assert.ok(typeof file === 'string' && file.length > 0 && !/[\\\0:]/.test(file), 'unsafe source path');
  assert.ok(!path.posix.isAbsolute(file) && !path.win32.isAbsolute(file) && file.split('/').every(p => p && p !== '.' && p !== '..'), 'unsafe source path');
  return file;
}
function repoName(repo) { assert.match(repo ?? '', /^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/, 'exact catalog repository required'); return repo; }
async function boundedFile(file, cap = METADATA_LIMIT) {
  const handle = await open(file, 'r');
  try {
    const info = await handle.stat(); assert.ok(info.isFile() && info.size <= cap, 'bounded source metadata required');
    const bytes = Buffer.alloc(info.size); let offset = 0;
    while (offset < bytes.length) { const r = await handle.read(bytes, offset, bytes.length - offset, offset); assert.ok(r.bytesRead, 'truncated source metadata'); offset += r.bytesRead; }
    assert.equal((await handle.stat()).size, info.size, 'source metadata changed'); return bytes;
  } finally { await handle.close(); }
}
async function boundedResponse(response, cap, expectedBytes) {
  assert.equal(response.status, 200, 'required source metadata/content unavailable');
  assert.ok(response.body, 'required source has no content');
  const parts = []; let size = 0;
  try {
    for await (const chunk of response.body) {
      size += chunk.length; assert.ok(size <= cap && (expectedBytes === undefined || size <= expectedBytes), 'bounded source body exceeded actual size/limit'); parts.push(Buffer.from(chunk));
    }
  } catch (error) { await response.body.cancel?.().catch(() => {}); throw error; }
  assert.ok(size > 0, 'required source has no content');
  if (expectedBytes !== undefined) assert.equal(size, expectedBytes, 'incomplete source body');
  return Buffer.concat(parts, size);
}
function transport(deps) {
  const fetchImpl = deps.fetchImpl ?? globalThis.fetch;
  return async (url, options = {}) => {
    const remaining = deps.jobDeadline === undefined ? 60_000 : Math.min(60_000, deps.jobDeadline - Date.now());
    assert.ok(remaining > 0, 'bootstrap input job deadline exceeded');
    deps.signal?.throwIfAborted();
    const signals = [AbortSignal.timeout(remaining), ...(deps.signal ? [deps.signal] : [])];
    return fetchImpl(url, { ...options, signal: AbortSignal.any(signals), redirect: options.redirect ?? 'error' });
  };
}
// Mirrors minimax_h3_import::import_component_configuration_compatible: the canonical
// assembly role is the declared componentId, or the last segment of its catalog subdirectory.
// This classifies the catalog binding, never checkpoint architecture or tensor shapes.
function componentDeclaration(row) {
  const subdir = row.subdir ?? row.files?.find(file => file.endsWith('/*'))?.slice(0, -2);
  if (subdir !== undefined) safeFile(subdir);
  const id = row.componentId ?? subdir?.split('/').at(-1);
  assert.ok(typeof id === 'string' && id.length > 0, 'explicit unambiguous catalog component role required');
  return { id, subdir };
}
// Hard co-requisites include load-only companion partitions. Primary DiT rows and soft
// predictors remain excluded by catalog metadata, not by names of component directories.
export function selectComponentRows(catalog, selections, backend) {
  assert.ok(['mlx', 'cuda'].includes(backend), 'known terminal backend required');
  assert.ok(Array.isArray(selections) && selections.length > 0, 'explicit base/component selections required');
  const os = backend === 'mlx' ? 'macos' : 'windows', seen = new Set();
  return selections.map(({ modelId, tier }) => {
    assert.ok(['minimax_h3', 'minimax_h3_ref'].includes(modelId), 'unknown compatible base');
    assert.ok(['q4', 'q8', 'bf16'].includes(tier), 'explicit supported component tier required');
    assert.ok(!seen.has(modelId), 'one explicit component tier per base required'); seen.add(modelId);
    const matches = catalog.models?.filter(m => m.id === modelId); assert.equal(matches?.length, 1, 'unknown/ambiguous catalog base');
    const model = matches[0]; assert.equal(model.family, 'minimax-h3', 'genuine H3 catalog base required');
    const available = model.downloads.filter(r => !r.platforms || r.platforms.includes(os));
    assert.ok(available.some(r => r.coRequisite !== true && r.variant === tier), 'selected tier unavailable on this platform');
    const rows = available.filter(r => r.coRequisite === true && r.required !== 'soft' && (!r.variant || r.variant === tier));
    assert.ok(rows.length > 0, 'base declares no reusable components');
    for (const row of rows) {
      assert.equal(row.provider, 'huggingface', 'unknown catalog source provider'); repoName(row.repo); sha40(row.revision);
      componentDeclaration(row); assert.ok(!row.choice, 'explicit unambiguous catalog component required');
      assert.ok(Array.isArray(row.files) && row.files.length > 0, 'catalog component file patterns required');
      for (const pattern of row.files) { safeFile(pattern); patternToRegExp(pattern); }
    }
    assert.equal(new Set(rows.map(r => componentDeclaration(r).id)).size, rows.length, 'ambiguous catalog component selection');
    return { modelId, tier, rows };
  });
}

// HF tree metadata is authoritative per file. A Git object id is deliberately NOT a SHA256.
export async function readPinnedHfTree(repo, revision, deps = {}) {
  repoName(repo); sha40(revision);
  const fetch = transport(deps), prefix = `https://huggingface.co/api/models/${repo}`;
  const info = JSON.parse((await boundedResponse(await fetch(`${prefix}/revision/${revision}?blobs=true`), METADATA_LIMIT)).toString('utf8'));
  assert.equal(info.id, repo, 'HF served a different catalog source'); assert.equal(info.sha, revision, 'HF revision mismatch');
  const base = `${prefix}/tree/${revision}`, files = [], seen = new Set();
  let next = `${base}?recursive=1&expand=1`;
  for (let page = 0; next; page++) {
    assert.ok(page < 1000, 'HF metadata pagination exceeded bound');
    const url = new URL(next); assert.equal(url.origin, 'https://huggingface.co'); assert.equal(url.pathname, new URL(base).pathname, 'HF pagination escaped selected source');
    const response = await fetch(next), body = JSON.parse((await boundedResponse(response, METADATA_LIMIT)).toString('utf8'));
    assert.ok(Array.isArray(body), 'expanded HF tree required');
    for (const row of body) {
      if (row.type === 'directory') continue;
      assert.equal(row.type, 'file', 'unknown HF tree entry'); safeFile(row.path);
      assert.ok(!seen.has(row.path), 'duplicate HF metadata file'); seen.add(row.path); files.push(row);
      assert.ok(files.length <= 100_000, 'HF file census exceeded bound');
    }
    const links = response.headers?.get('link') ?? '';
    const following = links.split(',').filter(s => /;\s*rel="?next"?\s*$/.test(s));
    assert.ok(following.length <= 1, 'ambiguous metadata pagination');
    next = following.length ? /^\s*<([^>]+)>/.exec(following[0])?.[1] : undefined;
    assert.ok(!following.length || next, 'invalid metadata pagination');
  }
  assert.ok(files.length > 0, 'required HF source metadata empty'); return files;
}
function lfsIdentity(row) {
  positive(row.size);
  if (row.lfs !== undefined && row.lfs !== null) {
    const raw = row.lfs.oid ?? row.lfs.sha256;
    assert.ok(typeof raw === 'string', 'LFS content SHA256 missing');
    const sha256 = raw.trim().replace(/^sha256:/, '').toLowerCase(); sha64(sha256);
    if (row.lfs.sha256 !== undefined) assert.equal(row.lfs.sha256.toLowerCase(), sha256, 'conflicting LFS content hashes');
    if (row.lfs.size !== undefined) assert.equal(row.lfs.size, row.size, 'LFS size differs from actual file size');
    return sha256;
  }
  return undefined;
}
function fileLegal(book, componentId) {
  const inherited = book.stagingLicenses?.singularity;
  assert.equal(inherited?.licenseId, 'MiniMax-H3-Community-License-Agreement', 'unknown H3 source license');
  assert.equal(inherited.inheritedLicenseId, inherited.licenseId, 'unknown inherited H3 license');
  // The existing bundled minimax-h3 license record explicitly assigns Apache-2.0 to its TE.
  return { ...inherited, ...(componentId === 'text_encoder' ? { licenseId: 'Apache-2.0' } : {}) };
}
async function acknowledgment(callback, entry, context) {
  const ack = await callback(entry, context);
  assert.ok(ack && ack.acknowledged === true, 'missing explicit operator acknowledgment');
  assert.equal(ack.licenseId, entry.legal.licenseId, 'source license outside approved scope');
  if (entry.legal.inheritedLicenseId) { assert.equal(ack.inheritedLicenseId, entry.legal.inheritedLicenseId, 'inherited license outside approved scope'); assert.equal(ack.inheritedAcknowledged, true, 'missing inherited acknowledgment'); }
  if (entry.legal.attributionRequired) assert.ok(typeof ack.attribution === 'string' && ack.attribution.includes(entry.legal.attributionRequired), 'required attribution outside approved scope');
  if (entry.expectedSha256) assert.equal(ack.sha256, entry.expectedSha256, 'acknowledgment not bound to actual source hash');
  return ack;
}
async function ownedDirectory(root, relative) {
  let current = root;
  for (const part of safeFile(relative).split('/')) {
    current = path.join(current, part);
    try { await mkdir(current); } catch (error) { if (error.code !== 'EEXIST') throw error; }
    const info = await lstat(current); assert.ok(info.isDirectory() && !info.isSymbolicLink(), 'private directory cannot be a symlink');
  }
  return current;
}
export function publishedH3Config(source) {
  const match = /pub fn published\(\) -> Self\s*\{\s*Self\(json!\((\{[\s\S]*?\})\)\)/.exec(source);
  assert.ok(match, 'current H3Config::published JSON literal missing');
  const dit = JSON.parse(match[1]); assert.equal(dit._class_name, 'MiniMaxH3Transformer3DModel');
  assert.ok(!dit.quantization && !dit.quantization_config, 'published architecture must not be a packed policy'); return dit;
}

// Parse data literals only. Never import/evaluate downloaded or producer Python.
export function verifyPrunedReferenceDeclaration(bytes, expectedSha256, reference) {
  sha64(expectedSha256); assert.equal(digest(bytes), expectedSha256, 'frozen CPU producer source SHA256 mismatch');
  const source = bytes.toString('utf8');
  const revision = /^REVISION = "([a-f0-9]{40})"$/m.exec(source)?.[1];
  const sources = /^SOURCES = (\{[\s\S]*?^\})/m.exec(source)?.[1];
  assert.ok(revision && sources, 'frozen CPU producer declarations unavailable');
  assert.equal(revision, reference.referenceRevision, 'frozen producer upstream revision mismatch');
  assert.deepEqual(JSON.parse(sources.replace(/,\s*\}/g, '}')), reference.referenceSha256, 'reference source set differs from frozen producer');
}

export async function prepareBootstrapInputs({ swRoot, stageRoot, evidenceRoot, backend, componentSelections, compatibleBase, nativeEvidence, acknowledgmentFor, env = process.env, proofKind, deps = {} }) {
  assert.equal(typeof acknowledgmentFor, 'function', 'explicit legal scope callback required');
  assert.equal(compatibleBase, 'builtin:minimax_h3', 'explicit compatibleBase selection required for published config');
  const sourceRoot = await realpath(swRoot), nativeRoot = await realpath(nativeEvidence.root);
  const rootInfo = await lstat(stageRoot); assert.ok(rootInfo.isDirectory() && !rootInfo.isSymbolicLink(), 'owner-bounded private staging root required');
  const root = await realpath(stageRoot), evidence = await realpath(evidenceRoot);
  for (const external of [root, evidence]) for (const repo of [sourceRoot, nativeRoot]) assert.ok(external !== repo && !below(repo, external) && !below(external, repo), 'all created files must remain outside source repositories');
  const owner = JSON.parse((await boundedFile(path.join(root, 'sc26691-owner.json'))).toString('utf8'));
  assert.equal(owner.kind, 'h3-finetune-stage-owner'); assert.equal(await realpath(owner.root), root); assert.equal(owner.backend, backend); sha40(owner.codeSha);
  assert.match(String(owner.runId), /^\d+$/); assert.match(String(owner.runAttempt), /^\d+$/);
  assert.equal(path.basename(root), `${owner.runId}-${owner.runAttempt}`, 'private root must match actual owner run/attempt');
  assert.equal(path.basename(path.dirname(root)), 'sc26691-private', 'existing per-run staging namespace required');
  const catalog = JSON.parse(stripJsoncComments((await boundedFile(path.join(sourceRoot, 'config/manifests/builtin.models.jsonc'), 16 * 1024 * 1024)).toString('utf8')));
  const book = JSON.parse((await boundedFile(path.join(sourceRoot, 'docs/testing/sc-26691-h3-cases.json'))).toString('utf8'));
  const selections = selectComponentRows(catalog, componentSelections, backend);
  const licenses = JSON.parse((await boundedFile(path.join(sourceRoot, 'apps/desktop/licenses/manifest.json'))).toString('utf8'));
  const licenseRecords = licenses.components?.filter(c => selections.every(s => c.models?.includes(s.modelId)));
  assert.equal(licenseRecords?.length, 1, 'selected bases require one known bundled H3 license source');
  const documents = licenseRecords[0].documents.map(d => d.key);
  assert.ok(documents.includes('minimax-h3-community-license') && documents.includes('minimax-h3-qwen-apache'), 'unknown component license declaration');
  const ref = book.numeric?.['pruned-interpolation']; sha40(ref?.referenceRevision);
  assert.equal(ref.referenceUrlPrefix, `https://raw.githubusercontent.com/deepbeepmeep/WanGP/${ref.referenceRevision}/models/minimax_h3/`, 'unknown frozen reference source');
  assert.deepEqual(Object.keys(ref.referenceSha256).sort(), ['components/packing.py', 'prune_checkpoint.py', 'transformer.py']);
  for (const sha of Object.values(ref.referenceSha256)) sha64(sha);
  assert.ok(deps.frozenProducer === undefined || proofKind === 'unit_fake', 'frozen producer override is only for tiny unit fixtures');
  const frozen = deps.frozenProducer ?? FROZEN_PROVENANCE.find(p => p.script === 'scripts/h3_pruned_adaln_oracle.py');
  const producerBytes = await boundedFile(path.join(nativeRoot, 'scripts/h3_pruned_adaln_oracle.py'), REFERENCE_LIMIT);
  verifyPrunedReferenceDeclaration(producerBytes, frozen.sha256, ref);
  const configSource = path.join(sourceRoot, 'crates/sceneworks-core/src/minimax_h3_codec.rs');
  const configSourceBytes = await boundedFile(configSource), dit = publishedH3Config(configSourceBytes.toString('utf8'));
  const fetch = transport(deps), hfTrees = new Map(), resolved = [], componentsByModel = {}, componentRecords = [], seedFiles = [], inputs = [];
  // Scope preflight for EVERY named source/license before any remote metadata or model body.
  // A hash is not available yet for non-LFS sidecars: context.scopeOnly requests scope checking,
  // not fabricated consent. The same actual callback binds the measured hash before staging.
  for (const { modelId, rows } of selections) for (const row of rows) {
    const { id: componentId } = componentDeclaration(row), legal = fileLegal(book, componentId);
    await acknowledgment(acknowledgmentFor, { repoType: 'model', repo: row.repo, revision: row.revision, legal }, { component: true, modelId, componentId, legal, scopeOnly: true });
  }
  const referenceAcknowledgments = new Map();
  for (const [file, sha256] of Object.entries(ref.referenceSha256)) {
    const entry = { repoType: 'source', repo: 'deepbeepmeep/WanGP', revision: ref.referenceRevision, file, expectedSha256: sha256 };
    const context = { reference: true, kind: 'pinned-reference-source', repo: entry.repo, revision: entry.revision, path: file, sha256 };
    const ack = await acknowledgmentFor(entry, context);
    assert.equal(ack?.scope, 'existing-pinned-producer-source', 'pinned producer source scope required');
    assert.equal(ack.acknowledged, true, 'missing operator acknowledgment for selected producer dependency');
    referenceAcknowledgments.set(file, ack);
  }
  for (const { modelId, tier, rows } of selections) {
    componentsByModel[modelId] = [];
    for (const row of rows) {
      const { id: componentId, subdir: componentSubdir } = componentDeclaration(row);
      const key = `${row.repo}@${row.revision}`;
      if (!hfTrees.has(key)) hfTrees.set(key, await readPinnedHfTree(row.repo, row.revision, deps));
      const tree = hfTrees.get(key), selected = new Map();
      for (const pattern of row.files) {
        const matches = tree.filter(file => patternToRegExp(pattern).test(file.path)); assert.ok(matches.length > 0, `catalog pattern has no actual files: ${pattern}`);
        for (const file of matches) selected.set(file.path, file);
      }
      for (const file of selected.values()) {
        const entry = { repoType: 'model', repo: row.repo, revision: row.revision, file: file.path, expectedBytes: file.size, legal: fileLegal(book, componentId) };
        entry.expectedSha256 = lfsIdentity(file);
        if (!entry.expectedSha256) {
          assert.ok(file.size <= SMALL_FILE_LIMIT && !/\.(safetensors|bin|pt|pth|ckpt)$/i.test(file.path), 'weight/large file requires actual LFS content SHA256, never Git blob identity or estimates');
          const blob = file.oid ?? file.blobId; sha40(blob);
          const body = await boundedResponse(await fetch(sourceUrl(entry), { redirect: 'follow' }), SMALL_FILE_LIMIT, file.size);
          const gitSha1 = createHash('sha1').update(`blob ${body.length}\0`).update(body).digest('hex');
          assert.equal(gitSha1, blob, 'Git blob SHA1 mismatch (not a content SHA256)'); entry.expectedSha256 = digest(body);
        }
        const context = { component: true, modelId, componentId, legal: entry.legal };
        const ack = await acknowledgment(acknowledgmentFor, entry, context);
        resolved.push({ entry, ack, modelId, tier, componentId, componentSubdir, catalogDownload: row });
      }
    }
  }
  const directory = await ownedDirectory(root, 'bootstrap-inputs');
  const referenceDir = await ownedDirectory(root, 'bootstrap-inputs/reference');
  assert.ok(deps.cacheRoot === undefined || proofKind === 'unit_fake', 'cache root override is only for private unit fixtures');
  const hub = deps.cacheRoot ?? cacheHub({ env, platform: backend === 'mlx' ? 'darwin' : 'win32' });
  assert.ok(path.isAbsolute(hub), 'known exact HF cache root must be absolute');
  // Verify the existing CPU-tool dependencies before any large model transfer.
  for (const [file, sha256] of Object.entries(ref.referenceSha256)) {
    const ack = referenceAcknowledgments.get(file);
    const bytes = await boundedResponse(await fetch(`${ref.referenceUrlPrefix}${safeFile(file)}`), REFERENCE_LIMIT); assert.equal(digest(bytes), sha256, 'frozen producer reference SHA256 mismatch');
    const parent = path.posix.dirname(file); if (parent !== '.') await ownedDirectory(root, `bootstrap-inputs/reference/${parent}`);
    const target = path.join(referenceDir, file); await writeFile(target, bytes, { flag: 'wx', mode: 0o600 });
    inputs.push({ path: target, sha256, bytes: bytes.length, role: 'reference', source: 'deepbeepmeep/WanGP', revision: ref.referenceRevision, file, explicitAcknowledgment: ack });
  }
  const configBytes = Buffer.from(JSON.stringify(dit, null, 2) + '\n'), configPath = path.join(directory, 'config.json');
  await writeFile(configPath, configBytes, { flag: 'wx', mode: 0o600 });
  const config = { path: configPath, sha256: digest(configBytes), dit, compatibleBase, provenance: { source: configSource, sha256: digest(configSourceBytes), codeSha: owner.codeSha, declaration: 'H3Config::published' } };
  inputs.push({ path: config.path, sha256: config.sha256, role: 'source-config' });
  const completed = new Map();
  for (const { entry, ack, modelId, tier, componentId, componentSubdir, catalogDownload } of resolved) {
    const key = `${entry.repo}@${entry.revision}/${entry.file}`;
    let record = completed.get(key);
    if (!record) {
      record = await (deps.stageOneFile ?? stageOneFile)({ entry, root, acknowledgment: ack, fetchImpl: deps.fetchImpl ?? globalThis.fetch, env, hub, signal: deps.signal, jobDeadline: deps.jobDeadline, maxRetries: deps.maxRetries ?? 3 });
      assert.equal(record.status, 'completed'); assert.equal(record.actualBytes, entry.expectedBytes); assert.equal(record.observedSha256, entry.expectedSha256);
      const actual = await realpath(record.finalPath); assert.ok(below(root, actual), 'staged component escaped private owner');
      const info = await lstat(record.finalPath); assert.ok(info.isFile() && !info.isSymbolicLink()); assert.equal(info.size, record.actualBytes);
      componentRecords.push(record); completed.set(key, record);
      inputs.push({ path: record.finalPath, sha256: record.observedSha256, role: 'component' });
      seedFiles.push({ input: record.finalPath, destination: `hf/hub/models--${entry.repo.replaceAll('/', '--')}/snapshots/${entry.revision}/${entry.file}`, mode: 'hardlink' });
    }
    if (isProductComponentMember(entry.file)) componentsByModel[modelId].push({ id: componentId, source: entry.repo, revision: entry.revision, sha256: record.observedSha256, path: record.finalPath, file: entry.file, bytes: record.actualBytes, tier, licenseAcknowledged: ack.acknowledged, explicitAcknowledgment: ack, provenance: { componentSubdir, catalogDownload: structuredClone(catalogDownload) } });
  }
  // hf-hub cache lookup may require refs/<requested revision> even when the immutable
  // snapshot exists. These are private cache-index bytes, not model components or refs/main.
  const revisions = new Map(resolved.map(({ entry }) => [`${entry.repo}@${entry.revision}`, entry]));
  for (const { repo, revision } of revisions.values()) {
    const destination = `hf/hub/models--${repo.replaceAll('/', '--')}/refs/${revision}`;
    await ownedDirectory(root, path.posix.dirname(destination));
    const target = path.join(root, destination), bytes = Buffer.from(revision, 'ascii');
    await writeFile(target, bytes, { flag: 'wx', mode: 0o600 });
    inputs.push({ path: target, sha256: digest(bytes), role: 'cache-ref' });
    seedFiles.push({ input: target, destination, mode: 'copy' });
  }
  return { componentsByModel, seedFiles, inputs, config, referenceDir, componentRecords, referenceRevision: ref.referenceRevision };
}
