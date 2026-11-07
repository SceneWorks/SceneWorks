// Exact shipped worker resolver paths. Read approved packages; copy only into nonce-owned data.
// Never download, convert, redirect an external cache, or synthesize installation provenance.
import assert from 'node:assert/strict';
import { copyFile, mkdir, readFile, realpath, stat, statfs, writeFile } from 'node:fs/promises';
import { constants } from 'node:fs';
import path from 'node:path';
import { contained, fileHash } from './flux3-image-klein-terminal-session.mjs';
import { planHash } from './flux3-image-klein-terminal-budget.mjs';
const sha = /^[a-f0-9]{64}$/;
export class ArtifactReadinessBlocked extends Error {}
export const snapshotPath = (home, repo, rev) => path.join(home, 'hub', `models--${repo.replaceAll('/', '--')}`, 'snapshots', rev);
export function relativeMember(value) { assert(typeof value === 'string' && value.length && !value.includes('\\') && !value.includes(':') && value.split('/').every(p => p && p !== '.' && p !== '..'), 'plain repository-relative member required'); return value; }
const verifiedFiles = new Map();
export function invalidateArtifactVerification() { verifiedFiles.clear(); }
export function artifactStagingPlan(cells, registry, inputSpec) {
  const packages = new Map();
  for (const c of cells.filter(c => ['image','replay','cancel','training','installation'].includes(c.kind) && c.status !== 'BLOCKED')) {
    const row = registry.find(t => t.backend === c.backend && t.modelId === c.modelId)?.downloads.find(d => d.variant === c.tier && d.localDerivation); if (!row || packages.has(c.artifactKey)) continue;
    const inventory = inputSpec.packages?.[c.artifactKey]; assert(inventory?.files?.length, 'approved derived inventory required before copy/disk admission');
    packages.set(c.artifactKey, { artifactKey: c.artifactKey, relativeWorkerPath: `data/models/derived/${c.modelId}/${c.tier}/${row.localDerivation.conversion}`, copyFiles: inventory.files.length, copyBytes: inventory.files.reduce((n,f) => { assert(Number.isSafeInteger(f.bytes) && f.bytes > 0); return n + f.bytes; }, 0) });
  }
  return { kind: 'declared-staging-cost-not-resource-measurement', derivedPackages: [...packages.values()], totalCopyBytes: [...packages.values()].reduce((n,p) => n + p.copyBytes, 0), denseSnapshotCopies: 0, reuse: 'once per approved artifact digest/backend campaign; fresh workers reuse the same tree' };
}
export async function verifyInventoryFiles(inventory) {
  if (!(inventory?.authority === 'parent-approved-pinned-inventory' && inventory.files?.length && inventory.root)) throw new ArtifactReadinessBlocked('approved complete pinned inventory missing; separately provision/authorize the selected runner account (base HF403 remains BLOCKED), never fallback or auto-download');
  assert.equal(new Set(inventory.files.map(f => f.path)).size, inventory.files.length);
  for (const f of inventory.files) { relativeMember(f.path); assert(sha.test(f.sha256) && Number.isSafeInteger(f.bytes) && f.bytes > 0); const file = await contained(path.join(inventory.root, f.path), inventory.containmentRoot ?? inventory.root); const info = await stat(file, { bigint: true }); assert.equal(Number(info.size), f.bytes); const key = `${file}:${f.sha256}`, stamp = `${info.dev}:${info.ino}:${info.size}:${info.mtimeNs}:${info.ctimeNs}`; if (verifiedFiles.get(key) !== stamp) { assert.equal(await fileHash(file), f.sha256); verifiedFiles.set(key, stamp); } }
}
const inventoryMap = inventory => new Map(inventory.files.map(f => [f.path, f]));
async function ownedDirectory(directory, state) { const relative = path.relative(await realpath(state), await realpath(directory)); assert(relative && relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative), 'owned package directory escaped nonce state'); }
async function sameRoot(actual, expected, label) { assert.equal(await realpath(actual), await realpath(expected), `${label}: inventory root differs from actual worker resolver`); }
async function seedInstallReceipt(inventory, modelId, state) {
  const source = inventory.installationReceipt; if (!(source?.path && source.root && sha.test(source.sha256))) throw new ArtifactReadinessBlocked('genuine provisioned installation receipt required; no fabricated cache receipt or automatic download');
  const file = await contained(source.path, source.root); assert.equal(await fileHash(file), source.sha256); const envelope = JSON.parse(await readFile(file, 'utf8'));
  const receipt = (envelope.receipts ?? [envelope]).find(r => r.schemaVersion === 2 && r.modelId === modelId && r.repo === inventory.repository && r.snapshotRevision === inventory.resolvedRevision && ['bf16','default'].includes(r.variant ?? 'default'));
  assert(receipt?.jobId && receipt.completedAt); assert.deepEqual(receipt.manifestFiles, []); assert.deepEqual([...receipt.resolvedFiles].sort(), inventory.files.map(f => f.path).sort(), 'installation receipt must bind complete approved snapshot');
  const destination = path.join(state, 'data/models', modelId); await mkdir(destination, { recursive: true }); await writeFile(path.join(destination, '.sceneworks-download-complete.json'), JSON.stringify({ ...receipt, receipts: [receipt] }) + '\n', { flag: 'wx' });
  return { sourceSha256: source.sha256, receipt, marker: path.join(destination, '.sceneworks-download-complete.json') };
}
async function checkMarker(record) {
  const v = JSON.parse(await readFile(record.marker, 'utf8')), r = (v.receipts ?? [v]).find(r => r.repo === record.receipt.repo && r.modelId === record.receipt.modelId && r.snapshotRevision === record.receipt.snapshotRevision && r.jobId === record.receipt.jobId);
  assert(r?.schemaVersion === 2); assert.deepEqual(r.manifestFiles, []); assert.deepEqual(r.resolvedFiles, record.receipt.resolvedFiles);
}
export async function bindResolvedComponents(inventory, snapshot, packageRoot, tier, denseInventory) {
  const config = JSON.parse(await readFile(path.join(packageRoot, 'config.json'), 'utf8'));
  const te = path.join(snapshot, relativeMember(config.text_encoder?.path)), ae = path.join(snapshot, relativeMember(config.vae?.path));
  assert((await stat(te)).isDirectory() && (await stat(ae)).isFile());
  const rel = path.relative(await realpath(snapshot), await realpath(te)); assert(rel && !rel.startsWith('..') && !path.isAbsolute(rel), 'selected encoder escaped pinned snapshot');
  await contained(ae, denseInventory.containmentRoot ?? snapshot);
  const dense = inventoryMap(denseInventory);
  for (const role of ['text_encoder','autoencoder']) {
    const component = inventory.externalComponents?.find(c => c.role === role);
    if (tier !== 'bf16') {
      assert(component && component.repository === denseInventory.repository && component.resolvedRevision === denseInventory.resolvedRevision, `missing exact ${role} identity`);
      await sameRoot(component.root, role === 'text_encoder' ? te : path.dirname(ae), role);
      await verifyInventoryFiles(component);
      for (const f of component.files) { const full = path.join(component.root, relativeMember(f.path)), member = path.relative(snapshot, full).split(path.sep).join('/'); assert.deepEqual(dense.get(member), { path: member, bytes: f.bytes, sha256: f.sha256 }, 'component bytes are not in actual pinned snapshot inventory'); }
      if (role === 'autoencoder') assert(component.files.some(f => path.resolve(component.root, f.path) === path.resolve(ae)), 'actual selected AE file is not inventoried');
    }
  }
  const tePrefix = `${config.text_encoder.path}/`, teFiles = denseInventory.files.filter(f => f.path.startsWith(tePrefix)); assert(teFiles.some(f => f.path === `${tePrefix}config.json`) && dense.has(config.vae.path));
  return { text_encoder: { root: await realpath(te), files: teFiles }, autoencoder: { file: await realpath(ae), sha256: dense.get(config.vae.path).sha256 } };
}
async function verifyConversion(inventory, dense, target, tier) {
  const row = target.downloads.find(d => d.variant === tier && d.localDerivation), recipe = row?.localDerivation; assert(recipe, 'no real advertised derivation');
  const weights = inventoryMap(inventory).get(recipe.weightsFile); assert(weights && weights.bytes === recipe.weightsBytes && weights.sha256 === recipe.weightsSha256, 'catalog packed-weight pin mismatch');
  const receiptFile = inventoryMap(inventory).get('flux3-conversion-receipt.json'); assert(receiptFile, 'actual native conversion receipt required');
  const r = JSON.parse(await readFile(path.join(inventory.root, receiptFile.path), 'utf8')); assert.equal(r.format_version, 1); assert.equal(r.source_repository, dense.repository); assert.equal(r.source_revision, dense.resolvedRevision); assert.equal(r.algorithm_version, 1);
  assert.equal(r.algorithm, target.backend === 'mlx' ? 'mlx-affine' : tier === 'q4' ? 'ggml-q4_1' : 'ggml-q8_0'); assert.equal(r.bits, tier === 'q4' ? 4 : 8); assert.equal(r.group_size, target.backend === 'mlx' ? 64 : 32);
  const config = JSON.parse(await readFile(path.join(inventory.root, 'config.json'), 'utf8')); assert.deepEqual(config.quantization, { format: r.algorithm, bits: r.bits, group_size: r.group_size, algorithm_version: 1 }, 'native package config does not declare the actual requested storage recipe');
  const source = inventoryMap(dense), owned = inventoryMap(inventory);
  for (const [name, hash] of Object.entries(r.source_artifacts)) assert.equal(source.get(relativeMember(name))?.sha256, hash, 'conversion source artifacts differ from selected snapshot');
  const checkFiles = (files, actual) => { for (const [name, value] of Object.entries(files)) { relativeMember(name); assert.equal(actual.get(name)?.sha256, value.sha256); assert.equal(actual.get(name)?.bytes, value.bytes); } };
  checkFiles(r.package_files, owned);
  for (const c of r.components) { assert.equal(c.source_repository, dense.repository); assert.equal(c.source_revision, dense.resolvedRevision); checkFiles(c.files, c.placement === 'external' ? source : owned); for (const [name, hash] of Object.entries(c.source_artifacts)) assert.equal(source.get(relativeMember(name))?.sha256, hash); }
  assert(r.components.some(c => c.role === 'text_encoder' && c.placement === 'external') && r.components.some(c => c.role === 'autoencoder' && c.placement === 'external'));
  return recipe;
}
export async function stageTerminalArtifacts({ admission, identity, inputSpec, cells, registry }, { diskFree = async directory => { const fs = await statfs(directory, { bigint: true }); return Number(fs.bavail * fs.bsize); } } = {}) {
  const mapping = { schemaVersion: 1, ownedState: await realpath(identity.state), hfHome: admission.hfHome, snapshots: {}, packages: {}, pup: null, stagingPlan: artifactStagingPlan(cells, registry, inputSpec), copies: 0, copiedBytes: 0 };
  if (mapping.stagingPlan.totalCopyBytes) assert(await diskFree(identity.state) >= mapping.stagingPlan.totalCopyBytes + 30 * 1024 ** 3, 'insufficient actual free disk for all unique staged package copies + state safety margin (not a measurement)');
  const active = cells.filter(c => ['image','replay','cancel','training','installation'].includes(c.kind) && c.status !== 'BLOCKED');
  async function denseFor(target) {
    if (mapping.snapshots[target.modelId]) return mapping.snapshots[target.modelId];
    const inventory = inputSpec.packages?.[`${target.modelId}:${target.backend}:bf16`]; await verifyInventoryFiles(inventory);
    const row = target.downloads.find(d => d.variant === 'bf16'); assert.equal(inventory.repository, row.repo); assert.equal(inventory.resolvedRevision, row.revision);
    const root = snapshotPath(admission.hfHome, row.repo, row.revision); await sameRoot(inventory.root, root, 'dense snapshot');
    const record = { root, inventory, install: await seedInstallReceipt(inventory, target.modelId, identity.state) }; mapping.snapshots[target.modelId] = record; return record;
  }
  for (const cell of active) {
    const target = registry.find(t => t.backend === cell.backend && t.modelId === cell.modelId), dense = await denseFor(target), inventory = inputSpec.packages?.[cell.artifactKey]; await verifyInventoryFiles(inventory);
    let root = inventory.root, conversionId;
    if (['q4','q8'].includes(cell.tier)) {
      const recipe = await verifyConversion(inventory, dense.inventory, target, cell.tier); conversionId = recipe.conversion;
      root = path.join(identity.state, 'data/models/derived', cell.modelId, cell.tier, conversionId); await mkdir(root, { recursive: true }); await ownedDirectory(root, identity.state);
      if (mapping.packages[cell.artifactKey]) assert.equal(mapping.packages[cell.artifactKey].inputInventorySha256, planHash(inventory), 'conflicting package for the same worker path');
      else for (const f of inventory.files) { const to = path.join(root, relativeMember(f.path)); await mkdir(path.dirname(to), { recursive: true }); await ownedDirectory(path.dirname(to), identity.state); await copyFile(await contained(path.join(inventory.root, f.path), inventory.containmentRoot ?? inventory.root), to, constants.COPYFILE_EXCL); mapping.copies++; mapping.copiedBytes += f.bytes; }
    } else await sameRoot(root, cell.tier === 'bf16' ? dense.root : path.join(dense.root, 'variants', cell.tier), 'actual package');
    const selected = { ...inventory, root, containmentRoot: ['q4','q8'].includes(cell.tier) ? root : inventory.containmentRoot };
    await verifyInventoryFiles(selected);
    const components = await bindResolvedComponents(inventory, dense.root, root, cell.tier, dense.inventory);
    mapping.packages[cell.artifactKey] = { root, snapshotRoot: dense.root, modelId: cell.modelId, tier: cell.tier, conversionId, inputInventorySha256: planHash(inventory), selectedInventory: selected, components };
    if (cell.options.enhancePrompt) {
      const pup = inventory.components?.prompt_upsampler; await verifyInventoryFiles(pup); const row = target.model.downloads.find(d => d.componentId === 'prompt_upsampler'); assert.equal(pup.repository, row.repo); assert.equal(pup.resolvedRevision, row.revision);
      await sameRoot(pup.root, snapshotPath(admission.hfHome, row.repo, row.revision), 'actual PUP resolver');
      if (!mapping.pup) mapping.pup = { root: pup.root, inputInventorySha256: planHash(pup), inventory: pup, install: await seedInstallReceipt(pup, 'flux3_image_klein_pup', identity.state) };
      else assert.equal(mapping.pup.inputInventorySha256, planHash(pup), 'conflicting PUP component trees');
    }
  }
  return mapping;
}
export async function revalidateMappedArtifacts(mapping) {
  if (!mapping) return;
  for (const dense of Object.values(mapping.snapshots)) { await verifyInventoryFiles(dense.inventory); await checkMarker(dense.install); }
  for (const selected of Object.values(mapping.packages)) { await verifyInventoryFiles(selected.selectedInventory); const dense = mapping.snapshots[selected.modelId]; assert.deepEqual(await bindResolvedComponents(selected.selectedInventory, dense.root, selected.root, selected.tier, dense.inventory), selected.components); }
  if (mapping.pup) { await verifyInventoryFiles(mapping.pup.inventory); await checkMarker(mapping.pup.install); }
}
export async function verifyTerminalMapping(session, cell, inventory) {
  const mapping = session.artifactMapping, selected = mapping?.packages[cell.artifactKey]; assert(selected, 'no resolver-visible materialization; external package hash alone is not installed proof');
  await sameRoot(mapping.ownedState, session.identity.state, 'owned artifact state'); assert.equal(selected.inputInventorySha256, planHash(inventory));
  const dense = mapping.snapshots[cell.modelId]; await sameRoot(dense.root, snapshotPath(mapping.hfHome, inventory.repository, inventory.resolvedRevision), 'worker-selected snapshot'); await checkMarker(dense.install);
  if (selected.conversionId) { await sameRoot(selected.root, path.join(session.identity.state, 'data/models/derived', cell.modelId, cell.tier, selected.conversionId), 'worker derived tier'); await ownedDirectory(selected.root, session.identity.state); }
  else await sameRoot(selected.root, cell.tier === 'bf16' ? dense.root : path.join(dense.root, 'variants', cell.tier), 'worker package');
  await verifyInventoryFiles(selected.selectedInventory); const components = await bindResolvedComponents(inventory, dense.root, selected.root, cell.tier, dense.inventory); assert.deepEqual(components, selected.components);
  if (cell.options.enhancePrompt) { const pup = inventory.components?.prompt_upsampler; assert(mapping.pup && mapping.pup.inputInventorySha256 === planHash(pup)); await sameRoot(pup.root, snapshotPath(mapping.hfHome, pup.repository, pup.resolvedRevision), 'worker PUP'); await checkMarker(mapping.pup.install); await verifyInventoryFiles(pup); }
  return selected;
}
