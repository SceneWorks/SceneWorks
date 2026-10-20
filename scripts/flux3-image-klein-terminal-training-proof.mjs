// Native resume identity binding. No Rust Debug emulation; native pure witness owns config/factors.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile, realpath, stat } from 'node:fs/promises';
import { createReadStream } from 'node:fs';
import path from 'node:path';
import { contained, fileHash } from './flux3-image-klein-terminal-session.mjs';
import { planHash } from './flux3-image-klein-terminal-budget.mjs';
import { verifyInventoryFiles } from './flux3-image-klein-terminal-artifacts.mjs';
const sha = /^[a-f0-9]{64}$/;
export const RNG_ALGORITHM = 'flux3-splitmix64-boxmuller-f32-v1';
function u64(value) { const b = Buffer.alloc(8); b.writeBigUInt64LE(BigInt(value)); return b; }
function field(h, tag, value) { const t = Buffer.from(tag), bytes = Buffer.isBuffer(value) ? value : Buffer.from(value); h.update(u64(t.length)).update(t).update(u64(bytes.length)).update(bytes); }
export async function nativeDatasetFingerprint(plan, state, expectedItems, decodePng) {
  assert.equal(plan.config.resolution, 512); assert.equal(plan.dataset.items.length, expectedItems.length); assert(expectedItems.length > 0);
  const h = createHash('sha256'), resolution = Buffer.alloc(4); resolution.writeUInt32LE(plan.config.resolution); field(h, 'format', 'candle-training-request-v1'); field(h, 'resolution', resolution); field(h, 'item_count', u64(expectedItems.length));
  async function image(tag, file, expected) {
    const resolved = await contained(file, state); await contained(resolved, plan.dataset.rootPath); const bytes = await readFile(resolved);
    if (createHash('sha256').update(bytes).digest('hex') !== expected.sha256) {
      assert(expected.decodedContentSha256 && bytes.length >= 24, 'staged training bytes differ from approved input'); const decoded = decodePng(bytes, bytes.readUInt32BE(16), bytes.readUInt32BE(20), { allowBlank: true, forReference: true }); assert.equal(decoded.decodedContentSha256, expected.decodedContentSha256);
    }
    const nativePath = process.platform === 'win32' ? path.toNamespacedPath(await realpath(resolved)) : await realpath(resolved);
    field(h, tag, nativePath); h.update(u64((await stat(resolved)).size)); for await (const chunk of createReadStream(resolved)) h.update(chunk);
  }
  for (const [i, item] of plan.dataset.items.entries()) {
    const expected = expectedItems[i]; assert.equal(item.caption, expected.caption); assert(!item.controlImagePath && !item.subjectMaskPath); field(h, 'item_index', u64(i)); field(h, 'caption', item.caption); await image('image', item.imagePath, expected.target); field(h, 'has_control', Buffer.from([0]));
    const refs = item.referenceImagePaths ?? []; assert.equal(refs.length, expected.reference ? 1 : 0);
    if (refs.length) { field(h, 'reference_count', u64(refs.length)); await image('reference', refs[0], expected.reference); }
  }
  return h.digest('hex');
}
export async function expectedNativeArtifacts(dense) {
  await verifyInventoryFiles(dense.inventory); const rows = new Map(dense.inventory.files.map(f => [f.path, f.sha256])), config = JSON.parse(await readFile(path.join(dense.root, 'config.json'), 'utf8'));
  const te = config.text_encoder.path, ae = config.vae.path; const out = { 'dit/model.safetensors': rows.get('model.safetensors'), 'autoencoder/ae.safetensors': rows.get(ae), 'package/config.json': rows.get('config.json') };
  const index = JSON.parse(await readFile(path.join(dense.root, te, 'model.safetensors.index.json'), 'utf8')); const shards = [...new Set(Object.values(index.weight_map))].sort(); assert(shards.length > 0);
  for (const [i, name] of shards.entries()) { assert(!name.includes('..') && !path.isAbsolute(name)); out[`text_encoder/shard${i}`] = rows.get(`${te}/${name}`); }
  for (const name of ['config.json','tokenizer.json','tokenizer_config.json','model.safetensors.index.json','chat_template.json','generation_config.json']) if (rows.has(`${te}/${name}`)) out[`text_encoder/${name}`] = rows.get(`${te}/${name}`);
  assert(Object.values(out).every(v => sha.test(v ?? ''))); return out;
}
export async function bindTrainingExpectation(session, plan, requestedConfig, expectedItems, witness, decodePng) {
  assert.equal(witness.formatVersion, 1); assert.equal(witness.inputSha256, planHash(plan), 'witness does not cover exact persisted plan bytes');
  for (const key of ['sourceSha','inferenceMainSha','binarySha256','buildReceiptSha256']) assert.equal(witness[key], session.execution[key], 'native witness came from different build/source');
  assert(sha.test(witness.configSha256) && witness.rngAlgorithm === RNG_ALGORITHM); assert.deepEqual(plan.config, requestedConfig, 'persisted training config differs from actual submitted config');
  assert(plan.config.steps === 4 && plan.config.rank === 4 && plan.config.saveEvery === 1 && plan.config.resolution === 512 && plan.config.batchSize === 1 && plan.config.gradientAccumulation === 1);
  const dense = session.artifactMapping?.snapshots.flux3_klein_base; assert(dense, 'training requires mapped real dense base/teacher/AE inventory'); assert.equal(await realpath(plan.target.baseModelPath), await realpath(dense.root));
  assert.equal(witness.baseRepository, dense.inventory.repository); assert.equal(witness.baseRevision, dense.inventory.resolvedRevision); assert.equal(witness.rank, plan.config.rank); assert.equal(witness.alpha, plan.config.alpha); assert.equal(witness.seed, plan.config.seed); assert.equal(witness.accumulation, 1); assert.equal(witness.resolution, 512); assert.equal(witness.steps, 4);
  assert(witness.factors && Object.keys(witness.factors).length && Object.values(witness.factors).every(shape => Array.isArray(shape) && shape.length === 2 && shape.every(v => Number.isInteger(v) && v > 0)));
  assert(Array.isArray(witness.activeFactorKeys) && witness.activeFactorKeys.length && new Set(witness.activeFactorKeys).size === witness.activeFactorKeys.length && witness.activeFactorKeys.every(key => Object.hasOwn(witness.factors, key)), 'native witness must identify exact active workflow factors');
  assert(witness.configuration && witness.configuration.config_sha256 === witness.configSha256 && witness.configuration.rng === RNG_ALGORITHM && witness.configuration.accumulation === '1');
  const dataset = await nativeDatasetFingerprint(plan, session.identity.state, expectedItems, decodePng), artifacts = await expectedNativeArtifacts(dense);
  return { configSha256: witness.configSha256, configuration: witness.configuration, factors: witness.factors, activeFactorKeys: witness.activeFactorKeys, trainingReference: witness.trainingReference, baseRepository: witness.baseRepository, baseRevision: witness.baseRevision, artifacts, dataset, planSha256: planHash(plan), seed: witness.seed, policy: `micro_steps=4;accumulation=1;sampler_len=${expectedItems.length}`, samplerLength: expectedItems.length, witness };
}
export function verifyNativeContinuation(state, expected, previous) {
  const identity = state.identity, cursor = state.cursor; assert.equal(identity.schema, 1); assert.equal(identity.training_reference, expected.trainingReference); assert.equal(identity.base_repo, expected.baseRepository); assert.equal(identity.base_revision, expected.baseRevision);
  assert.deepEqual(identity.artifacts, expected.artifacts, 'native model/AE/teacher inventory mismatch'); assert.equal(identity.dataset, expected.dataset, 'native dataset/captions/path/content identity mismatch'); assert.deepEqual(identity.factors, expected.factors, 'native factor selection/shape differs from actual mapped plan');
  assert.equal(identity.configuration.recipe_sha256, identity.preprocessing); assert(sha.test(identity.preprocessing)); const { recipe_sha256, ...configuration } = identity.configuration; assert.deepEqual(configuration, expected.configuration, 'native options/config identity differs from pure mapper witness'); assert.equal(identity.configuration.config_sha256, expected.configSha256);
  assert.equal(state.policy, expected.policy); assert.equal(cursor.schema, 1); assert(Number.isInteger(cursor.micro_step) && cursor.micro_step > 0 && cursor.micro_step <= 4); assert.equal(cursor.update_step, cursor.micro_step); assert.equal(cursor.accumulation_count, 0, 'accumulation-one checkpoint cannot carry pending gradients');
  assert(Number.isInteger(cursor.sampler_epoch) && cursor.sampler_epoch >= 0 && Number.isInteger(cursor.sampler_cursor) && cursor.sampler_cursor >= 0 && cursor.sampler_cursor <= expected.samplerLength); assert.equal(cursor.micro_step, cursor.sampler_epoch * expected.samplerLength + cursor.sampler_cursor);
  const rng = Buffer.concat([Buffer.from(RNG_ALGORITHM), Buffer.from([0]), u64(expected.seed), u64(cursor.micro_step)]); assert.deepEqual(cursor.rng_state, [...rng], 'RNG algorithm/seed/next-step lineage mismatch; refusing reset');
  assert.deepEqual(state.pendingKeys, []); assert(!Object.keys(state.tensorManifest).some(k => k.startsWith('pending/'))); const gain = Buffer.alloc(8); gain.writeDoubleBE(Number(expected.witness.alpha) / expected.witness.rank); assert.equal(state.gainBits, gain.toString('hex'));
  assert(Number.isInteger(cursor.last_loss_bits) && cursor.last_loss_bits >= 0 && cursor.last_loss_bits <= 0xffffffff); const loss = Buffer.alloc(4); loss.writeUInt32LE(cursor.last_loss_bits); assert(Number.isFinite(loss.readFloatLE()) && loss.readFloatLE() >= 0);
  const active = new Set(expected.activeFactorKeys), factorNames = Object.keys(expected.factors);
  assert.deepEqual(Object.keys(state.tensorManifest).filter(k => k.startsWith('factor/')).sort(), factorNames.map(k => `factor/${k}`).sort());
  assert.deepEqual(Object.keys(state.optimizer).sort(), (state.format === 'flux3-mlx-training-resume-v1' ? [...active] : factorNames).sort(), 'optimizer membership differs from native workflow active set');
  for (const [name, shape] of Object.entries(expected.factors)) {
    assert.deepEqual(state.tensorManifest[`factor/${name}`]?.shape, shape);
    const optim = state.optimizer[name]; if (!optim) continue;
    for (const key of [optim.first, optim.second]) assert.deepEqual(state.tensorManifest[key]?.shape, shape, 'optimizer moment shape differs from its actual factor');
    if (active.has(name)) assert.equal(optim.step, cursor.update_step, 'active native optimizer did not update every accumulation-one step');
  }
  if (previous) {
    const advanced = keys => keys.some(key => state.tensorManifest[key].sha256 !== previous.tensorManifest[key]?.sha256);
    assert(advanced([...active].map(key => `factor/${key}`)), 'active learned factors did not advance after resume');
    assert(advanced([...active].flatMap(key => [state.optimizer[key].first, state.optimizer[key].second])), 'active native optimizer moments did not advance after resume');
    for (const name of factorNames.filter(key => !active.has(key))) assert.equal(state.tensorManifest[`factor/${name}`].sha256, previous.tensorManifest[`factor/${name}`]?.sha256, 'inactive workflow factor changed after resume'); assert.deepEqual(identity, previous.identity, 'resumed immutable identity changed (dataset/options/factors/components)'); assert.equal(state.policy, previous.policy); assert.equal(state.gainBits, previous.gainBits); assert(cursor.micro_step > previous.cursor.micro_step, 'native cursor did not advance'); assert.equal(cursor.micro_step, 4, 'completed four-step run must finish the actual native policy'); }
}
