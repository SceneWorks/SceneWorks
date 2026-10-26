import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdirSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { inventoryMossTrainingCache, mossTrainingHubRoot, requiredMossTrainingCacheFiles } from './moss-training-cache-inventory.mjs';

const revision = '1'.repeat(40);
const row = (repo, file, extra = {}) => ({ provider: 'huggingface', repo, revision, files: [file], ...extra });
const catalog = { models: [
  { id: 'moss_image_base', downloads: [row('Test/Base', 'transformer/model.safetensors'), row('Test/Sft', 'vae/model.safetensors')] },
  { id: 'moss_image_sft', downloads: [row('Test/Sft', 'transformer/model.safetensors'), row('Test/Sft', 'vae/model.safetensors')], conditionalComponents: [
    row('Test/Dmd', 'lora.safetensors', { requiredFor: ['moss_dmd_unmerged_bf16'] }),
    row('Test/Dmd', 'transformer/model.safetensors', { requiredFor: ['moss_dmd_full_fp32'] }),
    row('Test/Edit', 'edit.safetensors', { requiredFor: ['moss_image_edit'] })
  ] }
] };
function setup(t) {
  const root = mkdtempSync(path.join(tmpdir(), 'moss-cache-stat-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  mkdirSync(path.join(root, 'config/manifests'), { recursive: true });
  writeFileSync(path.join(root, 'config/manifests/builtin.models.jsonc'), JSON.stringify(catalog));
  return { root, home: path.join(root, 'home'), env: {} };
}
function materializeFiles(options, phase) {
  const report = inventoryMossTrainingCache({ ...options, phase });
  for (const item of report.requiredFiles) {
    mkdirSync(path.dirname(item.path), { recursive: true });
    writeFileSync(item.path, 'presence only, not model weights');
  }
  return report;
}
test('Hub roots preserve native override precedence and OS-home default', () => {
  const root = path.resolve('source'), home = path.resolve('native-home');
  assert.equal(mossTrainingHubRoot({ root, home, env: {} }), path.join(home, '.cache/huggingface/hub'));
  assert.equal(mossTrainingHubRoot({ root, home, env: { HF_HOME: 'selected-home' } }), path.join(root, 'selected-home/hub'));
  assert.equal(mossTrainingHubRoot({ root, home, env: { HF_HOME: '/home', HUGGINGFACE_HUB_CACHE: '/legacy', HF_HUB_CACHE: ' /hub ' } }), path.resolve('/hub'));
  assert.equal(mossTrainingHubRoot({ root, home, env: { HF_HUB_CACHE: ' ', HUGGINGFACE_HUB_CACHE: 'legacy' } }), path.join(root, 'legacy'));
});
test('missing immutable snapshots and files report absence; literal files suffice without parsing model data', t => {
  const options = setup(t);
  const missing = inventoryMossTrainingCache({ ...options, phase: 'capture' });
  assert.equal(missing.ready, false);
  assert.ok(missing.missing.some(item => item.expected === 'directory'));
  assert.ok(missing.missing.some(item => item.expected === 'file'));
  materializeFiles(options, 'capture');
  const complete = inventoryMossTrainingCache({ ...options, phase: 'capture' });
  assert.equal(complete.ready, true);
  assert.match(complete.verification, /stat-only/);
  rmSync(complete.requiredFiles[0].path);
  assert.equal(inventoryMossTrainingCache({ ...options, phase: 'capture' }).ready, false);
});
test('cross-load does not require DMD; capture requires both DMD modes and excludes unrelated editing', t => {
  const options = setup(t);
  materializeFiles(options, 'cross-load');
  assert.equal(inventoryMossTrainingCache({ ...options, phase: 'cross-load' }).ready, true);
  const capture = inventoryMossTrainingCache({ ...options, phase: 'capture' });
  assert.equal(capture.ready, false);
  assert.ok(capture.missing.some(item => item.file === 'lora.safetensors'));
  assert.ok(capture.missing.some(item => item.repository === 'Test/Dmd' && item.file === 'transformer/model.safetensors'));
  assert.ok(capture.requiredFiles.every(item => item.repository !== 'Test/Edit'));
});
test('normal Hub file symlinks are accepted and directories cannot stand in for required files', t => {
  const options = setup(t), report = materializeFiles(options, 'cross-load'), target = report.requiredFiles[0].path;
  rmSync(target);
  // Windows Hub caches can use regular copies; creating symlinks there requires
  // a host privilege unrelated to this metadata-only check.
  if (process.platform !== 'win32') {
    const blob = path.join(options.root, 'blob'); writeFileSync(blob, 'literal source bytes'); symlinkSync(blob, target);
    assert.equal(inventoryMossTrainingCache({ ...options, phase: 'cross-load' }).ready, true);
    rmSync(target);
  }
  mkdirSync(target);
  assert.equal(inventoryMossTrainingCache({ ...options, phase: 'cross-load' }).ready, false);
});
test('only exact immutable production file identities enter the stat inventory', () => {
  assert.throws(() => requiredMossTrainingCacheFiles(catalog, 'finalize'), /phase/);
  const changed = structuredClone(catalog); changed.models[0].downloads[0].revision = 'main';
  assert.throws(() => requiredMossTrainingCacheFiles(changed, 'capture'), /exact/);
  changed.models[0].downloads[0].revision = revision; changed.models[0].downloads[0].files = ['../outside'];
  assert.throws(() => requiredMossTrainingCacheFiles(changed, 'capture'), /literal/);
});
