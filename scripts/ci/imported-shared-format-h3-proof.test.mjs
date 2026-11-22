import assert from 'node:assert/strict';
import test from 'node:test';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { makeScenario, validateNativeH3Receipts } from './imported-shared-format-h3-proof.mjs';

test('one existing H3 source uses fixed DiT precision and retains the inherited tier', async () => {
  const fixture = JSON.parse(await readFile(new URL('../../docs/testing/shared-format-h3-convrot.json', import.meta.url)));
  const inputs = fixture.binding.model.components.map(component => ({ role: 'component',
    path: path.join('D:/owned', 'inputs/hf/hub', `models--${component.source.replaceAll('/', '--')}`,
      'snapshots', component.revision, component.file), sha256: component.sha256 }));
  const { c, scenario } = makeScenario(fixture, 'D:/owned', 'D:/owned/model.safetensors', inputs, []);
  assert.equal(c.kind, 'full-av');
  assert.equal(c.runtime.gpuId, '0');
  assert.equal(c.runtime.physicalGpuIndex, 0);
  assert.equal(c.runtime.deviceId, 'GPU-b1a31911-c7b4-2901-3d8b-9a62e228bfc0');
  assert.equal(c.runtime.physicalGpuUuid, c.runtime.deviceId);
  const imports = scenario.steps.filter(step => step.path === '/api/v1/models/import');
  const generations = scenario.steps.filter(step => step.path === '/api/v1/video/jobs');
  assert.equal(imports.length, 1); assert.equal(generations.length, 1);
  assert(imports[0].assert.some(row => row.pointer === '/payload/manifestEntry/importComponentTier' && row.equals === 'q4'));
  assert.equal(generations[0].body.advanced.mlxQuantize, undefined);
  assert.deepEqual(generations[0].body.loras, []);
  assert.equal(generations[0].body.advanced.steps, 8);
  assert.equal(generations[0].body.advanced.schedulerShift, 12);
  assert.equal(generations[0].body.advanced.audioSchedulerShift, 3);
});

test('H3 native receipt requires the actual completed job and its sealed source closure', () => {
  const bundle = { jobs: [{ id: 'job-1', status: 'completed' }], bindings: [{ job: { id: 'job-1' },
    event: { prepared_sources: JSON.stringify([{ path: 'D:/owned/dit.safetensors' }]) } }] };
  const row = { event: 'checkpoint_weight_materialization', jobId: 'job-1', receipts: JSON.stringify([
    { sourcePath: 'D:/owned/dit.safetensors', checkpointWeightFacts: { materialization: { status: 'reported',
      rows: [{ codecId: 'convrot-int8-native', representation: 'native-packed', tensorCount: 4 }] } } },
  ]) };
  assert.equal(validateNativeH3Receipts(JSON.stringify(row), bundle).nativeTensors, 4);
  const grouped = structuredClone(row), groupBundle = structuredClone(bundle);
  groupBundle.bindings[0].event.prepared_sources = JSON.stringify([
    { path: 'D:/owned/dit.safetensors' }, { path: 'D:/owned/shard-2.safetensors' },
  ]);
  const groupedReceipts = JSON.parse(grouped.receipts);
  delete groupedReceipts[0].sourcePath;
  groupedReceipts[0].sourcePaths = ['D:/owned/dit.safetensors', 'D:/owned/shard-2.safetensors'];
  grouped.receipts = JSON.stringify(groupedReceipts);
  assert.equal(validateNativeH3Receipts(JSON.stringify(grouped), groupBundle).nativeTensors, 4);
  assert.throws(() => validateNativeH3Receipts(JSON.stringify(grouped), bundle));
  for (const mutate of [
    value => { value.jobId = 'another-job'; },
    value => { const r = JSON.parse(value.receipts); r[0].sourcePath = 'D:/foreign/dit.safetensors'; value.receipts = JSON.stringify(r); },
    value => { value.receipts = value.receipts.replace('native-packed', 'dense-fallback'); },
    value => { value.receipts = value.receipts.replace('"tensorCount":4', '"tensorCount":0'); },
  ]) {
    const changed = structuredClone(row); mutate(changed);
    assert.throws(() => validateNativeH3Receipts(JSON.stringify(changed), bundle));
  }
});
