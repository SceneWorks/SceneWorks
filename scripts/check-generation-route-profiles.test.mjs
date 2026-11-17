import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { verifyProfileReceipt } from './check-generation-route-profiles.mjs';

function fixture() {
  const ios = 'snapshot.generator_ids, ["image"], snapshot.text_llm_ids, ["llm"], snapshot.transform_ids, ["restore"], snapshot.snapshot_preparer_backends, ["mlx"]';
  const cpu = 'snapshot.text_llm_ids, ["candle-llama"] candle_gen_catalog::provider_registry() candle_audio_catalog::provider_registry()';
  const files = { 'crates/bundles/runtime-ios/src/lib.rs': ios, 'crates/bundles/runtime-cpu/src/lib.rs': cpu };
  return { files, receipt: { inferenceRevision: 'old', sources: Object.fromEntries(Object.entries(files).map(([path, source]) => [path, createHash('sha256').update(source).digest('hex')])),
    ios: { generator_ids: ['image'], text_llm_ids: ['llm'], transform_ids: ['restore'], snapshot_preparer_backends: ['mlx'] }, cpu: { text_llm_ids: ['candle-llama'] }, cpuSharedCatalogs: ['candle_gen_catalog::provider_registry()', 'candle_audio_catalog::provider_registry()'] } };
}
test('profile provenance verifies revision bytes and independently extracted assertions', () => {
  const { files, receipt } = fixture();
  assert.equal(verifyProfileReceipt(receipt, (revision, path) => { assert.equal(revision, 'old'); return files[path]; }).iosGenerators, 1);
  receipt.ios.generator_ids = [];
  assert.throws(() => verifyProfileReceipt(receipt, (_, path) => files[path]), /iOS generator_ids/);
});
test('changed owning source is rejected even if the expected member survives', () => {
  const { files, receipt } = fixture();
  assert.throws(() => verifyProfileReceipt(receipt, (_, path) => files[path] + '\n'), /source receipt changed/);
});
