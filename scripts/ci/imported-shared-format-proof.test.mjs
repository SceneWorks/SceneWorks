// Metadata only: no model, tensor, CUDA or subprocess execution.
import assert from 'node:assert/strict';
import test from 'node:test';
import { CASES, validateResult } from './imported-shared-format-proof.mjs';

function fixture(spec) {
  const fqn = `imported_comfy_quant_gpu_smoke::${spec.test}`;
  return {
    log: `${Array(spec.count).fill('RESULT status=pass native request').join('\n')}\n`
      + `MODEL_INFERENCE_BODY_COMPLETED ${fqn}\nMODEL_INFERENCE_DEVICE ${fqn} candle-cuda\n`
      + 'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 100 filtered out;\n'
      + (spec.count === 2 ? '{"fields":{"event":"generator_cache_miss"}}\n' : ''),
    packets: Array.from({ length: spec.count }, (_, iteration) => ({
      schemaVersion: 1, modelId: spec.model, iteration, sourceCodec: spec.codec, physicalGpu: '1',
      checkpointId: 'linked/root-1234567890abcdef/model.safetensors',
      importPlan: { checkpointId: 'linked/root-1234567890abcdef/model.safetensors',
        planId: `checkpoint-plan-${'a'.repeat(64)}`, semanticDigest: `sha256:${'b'.repeat(64)}` },
      adapterCount: spec.adapters, width: spec.width, height: spec.height, output: `${spec.model}-${iteration}.png`,
      runtimeFacts: { materialization: { status: 'reported', rows: [{ codecId: spec.codec, representation: 'native-packed', tensorCount: 3 }] } },
      metrics: [{ loadMs: 1234 }], hardwareMetrics: { peakMemoryBytes: 1234567 },
    })),
  };
}

test('all fixed cases require completed native requests', () => {
  for (const spec of CASES) { const f = fixture(spec); validateResult(spec, f.log, 0, f.packets); }
});

test('dense, missing device, ignored and empty execution cannot pass', () => {
  const spec = CASES[0];
  for (const change of [
    f => { f.packets[0].runtimeFacts.materialization.rows[0].representation = 'dense-fallback'; },
    f => { f.packets[0].runtimeFacts.materialization.rows[0].tensorCount = 0; },
    f => { f.log = f.log.replace('candle-cuda', 'cpu'); },
    f => { f.log = f.log.replace('1 passed', '0 passed'); },
    f => { f.log = f.log.replace('0 ignored', '1 ignored'); },
    f => { f.packets.pop(); },
    f => { f.log += '{"event":"generator_cache_miss"}\n'; },
  ]) { const f = fixture(spec); change(f); assert.throws(() => validateResult(spec, f.log, 0, f.packets)); }
});

test('receipt identity, adapter, physical device and metrics stay bound', () => {
  const spec = CASES[0];
  for (const change of [
    f => { f.packets[0].modelId = CASES[1].model; },
    f => { f.packets[0].physicalGpu = '0'; },
    f => { f.packets[0].adapterCount = 0; },
    f => { f.packets[0].metrics = []; },
    f => { f.packets[0].hardwareMetrics = {}; },
    f => { f.packets[0].importPlan.checkpointId = 'another-source'; },
    f => { f.packets[0].importPlan.planId = 'not-a-compiled-plan'; },
  ]) { const f = fixture(spec); change(f); assert.throws(() => validateResult(spec, f.log, 0, f.packets)); }
});
