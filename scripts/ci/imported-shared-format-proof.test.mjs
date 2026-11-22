// Metadata only: no model, tensor, CUDA or subprocess execution.
import assert from 'node:assert/strict';
import test from 'node:test';
import { CASES, guardedCases, validateResult, validateSdxlReferencePair } from './imported-shared-format-proof.mjs';

function fixture(spec) {
  const fqn = `imported_comfy_quant_gpu_smoke::${spec.test}`;
  return {
    log: `${Array(spec.count).fill('RESULT status=pass native request').join('\n')}\n`
      + `MODEL_INFERENCE_BODY_COMPLETED ${fqn}\nMODEL_INFERENCE_DEVICE ${fqn} candle-cuda\n`
      + 'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 100 filtered out;\n'
      + Array.from({ length: spec.count }, (_, iteration) => JSON.stringify({ fields: {
        event: 'image_pipeline_load_complete', sw_payload: JSON.stringify({
          jobId: `comfy-quant-image-generate-${iteration}`, engine: spec.provider, adapterCount: spec.adapters,
        }),
      } })).join('\n') + '\n'
      + (spec.count === 2 ? '{"fields":{"event":"generator_cache_miss"}}\n' : ''),
    packets: Array.from({ length: spec.count }, (_, iteration) => ({
      schemaVersion: 1, modelId: spec.model, iteration, sourceCodec: spec.codec, physicalGpu: '0',
      route: spec.route, provider: spec.provider,
      checkpointId: 'linked/root-1234567890abcdef/model.safetensors',
      importPlan: { checkpointId: 'linked/root-1234567890abcdef/model.safetensors',
        planId: `checkpoint-plan-${'a'.repeat(64)}`, semanticDigest: `sha256:${'b'.repeat(64)}` },
      adapterCount: spec.adapters, width: spec.width, height: spec.height, steps: spec.steps, seed: 42, output: `${spec.model}-${iteration}.png`,
      runtimeFacts: { materialization: { status: 'reported', rows: [{ codecId: spec.codec, representation: 'native-packed', tensorCount: 3 }] } },
      metrics: [{
        loadMs: 1234, sourceCodec: spec.codec,
        ...(spec.phaseExecution && { executionRepresentation: spec.phaseExecution }),
        ...(spec.group === 'nvfp4' && { quantLabel: 'nvfp4' }),
      }], hardwareMetrics: { peakMemoryBytes: 1234567 },
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
    f => { f.packets[0].route = 'candle_checkpoint_plan'; },
    f => { f.packets[0].provider = 'another-provider'; },
    f => { f.log = f.log.replace('image_pipeline_load_complete', 'image_pipeline_load_start'); },
    f => { f.log = f.log.replaceAll('krea_2_turbo', 'another-provider'); },
    f => { f.log = f.log.replaceAll('adapterCount\\":1', 'adapterCount\\":0'); },
    f => { f.packets[0].physicalGpu = '1'; },
    f => { f.packets[0].adapterCount = 0; },
    f => { f.packets[0].metrics = []; },
    f => { f.packets[0].hardwareMetrics = {}; },
    f => { f.packets[0].hardwareMetrics.peakMemoryBytes = 0; },
    f => { f.packets[0].importPlan.checkpointId = 'another-source'; },
    f => { f.packets[0].importPlan.planId = 'not-a-compiled-plan'; },
  ]) { const f = fixture(spec); change(f); assert.throws(() => validateResult(spec, f.log, 0, f.packets)); }
});

test('FP8 phase metrics report source-owned precision and measured Muse execution', () => {
  for (const spec of CASES.filter(spec => spec.group === 'fp8')) {
    for (const [key, value, error] of [
      ['sourceCodec', 'int8-per-row-v1', /FP8 source codec/],
      ['quantLabel', 'q8', /no requested quant tier/],
      ['quantBits', 8, /no requested quant bits/],
    ]) {
      const f = fixture(spec);
      f.packets[0].metrics[0][key] = value;
      assert.throws(() => validateResult(spec, f.log, 0, f.packets), error);
    }
  }

  const muse = CASES.find(spec => spec.model === 'linked_krea2_scalar_fp8');
  for (const executionRepresentation of [undefined, 'dense-fallback', 'mixed:2/1']) {
    const f = fixture(muse);
    f.packets[1].metrics[0].executionRepresentation = executionRepresentation;
    assert.throws(() => validateResult(muse, f.log, 0, f.packets), /report measured execution/);
  }

  const sdxl = CASES.find(spec => spec.model === 'linked_sdxl_fp8');
  const f = fixture(sdxl);
  Object.assign(f.packets[0].metrics[0], {
    quantLabel: null, quantBits: null, executionRepresentation: 'mixed:2/1',
  });
  validateResult(sdxl, f.log, 0, f.packets);
});

const guardedReference = {
  GITHUB_EVENT_NAME: 'workflow_dispatch',
  SCENEWORKS_GPU_ID: '0',
  SW_CI_CUDA_EXECUTION_ACTIVE: '230',
  SHARED_FORMAT_PROFILE: 'sdxl-fp8-reference',
};

test('SDXL reference selects only its two cases under the physical guard', () => {
  const [native, reference] = guardedCases(guardedReference);
  assert.equal(native, CASES[1]);
  assert.equal(reference.model, native.model);
  assert.equal(reference.test, 'imported_sdxl_fp8_dense_reference');
  assert.deepEqual(guardedCases({ ...guardedReference, SHARED_FORMAT_PROFILE: 'all' }), CASES);
  assert.deepEqual(guardedCases({ ...guardedReference, SHARED_FORMAT_PROFILE: 'fp8' }), CASES.slice(0, 2));
  for (const override of [
    { GITHUB_EVENT_NAME: 'pull_request' },
    { SCENEWORKS_GPU_ID: '1' },
    { SW_CI_CUDA_EXECUTION_ACTIVE: '' },
    { SW_CI_CUDA_EXECUTION_ACTIVE: '0' },
    { SHARED_FORMAT_PROFILE: 'dense' },
  ]) assert.throws(() => guardedCases({ ...guardedReference, ...override }));
});

test('reference permits the three identical CUDA observations emitted by the native run', () => {
  const [native] = guardedCases(guardedReference);
  const { log, packets } = fixture(native);
  const fqn = 'imported_comfy_quant_gpu_smoke::imported_sdxl_fp8_worker_gpu_smoke';
  const cuda = `MODEL_INFERENCE_DEVICE ${fqn} candle-cuda`;
  const repeated = log.replace(cuda, [cuda, cuda, cuda].join('\n'));
  validateResult(native, repeated, 0, packets, true);
  assert.throws(() => validateResult(native, repeated, 0, packets), /CUDA device evidence/);

  const missing = log.replace(cuda, '');
  const mixed = repeated + `MODEL_INFERENCE_DEVICE ${fqn} candle-cpu\n`;
  for (const invalid of [missing, mixed]) {
    assert.throws(() => validateResult(native, invalid, 0, packets, true), /CUDA device evidence/);
  }

  assert.throws(() => validateResult(native, repeated + `MODEL_INFERENCE_BODY_COMPLETED ${fqn}\n`, 0, packets, true));
});

test('reference receipt requires only dense E5 and a witnessed CUDA device', () => {
  const [native, reference] = guardedCases(guardedReference);
  const f = fixture(reference);
  const row = f.packets[0].runtimeFacts.materialization.rows[0];
  assert.throws(() => validateResult(reference, f.log, 0, f.packets), /only measured dense E5/);
  row.representation = 'dense-fallback';
  validateResult(reference, f.log, 0, f.packets);
  const g = fixture(native);
  g.packets[0].runtimeFacts.materialization.rows[0].representation = 'dense-fallback';
  assert.throws(() => validateResult(native, g.log, 0, g.packets), /measured native/);
  const mixedDevice = f.log + 'MODEL_INFERENCE_DEVICE imported_comfy_quant_gpu_smoke::imported_sdxl_fp8_dense_reference candle-cpu\n';
  assert.throws(() => validateResult(reference, mixedDevice, 0, f.packets));
  f.packets[0].runtimeFacts.materialization.rows.push({ ...row, representation: 'native-packed' });
  assert.throws(() => validateResult(reference, f.log, 0, f.packets), /only measured dense E5/);
});

test('reference comparison binds the exact original file and frozen controls', () => {
  const [native] = guardedCases(guardedReference);
  const packet = fixture(native).packets[0];
  packet.runtimeFacts.sourceBinding = {
    stableToken: 'cd9ee36e57a438aa89ba4b47b5c0703d0cd9bc6b27b871072e7057ea8298fb68@3552885118',
    sizeBytes: 3552885118,
  };
  packet.runtimeFacts.source = [{ codecId: native.codec, tensorCount: 2393, sourceBytes: 3385184004 }];
  packet.requestControls = {
    mode: 'text_to_image', prompt: 'a red fox in a sunlit autumn forest, sharp focus',
    negativePrompt: '', advanced: { steps: 20 }, loras: [],
  };
  Object.assign(packet.metrics[0], { sampler: 'default', scheduler: 'default', guidanceMethod: 'cfg', usePid: false });
  validateSdxlReferencePair(packet, structuredClone(packet));
  for (const mismatch of [
    p => { p.runtimeFacts.sourceBinding.sizeBytes += 1; },
    p => { p.runtimeFacts.source[0].tensorCount -= 1; },
    p => { p.importPlan.planId = 'a different plan'; },
    p => { p.requestControls.prompt = 'a different prompt'; },
    p => { p.requestControls.advanced.guidanceScale = 6; },
    p => { p.metrics[0].sampler = 'heun'; },
    p => { p.metrics[0].scheduler = 'karras'; },
  ]) {
    const changed = structuredClone(packet);
    mismatch(changed);
    assert.throws(() => validateSdxlReferencePair(packet, changed));
  }
});
