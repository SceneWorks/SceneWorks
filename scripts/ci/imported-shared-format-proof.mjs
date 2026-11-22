// Extends the existing manual imported-checkpoint validation; never called by ordinary CI.
import assert from 'node:assert/strict';
import { spawn, execFileSync } from 'node:child_process';
import { lstat, mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { fileSha256 } from '../lib/file-sha256.mjs';
import { lockedInferenceRevision } from '../select-windows-cargo-cache.mjs';
import { redact } from './imported-nvfp4-proof.mjs';

export const CASES = Object.freeze([
  { group: 'fp8', test: 'imported_scalar_fp8_worker_gpu_smoke', model: 'linked_krea2_scalar_fp8', codec: 'fp8-e4m3-scalar-v1', route: 'candle_krea_imported', provider: 'krea_2_turbo', count: 2, width: 1408, height: 2048, steps: 8, adapters: 1, phaseExecution: 'native-packed' },
  { group: 'fp8', test: 'imported_sdxl_fp8_worker_gpu_smoke', model: 'linked_sdxl_fp8', codec: 'fp8-e5m2-scalar-v1', route: 'candle_checkpoint_plan', provider: 'sdxl', count: 1, width: 512, height: 512, steps: 20, adapters: 0 },
  { group: 'convrot', test: 'imported_convrot_worker_gpu_smoke', model: 'linked_krea2_convrot', codec: 'convrot-int8-native', route: 'candle_checkpoint_plan', provider: 'krea_2_turbo', count: 1, width: 512, height: 512, steps: 8, adapters: 0 },
  { group: 'nvfp4', test: 'imported_native_nvfp4_worker_gpu_smoke', model: 'linked_krea2_native_nvfp4', codec: 'nvfp4-v1', route: 'candle_checkpoint_plan', provider: 'krea_2_turbo', count: 1, width: 512, height: 512, steps: 8, adapters: 0 },
]);

const SDXL_REFERENCE_PROFILE = 'sdxl-fp8-reference';
const SDXL_FP8_NATIVE = CASES.find(spec => spec.model === 'linked_sdxl_fp8');
const SDXL_FP8_DENSE_REFERENCE = Object.freeze({
  ...SDXL_FP8_NATIVE, test: 'imported_sdxl_fp8_dense_reference', phaseExecution: 'dense-fallback',
});

export function guardedCases(environment) {
  assert.equal(environment.GITHUB_EVENT_NAME, 'workflow_dispatch');
  assert.equal(environment.SCENEWORKS_GPU_ID, '0');
  assert(/^[1-9][0-9]*$/.test(environment.SW_CI_CUDA_EXECUTION_ACTIVE ?? ''),
    'physical guard must own the child');
  const profile = environment.SHARED_FORMAT_PROFILE;
  assert(['all', 'fp8', 'convrot', 'nvfp4', SDXL_REFERENCE_PROFILE].includes(profile));
  return profile === SDXL_REFERENCE_PROFILE
    ? [SDXL_FP8_NATIVE, SDXL_FP8_DENSE_REFERENCE]
    : CASES.filter(spec => profile === 'all' || spec.group === profile);
}

export function validateSdxlReferencePair(native, reference) {
  assert.deepEqual(native.runtimeFacts.sourceBinding, {
    stableToken: 'cd9ee36e57a438aa89ba4b47b5c0703d0cd9bc6b27b871072e7057ea8298fb68@3552885118',
    sizeBytes: 3552885118,
  }, 'normal and dense runs must use the original E5 checkpoint');
  assert.deepEqual(native.runtimeFacts.sourceBinding, reference.runtimeFacts.sourceBinding);
  assert.deepEqual(native.runtimeFacts.source, reference.runtimeFacts.source);
  assert.deepEqual(native.importPlan, reference.importPlan, 'both processes must open the same plan');
  assert.deepEqual(native.requestControls, {
    mode: 'text_to_image',
    prompt: 'a red fox in a sunlit autumn forest, sharp focus',
    negativePrompt: '',
    advanced: { steps: 20 },
    loras: [],
  });
  assert.deepEqual(native.requestControls, reference.requestControls, 'request controls must match');
  for (const packet of [native, reference]) {
    const phase = packet.metrics.find(row => Number.isFinite(row.loadMs));
    // Omitted SDXL controls resolve to DDIM, the normal sigma schedule and CFG 7 at this pin.
    assert.equal(phase.sampler, 'default');
    assert.equal(phase.scheduler, 'default');
    assert.equal(phase.guidanceMethod, 'cfg');
    assert.equal(phase.usePid, false);
  }
}

export function validateResult(spec, log, code, packets, proofPair = false) {
  assert.equal(code, 0, 'native worker test must succeed');
  assert.equal((log.match(/^test result: ok\. 1 passed; 0 failed; 0 ignored;/gm) ?? []).length, 1);
  const fqn = `imported_comfy_quant_gpu_smoke::${spec.test}`;
  assert.equal((log.match(new RegExp(`^MODEL_INFERENCE_BODY_COMPLETED ${fqn}$`, 'gm')) ?? []).length, 1);
  const devices = log.match(new RegExp(`^MODEL_INFERENCE_DEVICE ${fqn} .+$`, 'gm')) ?? [];
  // Independent proof observers may each witness the same device. The manual marker is singular.
  assert.deepEqual(proofPair ? [...new Set(devices)] : devices,
    [`MODEL_INFERENCE_DEVICE ${fqn} candle-cuda`], 'expected only in-scope CUDA device evidence');
  assert.equal((log.match(/^RESULT status=pass /gm) ?? []).length, spec.count);
  assert.equal(packets.length, spec.count);
  const loadEvents = log.split('\n').flatMap(line => {
    try {
      const fields = JSON.parse(line).fields;
      return fields?.event === 'image_pipeline_load_complete' ? [JSON.parse(fields.sw_payload)] : [];
    } catch { return []; }
  });
  assert.equal(loadEvents.length, spec.count, 'each completed request must identify its actual loaded provider');
  for (const [iteration, packet] of packets.entries()) {
    assert.equal(packet.schemaVersion, 1);
    assert.equal(packet.modelId, spec.model);
    assert.equal(packet.route, spec.route);
    assert.equal(packet.provider, spec.provider);
    assert.equal(loadEvents[iteration].jobId, `comfy-quant-image-generate-${iteration}`);
    assert.equal(loadEvents[iteration].engine, spec.provider);
    assert.equal(loadEvents[iteration].adapterCount, spec.adapters);
    assert.equal(packet.iteration, iteration);
    assert.equal(packet.sourceCodec, spec.codec);
    assert(/^linked\/root-[a-f0-9]{16}\/.+\.safetensors$/.test(packet.checkpointId), 'actual approved linked source required');
    assert.equal(packet.importPlan?.checkpointId, packet.checkpointId);
    assert(/^checkpoint-plan-[a-f0-9]{64}$/.test(packet.importPlan?.planId), 'actual compiled plan required');
    assert(/^sha256:[a-f0-9]{64}$/.test(packet.importPlan?.semanticDigest), 'actual semantic commitment required');
    assert.equal(packet.physicalGpu, '0');
    assert.equal(packet.adapterCount, spec.adapters);
    assert.equal(packet.width, spec.width);
    assert.equal(packet.height, spec.height);
    assert.equal(packet.steps, spec.steps);
    assert.equal(packet.seed, 42);
    assert.equal(packet.output, `${spec.model}-${iteration}.png`);
    assert.equal(packet.runtimeFacts?.materialization?.status, 'reported');
    const codecRows = packet.runtimeFacts.materialization.rows.filter(row => row.codecId === spec.codec);
    if (spec === SDXL_FP8_DENSE_REFERENCE) {
      assert(codecRows.length > 0 && codecRows.every(row =>
        row.representation === 'dense-fallback' && row.tensorCount > 0), 'only measured dense E5 materialization required');
    } else {
      assert(codecRows.some(row => row.representation === 'native-packed' && row.tensorCount > 0),
        'measured native materialization required');
    }
    const phase = packet.metrics.find(row => Number.isFinite(row.loadMs));
    assert(phase, 'request load timing required');
    if (spec.group === 'fp8') {
      assert.equal(phase.sourceCodec, spec.codec, 'phase metrics must identify the FP8 source codec');
      assert.equal(phase.quantLabel ?? null, null, 'FP8 source must have no requested quant tier');
      assert.equal(phase.quantBits ?? null, null, 'FP8 source must have no requested quant bits');
    }
    if (spec.phaseExecution) {
      assert.equal(phase.executionRepresentation, spec.phaseExecution, 'phase metrics must report measured execution');
    }
    if (spec.group === 'nvfp4') {
      assert.equal(phase.quantLabel, 'nvfp4', 'phase metrics must retain the NVFP4 tier identity');
      assert.equal(phase.quantBits ?? null, null, 'NVFP4 has no integral quant width');
    }
    assert(Number.isFinite(packet.hardwareMetrics?.peakMemoryBytes) && packet.hardwareMetrics.peakMemoryBytes > 0,
      'positive sampled physical memory required');
  }
  if (spec.count === 2) {
    assert.equal((log.match(/"event":"generator_cache_miss"/g) ?? []).length, 1,
      'identical cold/warm requests must load the generator once');
  }
}

async function boundedFile(directory, name) {
  const file = path.join(directory, name), info = await lstat(file);
  assert(info.isFile() && !info.isSymbolicLink() && info.nlink === 1);
  assert(info.size > 0 && info.size <= 64 * 1024 * 1024);
  return { name, bytes: info.size, sha256: await fileSha256(file) };
}

async function execute(binary, spec, output, proofPair) {
  const fqn = `imported_comfy_quant_gpu_smoke::${spec.test}`;
  const log = await new Promise((resolve, reject) => {
    const child = spawn(binary, [fqn, '--exact', '--ignored', '--nocapture', '--test-threads=1'],
      { env: {
        ...process.env, MODEL_PROOF_OUTPUT_DIR: output,
        ...(proofPair && { MODEL_INFERENCE_CASE_FQN: fqn }),
      }, windowsHide: true });
    let text = '', size = 0;
    const append = chunk => { size += chunk.length; if (size <= 16 * 1024 * 1024) text += chunk.toString(); };
    child.stdout.on('data', append); child.stderr.on('data', append);
    child.on('error', reject);
    child.on('close', code => { try { assert(size <= 16 * 1024 * 1024, 'bounded log exceeded'); resolve({ text, code }); } catch (error) { reject(error); } });
  });
  const safeLog = redact(log.text);
  await writeFile(path.join(output, `${spec.model}.log`), safeLog, { flag: 'wx' });
  assert.equal(log.code, 0, `worker failed; inspect ${spec.model}.log`);
  const packets = [];
  for (let iteration = 0; iteration < spec.count; ++iteration) {
    const filename = `${spec.model}-${iteration}.json`;
    await boundedFile(output, filename);
    packets.push(JSON.parse(await readFile(path.join(output, filename), 'utf8')));
  }
  validateResult(spec, safeLog, log.code, packets, proofPair);
  const files = [];
  for (const packet of packets) {
    const imageFile = await boundedFile(output, packet.output);
    const png = await readFile(path.join(output, packet.output));
    assert.equal(png.subarray(0, 8).toString('hex'), '89504e470d0a1a0a');
    assert.equal(png.readUInt32BE(16), spec.width); assert.equal(png.readUInt32BE(20), spec.height);
    files.push(imageFile, await boundedFile(output, `${spec.model}-${packet.iteration}.json`));
  }
  files.push(await boundedFile(output, `${spec.model}.log`));
  return { packets, evidence: { test: fqn, sourceCodec: spec.codec, generations: packets.length, files,
    timings: packets.map(p => ({ iteration: p.iteration, phases: p.metrics, hardware: p.hardwareMetrics })) } };
}

async function main() {
  const cases = guardedCases(process.env);
  const profile = process.env.SHARED_FORMAT_PROFILE;
  const proofPair = profile === SDXL_REFERENCE_PROFILE;
  const binary = path.resolve(process.env.SHARED_FORMAT_TEST_BINARY);
  const output = path.resolve(process.env.MODEL_PROOF_OUTPUT_DIR);
  const relative = path.relative(path.resolve(process.env.RUNNER_TEMP), output);
  assert(relative && !relative.startsWith('..') && !path.isAbsolute(relative));
  assert(!/^e:/i.test(output));
  const git = args => execFileSync('git', args, { encoding: 'utf8' }).trim();
  assert.equal(git(['status', '--porcelain', '--untracked-files=no']), '', 'clean source required');
  const source = git(['rev-parse', 'HEAD']); assert.equal(source, process.env.GITHUB_SHA);
  const pin = lockedInferenceRevision(await readFile('Cargo.lock', 'utf8'));
  const binaryIdentity = await fileSha256(binary);
  await mkdir(output, { recursive: false });
  const receipt = { schemaVersion: 1, status: 'running', source, inference: pin,
    binarySha256: binaryIdentity, profile, deviceProvenance: proofPair
      ? 'observed-learned-tensor-and-physical-UUID-guard'
      : 'source-enforced-native-materialization-and-physical-UUID-guard', cases: [] };
  try {
    const pair = [];
    for (const spec of cases) {
      const directory = proofPair && (spec === SDXL_FP8_DENSE_REFERENCE ? 'dense-reference' : 'native');
      const destination = directory ? path.join(output, directory) : output;
      if (directory) await mkdir(destination);
      const { evidence, packets } = await execute(binary, spec, destination, proofPair);
      if (directory) {
        evidence.directory = directory;
        pair.push(packets[0]);
      }
      receipt.cases.push(evidence);
    }
    if (proofPair) validateSdxlReferencePair(...pair);
    assert.equal(git(['rev-parse', 'HEAD']), source);
    assert.equal(git(['status', '--porcelain', '--untracked-files=no']), '');
    assert.equal(await fileSha256(binary), binaryIdentity);
    receipt.status = 'passed';
  } catch (error) { receipt.status = 'failed'; receipt.error = redact(String(error)); throw error; }
  finally { await writeFile(path.join(output, 'receipt.json'), JSON.stringify(receipt, null, 2)); }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch(error => { console.error(error.message); process.exitCode = 1; });
}
