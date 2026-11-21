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
  { group: 'fp8', test: 'imported_scalar_fp8_worker_gpu_smoke', model: 'linked_krea2_scalar_fp8', codec: 'fp8-e4m3-scalar-v1', count: 2, width: 1408, height: 2048, steps: 8, adapters: 1 },
  { group: 'fp8', test: 'imported_sdxl_fp8_worker_gpu_smoke', model: 'linked_sdxl_fp8', codec: 'fp8-e5m2-scalar-v1', count: 1, width: 512, height: 512, steps: 20, adapters: 0 },
  { group: 'convrot', test: 'imported_convrot_worker_gpu_smoke', model: 'linked_krea2_convrot', codec: 'convrot-int8-native', count: 1, width: 512, height: 512, steps: 8, adapters: 0 },
  { group: 'nvfp4', test: 'imported_native_nvfp4_worker_gpu_smoke', model: 'linked_krea2_native_nvfp4', codec: 'nvfp4-v1', count: 1, width: 512, height: 512, steps: 8, adapters: 0 },
]);

export function validateResult(spec, log, code, packets) {
  assert.equal(code, 0, 'native worker test must succeed');
  assert.equal((log.match(/^test result: ok\. 1 passed; 0 failed; 0 ignored;/gm) ?? []).length, 1);
  const fqn = `imported_comfy_quant_gpu_smoke::${spec.test}`;
  assert.equal((log.match(new RegExp(`^MODEL_INFERENCE_BODY_COMPLETED ${fqn}$`, 'gm')) ?? []).length, 1);
  assert.equal((log.match(new RegExp(`^MODEL_INFERENCE_DEVICE ${fqn} candle-cuda$`, 'gm')) ?? []).length, 1);
  assert.equal((log.match(/^RESULT status=pass /gm) ?? []).length, spec.count);
  assert.equal(packets.length, spec.count);
  for (const [iteration, packet] of packets.entries()) {
    assert.equal(packet.schemaVersion, 1);
    assert.equal(packet.modelId, spec.model);
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
    assert(packet.runtimeFacts.materialization.rows.some(row => row.codecId === spec.codec
      && row.representation === 'native-packed' && row.tensorCount > 0), 'measured native materialization required');
    assert(packet.metrics.some(row => Number.isFinite(row.loadMs)), 'request load timing required');
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

async function execute(binary, spec, output) {
  const fqn = `imported_comfy_quant_gpu_smoke::${spec.test}`;
  const log = await new Promise((resolve, reject) => {
    const child = spawn(binary, [fqn, '--exact', '--ignored', '--nocapture', '--test-threads=1'],
      { env: { ...process.env, MODEL_PROOF_OUTPUT_DIR: output }, windowsHide: true });
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
  validateResult(spec, safeLog, log.code, packets);
  const files = [];
  for (const packet of packets) {
    const imageFile = await boundedFile(output, packet.output);
    const png = await readFile(path.join(output, packet.output));
    assert.equal(png.subarray(0, 8).toString('hex'), '89504e470d0a1a0a');
    assert.equal(png.readUInt32BE(16), spec.width); assert.equal(png.readUInt32BE(20), spec.height);
    files.push(imageFile, await boundedFile(output, `${spec.model}-${packet.iteration}.json`));
  }
  files.push(await boundedFile(output, `${spec.model}.log`));
  return { test: fqn, sourceCodec: spec.codec, generations: packets.length, files,
    timings: packets.map(p => ({ iteration: p.iteration, phases: p.metrics, hardware: p.hardwareMetrics })) };
}

async function main() {
  assert.equal(process.env.GITHUB_EVENT_NAME, 'workflow_dispatch');
  assert.equal(process.env.SCENEWORKS_GPU_ID, '0');
  assert(/^[1-9][0-9]*$/.test(process.env.SW_CI_CUDA_EXECUTION_ACTIVE ?? ''), 'physical guard must own the child');
  const profile = process.env.SHARED_FORMAT_PROFILE;
  assert(['all', 'fp8', 'convrot', 'nvfp4'].includes(profile));
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
    binarySha256: binaryIdentity, profile, deviceProvenance: 'source-enforced-native-materialization-and-physical-UUID-guard', cases: [] };
  try {
    for (const spec of CASES.filter(c => profile === 'all' || c.group === profile)) receipt.cases.push(await execute(binary, spec, output));
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
