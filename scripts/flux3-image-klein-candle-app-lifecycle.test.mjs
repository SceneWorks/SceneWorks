import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import test from 'node:test';
import { candleChildSpawnOptions, candleSavedAssetContext, readbackCandleAppLifecycle } from './flux3-image-klein-candle-app-lifecycle.mjs';

test('supervised Candle children retain piped stdin rather than immediate EOF', () => {
  assert.deepEqual(candleChildSpawnOptions({ SCENEWORKS_WORKER_CHILD: '1' }, 11).stdio, ['pipe', 11, 11]);
  assert.deepEqual(candleChildSpawnOptions({ SCENEWORKS_WORKER_CHILD: '' }, 11).stdio, ['pipe', 11, 11]);
  assert.deepEqual(candleChildSpawnOptions({}, 11).stdio, ['ignore', 11, 11]);
});

test('owned supervisor keeps stdin open until explicit cleanup', { timeout: 10_000 }, async (t) => {
  const env = { ...process.env, SCENEWORKS_WORKER_CHILD: '1' };
  const child = spawn(process.execPath, ['-e', "process.stdout.write('ready'); process.stdin.resume(); process.stdin.on('end', () => process.exit(0));"], candleChildSpawnOptions(env, 'pipe'));
  t.after(() => { if (child.exitCode === null && child.signalCode === null) child.kill(); });
  const exited = once(child, 'exit');
  assert.equal(String((await once(child.stdout, 'data'))[0]), 'ready');
  assert.equal(child.exitCode, null);
  assert.equal(child.stdin.writable, true);
  child.stdin.end();
  assert.deepEqual(await exited, [0, null]);
});

test('Candle saved-asset caller preserves existing physical job backend and distinct engine adapter', () => {
  const worker = { id: 'owned-worker', gpuId: '1', capabilities: ['gpu', 'candle'] };
  const expected = { projectId: 'owned-project', workerId: worker.id };
  assert.deepEqual(candleSavedAssetContext(worker, expected), { ...expected, backend: '1', adapter: 'candle_flux3' });
  assert.throws(() => candleSavedAssetContext({ ...worker, gpuId: '0' }, expected));
  assert.throws(() => candleSavedAssetContext({ ...worker, gpuId: 'candle' }, expected));
  assert.throws(() => candleSavedAssetContext({ ...worker, id: 'foreign-worker' }, expected));
  assert.throws(() => candleSavedAssetContext({ ...worker, capabilities: ['gpu'] }, expected));
});

test('readback continuation contains only GET requests with both tensor backends disabled', () => {
  const source = readbackCandleAppLifecycle.toString();
  assert.doesNotMatch(source, /startWorker|\/image\/jobs|\bPOST\b/);
  assert.match(source, /endpoint\.call\('GET'/);
  assert.match(source, /CUDA_VISIBLE_DEVICES: '-1'/);
  assert.match(source, /SCENEWORKS_BACKEND_CANDLE_ENABLED: 'false'/);
});
