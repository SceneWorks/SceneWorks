import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import test from 'node:test';
import { candleChildSpawnOptions } from './flux3-image-klein-candle-app-lifecycle.mjs';

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
