// Real, bounded, ZERO-MODEL Windows CPU process tests. No CUDA/API/runner administration.
import assert from 'node:assert/strict';
import test from 'node:test';
import { spawn } from 'node:child_process';
import { mkdtemp, writeFile, readFile, rm } from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { prepareContainer, cleanupStatus } from './flux3-queue-container.mjs';
import { driverEnvironment } from './flux3-terminal-queue.mjs';
const windows = process.platform === 'win32', delay = ms => new Promise(r => setTimeout(r, ms));
const identity = { repository: 'SceneWorks/SceneWorks', runId: 123, attempt: 1, sourceSha: 'a'.repeat(40), inferenceMainSha: 'b'.repeat(40), controlSha256: 'c'.repeat(64), nonce: 'd'.repeat(64) };
const env = driverEnvironment(process.env, null);
async function fixture(t) { const dir = await mkdtemp(path.join(os.tmpdir(), 'flux3-owned-job-CPU-')); t.after(() => rm(dir, { recursive: true, force: true })); return dir; }
async function eventually(fn, ms = 15000) { const end = Date.now() + ms; while (Date.now() < end) { const r = await fn(); if (r) return r; await delay(50); } throw Error('CPU fixture deadline'); }
async function fileIfExists(f) { try { return JSON.parse(await readFile(f, 'utf8')); } catch (e) { if (e.code === 'ENOENT') return undefined; throw e; } }
function notAlive(pid) { try { process.kill(pid, 0); return false; } catch (e) { if (e.code === 'ESRCH') return true; throw e; } }
const containerModule = new URL('./flux3-queue-container.mjs', import.meta.url).href, queueModule = new URL('./flux3-terminal-queue.mjs', import.meta.url).href;

test('Windows: model parent exits leaving a real CPU grandchild; complete Job tree is killed before cleanup ACK', { skip: !windows, timeout: 60000 }, async t => {
  const dir = await fixture(t), pidFile = path.join(dir, 'grandchild.json'), script = path.join(dir, 'parent.mjs');
  await writeFile(script, `import {spawn} from 'node:child_process';import {writeFileSync} from 'node:fs';const c=spawn(process.execPath,['-e','setInterval(()=>{},1000)'],{stdio:'ignore'});writeFileSync(${JSON.stringify(pidFile)},JSON.stringify({pid:c.pid}));c.unref();process.exit(7);`);
  const c = await prepareContainer({ dir, identity, deadline: Date.now() + 20000, env });
  await c.start({ executable: process.execPath, args: [script], childEnv: env, cwd: dir, localLog: path.join(dir, 'parent.log') });
  assert.equal(await c.done, 7);
  const ack = await c.stop(); assert.equal(ack.activeProcesses, 0); assert.equal(ack.cleanupVerified, true);
  const { pid } = await fileIfExists(pidFile); await eventually(() => notAlive(pid));
});
test('Windows: nested Job assignment works, CPU child executes only after assignment, no inherited API credential', { skip: !windows, timeout: 60000 }, async t => {
  const dir = await fixture(t), script = path.join(dir, 'nested.mjs'), proof = path.join(dir, 'nested-proof.json');
  await writeFile(script, `import {prepareContainer} from ${JSON.stringify(containerModule)};import {driverEnvironment} from ${JSON.stringify(queueModule)};import {writeFile} from 'node:fs/promises';const env=driverEnvironment(process.env,null);const c=await prepareContainer({dir:${JSON.stringify(dir)},identity:${JSON.stringify(identity)},deadline:Date.now()+15000,env});await c.start({executable:process.execPath,args:['-e','if(process.env.GH_TOKEN||process.env.GITHUB_TOKEN||process.env.CUDA_VISIBLE_DEVICES!=="-1")process.exit(2)'],childEnv:env,cwd:${JSON.stringify(dir)},localLog:${JSON.stringify(path.join(dir, 'nested-child.log'))}});if(await c.done!==0)process.exit(3);const ack=await c.stop();await writeFile(${JSON.stringify(proof)},JSON.stringify({nested:true,activeProcesses:ack.activeProcesses}));`);
  const c = await prepareContainer({ dir, identity, deadline: Date.now() + 30000, env });
  await c.start({ executable: process.execPath, args: [script], childEnv: { ...env, GH_TOKEN: 'DECLARED-SYNTHETIC-NOT-A-CREDENTIAL' }, cwd: dir, localLog: path.join(dir, 'nested-parent.log') });
  assert.equal(await c.done, 0); await c.stop(); assert.deepEqual(await fileIfExists(proof), { nested: true, activeProcesses: 0 });
});
test('Windows: unexpected executor-supervisor death independently kills its CPU child; kernel zero is verified before release', { skip: !windows, timeout: 60000 }, async t => {
  const dir = await fixture(t), script = path.join(dir, 'supervisor.mjs'), definitionFile = path.join(dir, 'definition.json'), pidFile = path.join(dir, 'child.json');
  await writeFile(script, `import {prepareContainer} from ${JSON.stringify(containerModule)};import {driverEnvironment} from ${JSON.stringify(queueModule)};import {writeFile,readFile} from 'node:fs/promises';const env=driverEnvironment(process.env,null);const c=await prepareContainer({dir:${JSON.stringify(dir)},identity:${JSON.stringify(identity)},deadline:Date.now()+15000,env});await writeFile(${JSON.stringify(definitionFile)},JSON.stringify(c.definition));await c.start({executable:process.execPath,args:['-e',${JSON.stringify(`require('fs').writeFileSync(${JSON.stringify(pidFile)},JSON.stringify({pid:process.pid}));setInterval(()=>{},1000)`)}],childEnv:env,cwd:${JSON.stringify(dir)},localLog:${JSON.stringify(path.join(dir, 'child.log'))}});while(true){try{await readFile(${JSON.stringify(pidFile)});break}catch{await new Promise(r=>setTimeout(r,25))}}process.exit(9);`);
  const parent = spawn(process.execPath, [script], { env, stdio: 'ignore' });
  const code = await new Promise((resolve, reject) => { parent.once('error', reject); parent.once('exit', resolve); }); assert.equal(code, 9);
  const definition = await fileIfExists(definitionFile), child = await fileIfExists(pidFile);
  const ack = await eventually(() => cleanupStatus(definition, identity)); assert.equal(ack.activeProcesses, 0);
  await eventually(() => notAlive(child.pid)); await eventually(() => notAlive(definition.helper.pid));
});
test('Windows: fixed container deadline performs cleanup instead of simply releasing a still-running child', { skip: !windows, timeout: 60000 }, async t => {
  const dir = await fixture(t), c = await prepareContainer({ dir, identity, deadline: Date.now() + 15000, env });
  await c.start({ executable: process.execPath, args: ['-e', 'setInterval(()=>{},1000)'], childEnv: env, cwd: dir, localLog: path.join(dir, 'deadline.log') });
  assert.equal(await c.done, 1); const ack = await c.stop(); assert.equal(ack.activeProcesses, 0); assert(ack.childPid > 0, 'deadline cleaned a running child, not an empty container');
});
