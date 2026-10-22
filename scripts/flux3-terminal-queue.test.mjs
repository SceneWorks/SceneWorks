// DECLARED SYNTHETIC CPU tests: no device, model, runner job, API call, or campaign permit.
import assert from 'node:assert/strict';
import test from 'node:test';
import { generateKeyPairSync } from 'node:crypto';
import { spawn } from 'node:child_process';
import { readFile, mkdtemp, rm, mkdir, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { SLOTS, EXECUTOR, STALE_MS, API_FRESH_MS, queueInputs, electedRole, validateApi, ancestors, workerIdentity, signed, authenticated, verifyRoster, driverEnvironment, superviseChild, verifyQueueReservation, publishNew, holdUntilSafe, verifyObservedRoster } from './flux3-terminal-queue.mjs';
import { executionGroups, assertGroupPassed } from './flux3-terminal-queue-campaign.mjs';
import { loadProductTargets, terminalPlan } from './flux3-image-klein-terminal-plan.mjs';
import { guardHardware } from './flux3-image-klein-terminal-session.mjs';
const source = 'a'.repeat(40), inference = 'b'.repeat(40), digest = 'c'.repeat(64), NOW = 1000000;
function fixture() {
  const identity = { repository: 'SceneWorks/SceneWorks', runId: 123, attempt: 1, sourceSha: source, inferenceMainSha: inference, controlSha256: digest, nonce: 'd'.repeat(64) };
  const run = { id: 123, run_attempt: 1, head_sha: source, repository: { full_name: identity.repository }, event: 'workflow_dispatch', status: 'in_progress', path: '.github/workflows/memory-catalog-campaign.yml' };
  const jobs = [], table = [], members = {}, pulses = {}, keys = {};
  for (const [n, [runner, slot]] of Object.entries(SLOTS).entries()) {
    const pid = (n + 1) * 100, jobId = n + 1000;
    table.push({ pid, ppid: 0, startedAt: `worker-start-${n}`, executable: slot.worker, command: slot.worker });
    table.push({ pid: pid + 1, ppid: pid, startedAt: `supervisor-start-${n}`, executable: 'C:/node/node.exe', command: 'node.exe' });
    jobs.push({ id: jobId, name: `flux3-terminal-queue / reserve (${4 - n})`, run_id: 123, head_sha: source, status: 'in_progress', runner_name: runner, runner_id: slot.id });
    keys[runner] = generateKeyPairSync('ed25519');
    const process = p => ({ pid: p.pid, ppid: p.ppid, startedAt: p.startedAt, executable: p.executable });
    members[runner] = { identity: structuredClone(identity), runner, role: electedRole(runner), hostName: 'DECLARED-WINDOWS-HOST', jobId, worker: process(table.at(-2)), supervisor: process(table.at(-1)), publicKey: keys[runner].publicKey.export({ type: 'spki', format: 'pem' }) };
    pulses[runner] = signed({ identity: structuredClone(identity), runner, jobId, at: NOW, apiValidUntil: NOW + API_FRESH_MS, sequence: 1, status: 'holding' }, keys[runner].privateKey);
  }
  table.push({ pid: 110, ppid: 101, startedAt: 'driver-start', executable: 'C:/node/node.exe', command: 'node.exe' });
  const binding = { kind: 'flux3-four-slot-reservation', identity, members, deadline: NOW + 100000, api: { run, jobs, observedAt: NOW } };
  return { binding, pulses, table, keys, controllerPid: 110 };
}
const valid = f => verifyRoster(f.binding, f.pulses, f.table, f.controllerPid, NOW);
const env = overrides => ({ GITHUB_REPOSITORY: 'SceneWorks/SceneWorks', GITHUB_EVENT_NAME: 'workflow_dispatch', GITHUB_SHA: source, GITHUB_RUN_ID: '123', GITHUB_RUN_ATTEMPT: '1', FLUX3_QUEUE_SOURCE: source, FLUX3_QUEUE_INFERENCE: inference, FLUX3_QUEUE_CONTROL_SHA256: digest, ...overrides });

test('typed manual queue inputs reject empty/default sentinels, wrong source/event/repo/attempt', () => {
  assert.equal(queueInputs(env()).runId, 123);
  for (const value of ['none', '', undefined]) assert.throws(() => queueInputs(env({ FLUX3_QUEUE_CONTROL_SHA256: value })), /sentinel/);
  for (const over of [{ GITHUB_EVENT_NAME: 'push' }, { GITHUB_REPOSITORY: 'SceneWorks/inference' }, { GITHUB_SHA: 'e'.repeat(40) }, { GITHUB_RUN_ATTEMPT: '0' }, { FLUX3_QUEUE_SOURCE: 'main' }]) assert.throws(() => queueInputs(env(over)));
});
test('homogeneous matrix elects by actual runner, not matrix slot; exact org + repository scoped roster', () => {
  assert.equal(electedRole('cuda-windows'), 'executor');
  for (const runner of ['cuda-windows-2', 'cuda-windows-3', 'cuda-windows-4']) assert.equal(electedRole(runner), 'holder');
  assert.throws(() => electedRole('cuda-windows-5'), /unrecognized/);
  assert.deepEqual(Object.values(SLOTS).map(s => s.id), [2313, 2619, 25, 26]);
  assert.deepEqual(valid(fixture()), [200, 300, 400], 'only the three validated companions are additional allowed Workers');
});
test('API proof requires exact repo/run/attempt/source, four live job identities and actual runner IDs', () => {
  const f = fixture(); assert.equal(Object.keys(validateApi(f.binding.api.run, f.binding.api.jobs, f.binding.identity)).length, 4);
  for (const mutate of [
    x => { x.binding.api.run.repository.full_name = 'other/repo'; }, x => { x.binding.api.run.id++; }, x => { x.binding.api.run.run_attempt++; },
    x => { x.binding.api.run.head_sha = 'f'.repeat(40); }, x => { x.binding.api.run.path = '.github/workflows/real-weights.yml'; }, x => { x.binding.api.run.event = 'push'; },
    x => { x.binding.api.jobs[1].runner_id = 999; }, x => { x.binding.api.jobs[1].runner_name = 'cuda-windows-3'; }, x => { x.binding.api.jobs[1].head_sha = 'f'.repeat(40); },
    x => { x.binding.api.jobs[1].run_id++; }, x => { x.binding.api.jobs[1].id = x.binding.api.jobs[0].id; }, x => { x.binding.api.jobs[1].name = x.binding.api.jobs[0].name; },
    x => { x.binding.api.jobs[1].status = 'completed'; }, x => { x.binding.api.jobs[1].status = 'queued'; }, x => { x.binding.api.jobs.pop(); },
  ]) { const x = fixture(); mutate(x); assert.throws(() => valid(x)); }
});
test('barrier refuses missing, wrong host/name/source/nonce, duplicate identity, stale or forged heartbeat and any fifth Worker', () => {
  for (const [label, mutate] of [
    ['missing member', f => { delete f.binding.members['cuda-windows-4']; }],
    ['third machine', f => { f.binding.members['cuda-windows-3'].hostName = 'OTHER-HOST'; }],
    ['wrong runner', f => { f.binding.members['cuda-windows-4'].runner = 'cuda-windows-3'; }],
    ['wrong source', f => { f.binding.members['cuda-windows-2'].identity.sourceSha = 'f'.repeat(40); }],
    ['wrong nonce', f => { f.binding.members['cuda-windows-2'].identity.nonce = 'f'.repeat(64); }],
    ['missing pulse', f => { delete f.pulses['cuda-windows-4']; }],
    ['forged pulse', f => { f.pulses['cuda-windows-4'].value.at++; }],
    ['reused PID', f => { f.table.find(p => p.pid === 400).startedAt = 'new-process'; }],
    ['changed exe', f => { f.table.find(p => p.pid === 400).executable = 'D:/other/Runner.Worker.exe'; }],
    ['wrong parent', f => { f.table.find(p => p.pid === 401).ppid = 100; }],
    ['controller not owned', f => { f.table.find(p => p.pid === 110).ppid = 0; }],
    ['fifth Worker', f => { f.table.push({ pid: 500, ppid: 0, executable: 'D:/other/Runner.Worker.exe', command: 'Runner.Worker.exe', startedAt: 'foreign' }); }],
    ['expired API', f => { f.binding.api.observedAt = NOW - API_FRESH_MS - 1; }],
    ['deadline', f => { f.binding.deadline = NOW - 1; }],
  ]) { const f = fixture(); mutate(f); assert.throws(() => valid(f), undefined, label); }
  for (const values of [{ at: NOW - STALE_MS - 1 }, { at: NOW + 1 }, { status: 'released' }, { apiValidUntil: NOW - 1 }, { jobId: 999 }, { identity: { ...fixture().binding.identity, attempt: 2 } }]) {
    const f = fixture(), runner = 'cuda-windows-3'; f.pulses[runner] = signed({ ...f.pulses[runner].value, ...values }, f.keys[runner].privateKey); assert.throws(() => valid(f));
  }
});
test('actual Worker ancestry binds executable and start-time rather than trusting a PID list', () => {
  const f = fixture(); assert.equal(workerIdentity(f.table, 110, EXECUTOR).pid, 100);
  assert(ancestors(f.table, 110).has(100));
  assert.throws(() => workerIdentity(f.table, 110, 'cuda-windows-2'), /executable/);
  assert.throws(() => workerIdentity(f.table, 110, 'unknown'), /unrecognized/);
  assert.throws(() => verifyRoster(f.binding, f.pulses, f.table, 401, NOW), /not owned/);
});
test('production guard fails before CUDA for absent/mismatched queue proof; ownPids never exempts Workers; Mac route unchanged', async () => {
  const f = fixture(); let probes = 0;
  const options = { platform: 'win32', controllerPid: f.controllerPid, processTable: async () => f.table, exec: async () => { probes++; throw Error('GPU probe must be unreachable'); } };
  // Even all actual PIDs supplied as ownPids are not a Runner.Worker bypass.
  await assert.rejects(guardHardware({ backend: 'candle' }, [100,200,300,400], options), /foreign Runner.Worker/);
  await assert.rejects(guardHardware({ backend: 'candle', queueReservation: { path: '/untrusted/binding.json', sha256: digest, runId: 123, attempt: 1 } }, [], options));
  await assert.rejects(guardHardware({ backend: 'mlx', queueReservation: {} }, [], { ...options, platform: 'darwin' }), /Windows Candle only/);
  assert.equal(probes, 0);
});
test('signed heartbeat cannot be substituted with an unsigned/raw PID allowlist', () => {
  const f = fixture(); assert.equal(authenticated(f.pulses[EXECUTOR], f.binding.members[EXECUTOR].publicKey).runner, EXECUTOR);
  assert.throws(() => authenticated(f.pulses[EXECUTOR], f.binding.members['cuda-windows-2'].publicKey), /signature/);
  assert.throws(() => authenticated({ value: [200, 300, 400], signature: '' }, f.binding.members[EXECUTOR].publicKey), /signature/);
});
test('model child never inherits GitHub/API credentials or Actions runtime token endpoints', () => {
  const e = driverEnvironment({ PATH: 'safe', GH_TOKEN: 'DECLARED-NOT-A-CREDENTIAL', GITHUB_TOKEN: 'DECLARED-NOT-A-CREDENTIAL', ACTIONS_RUNTIME_TOKEN: 'DECLARED', ACTIONS_ID_TOKEN_REQUEST_URL: 'DECLARED', HF_TOKEN: 'DECLARED', SCENEWORKS_GPU_ID: '0', FLUX3_QUEUE_SOURCE: source }, { runId: 1 });
  assert.equal(e.PATH, 'safe'); assert.equal(e.CUDA_VISIBLE_DEVICES, '-1');
  assert.deepEqual(Object.keys(e).sort(), ['CUDA_VISIBLE_DEVICES', 'FLUX3_QUEUE_BINDING', 'PATH']);
});
test('no driver/ledger/model/GPU before full barrier; race/failure/interrupt releases owned reservation', async () => {
  let started = 0, released = 0;
  await assert.rejects(superviseChild({ tick: async () => { const f = fixture(); f.binding.api.jobs.pop(); valid(f); }, start: async () => { started++; }, release: async () => { released++; } }));
  assert.equal(started, 0); assert.equal(released, 1);
  let ticks = 0, stopped = 0, finish;
  await assert.rejects(superviseChild({ interval: 0, wait: async () => {}, tick: async () => { if (++ticks === 3) throw Error('peer vanished'); }, start: async () => { started++; return { done: new Promise(r => { finish = r; }), stop: async () => { stopped++; finish(1); } }; }, release: async () => { released++; } }), /peer vanished/);
  assert.equal(stopped, 1); assert.equal(released, 2);
  await assert.rejects(superviseChild({ tick: async () => {}, start: async () => { throw Error('spawn failed'); }, release: async () => { released++; } }), /spawn failed/);
  assert.equal(released, 3);
  await assert.rejects(superviseChild({ interval: 0, wait: async () => {}, tick: async () => {}, start: async () => ({ done: Promise.resolve(1), stop: async () => { stopped++; } }), release: async () => { released++; } }), /child failed/);
  assert.equal(released, 4); assert.equal(stopped, 2, 'root exit still requires cleanup');
});
test('supervisor heartbeat checks continue while its separate model child blocks on CPU work', async () => {
  let ticks = 0, releases = 0;
  await superviseChild({ interval: 10, tick: async () => { ticks++; }, release: async () => { releases++; }, start: async () => {
    const child = spawn(process.execPath, ['-e', 'const t=Date.now()+250;while(Date.now()<t){}'], { stdio: 'ignore' });
    return { done: new Promise((resolve, reject) => { child.once('error', reject); child.once('exit', resolve); }), stop: async () => child.kill() };
  } });
  assert(ticks >= 3); assert.equal(releases, 1);
});
test('campaign groups preserve exactly the 21 IDs/order, replay pairing and training/export/zero/on; first failure never advances', async () => {
  const plan = terminalPlan(await loadProductTargets()), groups = executionGroups(plan);
  assert.equal(groups.flat().length, 21); assert.equal(groups[0].length, 2);
  assert.equal(groups.find(g => g[0].purpose === 'training-reload').length, 3);
  const ids = ['training', 'zero', 'on'];
  assertGroupPassed({ cases: ids.map(caseId => ({ caseId, executionStatus: 'PASSED' })) }, ids);
  assert.throws(() => assertGroupPassed({ cases: ids.map((caseId, i) => ({ caseId, executionStatus: i === 0 ? 'FAILED' : 'NOT_RUN' })) }, ids), /FIRST_FAILURE_STOP/);
});
test('workflow/privacy contracts: one manual four-slot scheduler matrix; no upload or administration route', async () => {
  const workflow = await readFile(new URL('../.github/workflows/flux3-image-klein-terminal-queue.yml', import.meta.url), 'utf8');
  assert.match(workflow, /^on:\n {2}workflow_call:/m); assert.doesNotMatch(workflow, /^ {2}(push|pull_request|schedule):/m);
  assert.match(workflow, /slot: \[1, 2, 3, 4\]/); assert.match(workflow, /max-parallel: 4/); assert.match(workflow, /fail-fast: false/);
  assert.match(workflow, /runs-on: \[self-hosted, Windows, X64, cuda\]/); assert.match(workflow, /CUDA_VISIBLE_DEVICES: "-1"/);
  const code = await readFile(new URL('./flux3-terminal-queue.mjs', import.meta.url), 'utf8'), campaign = await readFile(new URL('./flux3-terminal-queue-campaign.mjs', import.meta.url), 'utf8');
  for (const text of [workflow, code, campaign]) assert.doesNotMatch(text, /upload-artifact|actions\/cache|Stop-ScheduledTask|Disable-ScheduledTask|Stop-Service|schtasks|Set-Acl|taskkill/i);
  const native = await readFile(new URL('./flux3-queue-container.ps1', import.meta.url), 'utf8');
  assert.match(native, /0x2000/); assert(native.indexOf('if(!AssignProcessToJobObject') < native.indexOf('ResumeThread(p.thread)'));
  assert.match(code, /if \(role === 'executor'\)/); assert.match(code, /setInterval/); assert.match(code, /'wx'/);
  assert.match(campaign, /existing ledger\/claim requires coordinator reconciliation/);
  const driver = await readFile(new URL('./flux3-image-klein-terminal-driver.mjs', import.meta.url), 'utf8'); assert.match(driver, /admission\.queueReservation && report\.executionStatus !== 'PASSED'/);
  const dispatcher = await readFile(new URL('../.github/workflows/memory-catalog-campaign.yml', import.meta.url), 'utf8');
  assert.match(dispatcher, /flux3_queue_control_sha256:[\s\S]*?default: none/); assert.match(dispatcher, /inputs\.profile == 'flux3-image-klein-terminal-queue'/);
});
test('rendezvous publication is atomic/exclusive: a race publishes one complete document and never replaces it', async t => {
  const dir = await mkdtemp(path.join(os.tmpdir(), 'flux3-queue-fixture-')); t.after(() => rm(dir, { recursive: true, force: true }));
  const file = path.join(dir, 'member.json'), value = { nonce: 'declared', payload: 'x'.repeat(10000) };
  const races = await Promise.allSettled([publishNew(file, value), publishNew(file, { invalid: true })]);
  assert.equal(races.filter(r => r.status === 'fulfilled').length, 1);
  const first = await readFile(file, 'utf8'); assert([JSON.stringify(value), JSON.stringify({ invalid: true })].includes(first));
  await assert.rejects(publishNew(file, { replaced: true }), { code: 'EEXIST' });
  assert.equal(await readFile(file, 'utf8'), first);
});
test('binding path cannot select an arbitrary reservation, and missing proof refuses (no PID fallback)', async () => {
  await assert.rejects(verifyQueueReservation({ queueReservation: { path: '/tmp/untrusted/binding.json', runId: 1, attempt: 1, sha256: digest } }, [], 1), /strictly equal/);
});

test('cleanup failure cannot release even after root exits; holders retain slots through failure/cancel/deadline until verified zero', async () => {
  let released = 0, cleanup = 0;
  await assert.rejects(superviseChild({ interval: 0, wait: async () => {}, tick: async () => {}, start: async () => ({ done: Promise.resolve(0), stop: async () => { cleanup++; throw Error('kernel accounting unavailable'); } }), release: async () => { released++; } }), /kernel accounting/);
  assert.equal(cleanup, 1); assert.equal(released, 0, 'no SAFE_TO_RELEASE on cleanup failure');
  let grandchildAlive = true, held = 0, probes = 0;
  await holdUntilSafe({ probe: async () => { probes++; if (probes === 1) throw Error('API failure/invalid ACK'); return !grandchildAlive; }, heartbeat: async () => { held++; assert.equal(released, 0); }, wait: async () => { if (held === 4) grandchildAlive = false; } });
  assert.equal(held, 4, 'holder remains occupied despite failure, cancellation or expired ordinary deadline');
  assert.equal(grandchildAlive, false); released++; assert.equal(released, 1);
});
test('verification samples its live clock after asynchronous observations, including a pulse published between reads', async () => {
  const f = fixture(); let now = NOW, collected = false;
  const result = await verifyObservedRoster(async () => {
    await Promise.resolve(); now += 25;
    const runner = 'cuda-windows-2'; f.pulses[runner] = signed({ ...f.pulses[runner].value, at: now }, f.keys[runner].privateKey);
    collected = true; return { binding: f.binding, pulses: f.pulses, table: f.table };
  }, f.controllerPid, () => { assert(collected, 'clock captured before observations'); return now; });
  assert.deepEqual(result, [200,300,400]);
  assert.throws(() => verifyRoster(f.binding, f.pulses, f.table, f.controllerPid, NOW), /heartbeat/, 'mutation: the old early numeric clock falsely refuses a healthy future pulse');
  await assert.rejects(verifyObservedRoster(async () => ({ binding: f.binding, pulses: f.pulses, table: f.table }), f.controllerPid, NOW));
});
test('one campaign-wide concurrency group serializes dispatches but never serializes the four siblings against each other', async () => {
  const dispatcher = await readFile(new URL('../.github/workflows/memory-catalog-campaign.yml', import.meta.url), 'utf8');
  const workflow = await readFile(new URL('../.github/workflows/flux3-image-klein-terminal-queue.yml', import.meta.url), 'utf8');
  const block = dispatcher.slice(dispatcher.indexOf('  flux3-terminal-queue:')).split('\n  qwen-app-lifecycle:')[0];
  const contract = b => {
    assert.match(b, /concurrency:\n      group: flux3-terminal-queue-physical-host\n      cancel-in-progress: false/);
    const group = /group: (.+)/.exec(b)[1]; assert.doesNotMatch(group, /\$\{\{|run_id|source_sha|ref/);
    return group;
  };
  assert.equal(contract(block), contract(block), 'all source branches/approvals of our queue profile share one group');
  assert.throws(() => contract(block.replace('group: flux3-terminal-queue-physical-host', 'group: per-run-id')), 'mutation: removing the common group breaks serialization');
  assert.doesNotMatch(workflow, /concurrency:/, 'no nested equal group or per-member group deadlock');
  assert.match(workflow, /max-parallel: 4/); assert.match(workflow, /slot: \[1, 2, 3, 4\]/);
});
