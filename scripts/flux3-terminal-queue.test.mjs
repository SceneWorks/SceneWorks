// DECLARED SYNTHETIC CPU tests: no device, model, runner job, API call, or campaign permit.
import assert from 'node:assert/strict';
import test from 'node:test';
import { generateKeyPairSync } from 'node:crypto';
import { spawn } from 'node:child_process';
import { readFile, mkdtemp, rm, mkdir, writeFile, rename, readdir } from 'node:fs/promises';
import { once } from 'node:events';
import { fileURLToPath } from 'node:url';
import { setTimeout as delay } from 'node:timers/promises';
import os from 'node:os';
import path from 'node:path';
import { SLOTS, EXECUTOR, STALE_MS, API_FRESH_MS, queueInputs, electedRole, validateApi, ancestors, workerIdentity, signed, authenticated, verifyRoster, driverEnvironment, superviseChild, verifyQueueReservation, publishNew, holdUntilSafe, verifyObservedRoster, verifyAccessProofs, digest as queueDigest, createAtomicPublisher, ATOMIC_RENAME_BUDGET_MS, ATOMIC_RENAME_BACKOFF_MS, HEARTBEAT_MS, assessStartupApi, createExecutorApiRefresh, API_REFRESH_MS, assertNoPeerFailure, unregisteredMemberAbsent, assertStartupDeadline, waitForOwnJob, queueFailureReport, QueuePhaseFailure, safeQueueFailure, QUEUE_PHASES, processIdentity, authenticatedTerminalCleanup, SAFE_REASON_CODES } from './flux3-terminal-queue.mjs';
import { executionGroups, assertGroupPassed, validateQueueOperation, executeAdmittedQueueOperation } from './flux3-terminal-queue-campaign.mjs';
import { loadProductTargets, terminalPlan } from './flux3-image-klein-terminal-plan.mjs';
import { guardHardware } from './flux3-image-klein-terminal-session.mjs';
const source = 'a'.repeat(40), inference = 'b'.repeat(40), digest = 'c'.repeat(64), NOW = 1000000;
function fixture() {
  const identity = { repository: 'SceneWorks/SceneWorks', runId: 123, attempt: 1, sourceSha: source, inferenceMainSha: inference, controlSha256: digest, nonce: 'd'.repeat(64) };
  const run = { id: 123, run_attempt: 1, head_sha: source, repository: { full_name: identity.repository }, event: 'workflow_dispatch', status: 'in_progress', path: '.github/workflows/memory-catalog-campaign.yml' };
  const jobs = [], table = [], members = {}, pulses = {}, keys = {};
  for (const [n, [runner, slot]] of Object.entries(SLOTS).entries()) {
    const pid = (n + 1) * 100, jobId = n + 1000;
    table.push({ pid, ppid: 0, startedAt: `worker-start-${n}`, executable: slot.worker, command: slot.worker, sessionId: 2 });
    table.push({ pid: pid + 1, ppid: pid, startedAt: `supervisor-start-${n}`, executable: 'C:/node/node.exe', command: 'node.exe', sessionId: 2 });
    jobs.push({ id: jobId, name: `flux3-terminal-queue / reserve (${4 - n})`, run_id: 123, head_sha: source, status: 'in_progress', runner_name: runner, runner_id: slot.id });
    keys[runner] = generateKeyPairSync('ed25519');
    const process = p => ({ pid: p.pid, ppid: p.ppid, startedAt: p.startedAt, executable: p.executable, sessionId: p.sessionId });
    members[runner] = { identity: structuredClone(identity), runner, role: electedRole(runner), hostName: 'DECLARED-WINDOWS-HOST', context: {sessionId:2,ownerSid:'S-1-5-21-123'}, jobId, worker: process(table.at(-2)), supervisor: process(table.at(-1)), publicKey: keys[runner].publicKey.export({ type: 'spki', format: 'pem' }) };
    pulses[runner] = signed({ identity: structuredClone(identity), runner, jobId, at: NOW, apiValidUntil: NOW + API_FRESH_MS, sequence: 1, status: 'holding' }, keys[runner].privateKey);
  }
  table.push({ pid: 110, ppid: 101, startedAt: 'driver-start', executable: 'C:/node/node.exe', command: 'node.exe', sessionId: 2 });
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
    ['different Windows session', f => { f.binding.members['cuda-windows-3'].context.sessionId = 3; }],
    ['different account', f => { f.binding.members['cuda-windows-3'].context.ownerSid = 'S-1-5-21-456'; }],
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

test('every holder must attest real query access to the same empty container before resume; missing/mutated proofs refuse', () => {
  const f = fixture(); f.binding.container = {jobName:'Local\\SceneWorksFlux3-123-1-'+'f'.repeat(40),context:f.binding.members[EXECUTOR].context};
  const proofs = {};
  for(const runner of Object.keys(SLOTS).filter(r=>r!==EXECUTOR)) proofs[runner]=signed({identity:f.binding.identity,runner,jobName:f.binding.container.jobName,containerSha256:queueDigest(f.binding.container),context:f.binding.container.context,queryAccess:true,zeroBeforeStart:true},f.keys[runner].privateKey);
  verifyAccessProofs(f.binding,proofs);
  const runner='cuda-windows-3';
  for(const change of [{queryAccess:false},{zeroBeforeStart:false},{containerSha256:'0'.repeat(64)},{context:{sessionId:99,ownerSid:'S-1-5-21-123'}}]) {
    const mutated={...proofs,[runner]:signed({...proofs[runner].value,...change},f.keys[runner].privateKey)};assert.throws(()=>verifyAccessProofs(f.binding,mutated));
  }
  const { [runner]:_omitted,...missing }=proofs;assert.throws(()=>verifyAccessProofs(f.binding,missing));
});

// Real same-account Windows reader without FileShare.Delete, only on an owned CPU fixture.
async function lockedPulse(t) {
  const root = 'D:/tmp/sc26243-run/queue';
  await mkdir(root, { recursive: true });
  const dir = await mkdtemp(path.join(root, 'atomic-cpu-fixture-'));
  const file = path.join(dir, 'cuda-windows-4.pulse.json'), release = path.join(dir, 'reader-release');
  const keys = generateKeyPairSync('ed25519'), old = signed({ sequence: 0, status: 'holding' }, keys.privateKey);
  await writeFile(file, JSON.stringify(old));
  const script = '$ErrorActionPreference="Stop"; $s=[IO.File]::Open($env:ATOMIC_TEST_FILE,[IO.FileMode]::Open,[IO.FileAccess]::Read,[IO.FileShare]::Read); try { [Console]::WriteLine("LOCKED"); $end=[DateTime]::UtcNow.AddSeconds(10); while (!(Test-Path -LiteralPath $env:ATOMIC_TEST_RELEASE)) { if ([DateTime]::UtcNow -gt $end) { throw "owned reader fixture deadline" }; Start-Sleep -Milliseconds 10 } } finally { $s.Dispose() }';
  const child = spawn('powershell.exe', ['-NoProfile', '-NonInteractive', '-EncodedCommand', Buffer.from(script, 'utf16le').toString('base64')], { windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'], env: { ...process.env, CUDA_VISIBLE_DEVICES: '-1', ATOMIC_TEST_FILE: file, ATOMIC_TEST_RELEASE: release } });
  let stderr = ''; child.stderr.on('data', b => { stderr += b; });
  const ended = once(child, 'exit');
  t.after(async () => { await writeFile(release, 'release'); const [code] = await ended; assert.equal(code, 0, stderr); await rm(dir, { recursive: true, force: true }); });
  let ready = ''; for await (const b of child.stdout) { ready += b; if (ready.includes('LOCKED')) break; }
  assert.match(ready, /LOCKED/, stderr);
  const probe = path.join(dir, 'rename-probe.tmp'); await writeFile(probe, JSON.stringify(old));
  await assert.rejects(rename(probe, file), { code: 'EPERM' }); await rm(probe);
  return { dir, file, keys, old, unlock: () => writeFile(release, 'release') };
}
const windowsOnly = { skip: process.platform !== 'win32', timeout: 15000 };
test('atomic publisher real Windows 150ms sharing hold retries EPERM; readers see complete signed JSON', windowsOnly, async t => {
  const f = await lockedPulse(t), publish = createAtomicPublisher();
  const next = signed({ sequence: 1, status: 'holding' }, f.keys.privateKey);
  const started = performance.now(), task = publish(f.file, () => next);
  task.catch(() => {}); // Attach immediately while the test deliberately holds the reader.
  await delay(150);
  assert.deepEqual(JSON.parse(await readFile(f.file, 'utf8')), f.old);
  await f.unlock();
  // A bounded set of normal promptly-closed readers, not a continuous lock campaign.
  for (let i = 0; i < 3; i++) {
    const seen = authenticated(JSON.parse(await readFile(f.file, 'utf8')), f.keys.publicKey);
    assert([0, 1].includes(seen.sequence)); await delay(37);
  }
  await task;
  assert(performance.now() - started < 1200, 'publication must remain within heartbeat margin');
  assert.deepEqual(authenticated(JSON.parse(await readFile(f.file, 'utf8')), f.keys.publicKey), next.value);
  assert.deepEqual(await readdir(f.dir), ['cuda-windows-4.pulse.json', 'reader-release']);
});
test('atomic publisher persistent Windows lock fails bounded, keeps old pulse and removes only its temp', windowsOnly, async t => {
  const f = await lockedPulse(t), started = performance.now();
  await assert.rejects(createAtomicPublisher()(f.file, { sequence: 1 }), { code: 'EPERM' });
  const elapsed = performance.now() - started;
  assert(elapsed >= ATOMIC_RENAME_BUDGET_MS - 50 && elapsed < 1200, `bounded rename took ${elapsed}ms`);
  assert.deepEqual(JSON.parse(await readFile(f.file, 'utf8')), f.old);
  assert.deepEqual(await readdir(f.dir), ['cuda-windows-4.pulse.json']);
});
test('atomic publisher cancellation interrupts Windows sharing wait without touching old pulse', windowsOnly, async t => {
  const f = await lockedPulse(t); let cancelled = false;
  const started = performance.now(), task = createAtomicPublisher()(f.file, { sequence: 1 }, { cancelled: () => cancelled });
  const rejection = assert.rejects(task, /publication cancelled/);
  await delay(100); cancelled = true; await rejection;
  assert(performance.now() - started < 400);
  assert.deepEqual(JSON.parse(await readFile(f.file, 'utf8')), f.old);
  assert.deepEqual(await readdir(f.dir), ['cuda-windows-4.pulse.json']);
});
test('atomic publisher serializes overlapping foreground/timer factories, signatures and sequences', windowsOnly, async t => {
  const f = await lockedPulse(t), publish = createAtomicPublisher(); let sequence = 0;
  const foreground = publish(f.file, () => signed({ sequence: ++sequence, status: 'holding' }, f.keys.privateKey));
  await delay(30);
  const timer = publish(f.file, () => signed({ sequence: ++sequence, status: 'safe_to_release' }, f.keys.privateKey));
  await delay(100); assert.equal(sequence, 1, 'queued factory must not run before preceding rename');
  await f.unlock(); await Promise.all([foreground, timer]);
  assert.deepEqual(authenticated(JSON.parse(await readFile(f.file, 'utf8')), f.keys.publicKey), { sequence: 2, status: 'safe_to_release' });
});
async function mutatedPublisher(mutate) {
  const source = await readFile(new URL('./flux3-terminal-queue.mjs', import.meta.url), 'utf8');
  const helper = source.slice(source.indexOf('export const ATOMIC_RENAME_BUDGET_MS'), source.indexOf('const atomic = createAtomicPublisher();'));
  const imports = 'import assert from "node:assert/strict"; import path from "node:path"; import {randomBytes} from "node:crypto"; import {writeFile,rename,unlink} from "node:fs/promises"; const sleep=ms=>new Promise(r=>setTimeout(r,ms));';
  const changed = mutate(helper); assert.notEqual(changed, helper, 'mutation must change actual publisher source');
  return (await import('data:text/javascript;base64,' + Buffer.from(imports + changed).toString('base64'))).createAtomicPublisher();
}
test('mutation removing per-target serialization violates ordered factory contract on real Windows lock', windowsOnly, async t => {
  const f = await lockedPulse(t);
  const publish = await mutatedPublisher(s => s.replace('previous.catch(() => {}).then', 'Promise.resolve().then'));
  let sequence = 0;
  const first = publish(f.file, () => ({ sequence: ++sequence })); first.catch(() => {});
  const second = publish(f.file, () => ({ sequence: ++sequence })); second.catch(() => {});
  await delay(100);
  assert.throws(() => assert.equal(sequence, 1), /2 !== 1/, 'same ordering assertion rejects missing serialization');
  await f.unlock(); await Promise.all([first, second]);
});
test('mutation removing the sub-second retry bound violates persistent-lock contract', windowsOnly, async t => {
  const f = await lockedPulse(t);
  const publish = await mutatedPublisher(s => s.replace('ATOMIC_RENAME_BUDGET_MS = 800', 'ATOMIC_RENAME_BUDGET_MS = 2000'));
  const started = performance.now(), task = publish(f.file, { sequence: 1 }); task.catch(() => {});
  await delay(1100);
  assert.deepEqual(JSON.parse(await readFile(f.file, 'utf8')), f.old);
  await f.unlock(); await task;
  assert.throws(() => assert(performance.now() - started < 1000), 'same wall-budget assertion rejects extended/unbounded retries');
});
test('atomic rename retry margin does not relax heartbeat freshness or backoff thresholds', () => {
  assert.equal(ATOMIC_RENAME_BUDGET_MS, 800); assert.equal(ATOMIC_RENAME_BACKOFF_MS, 50);
  assert.equal(HEARTBEAT_MS, 2000); assert.equal(STALE_MS, 20000);
  assert(ATOMIC_RENAME_BUDGET_MS < HEARTBEAT_MS && HEARTBEAT_MS + ATOMIC_RENAME_BUDGET_MS < STALE_MS);
});

test('queued global run permits own coherent membership, never active readiness or circular all-member dependency', async () => {
  const f = fixture(), { run, jobs } = f.binding.api;
  run.status = 'queued';
  const ownOnly = jobs.filter(j => j.runner_name === EXECUTOR);
  const state = assessStartupApi(run, ownOnly, f.binding.identity);
  assert.equal(state.ready, false); assert.equal(state.assignments[EXECUTOR], ownOnly[0].id);
  const job = await waitForOwnJob(EXECUTOR, f.binding.identity, async () => ({ run, jobs: ownOnly }), async () => assert.fail('own assignment must not wait for other members'));
  assert.equal(job, ownOnly[0].id);
  assert.throws(() => validateApi(run, jobs, f.binding.identity), /in_progress/);
  assert.equal(assessStartupApi(run, jobs, f.binding.identity).ready, false);
  run.status = 'in_progress'; assert.equal(assessStartupApi(run, jobs, f.binding.identity).ready, true);
  assert.equal(Object.keys(validateApi(run, jobs, f.binding.identity)).length, 4);
});
test('startup fake frames wait for missing/null/zero assignment then register; deadline and cancellation stop waiting', async () => {
  const f = fixture(), runner = 'cuda-windows-3', own = f.binding.api.jobs.find(j => j.runner_name === runner);
  const run = { ...f.binding.api.run, status: 'queued' };
  const frames = [undefined, { run, jobs: [] }, { run, jobs: [{ ...own, status: 'requested', runner_name: '', runner_id: 0, run_id: null, head_sha: null }] }, { run, jobs: [{ ...own, runner_name: null, runner_id: null }] }, { run, jobs: [own] }];
  let now = 0, cursor = 0;
  const got = await waitForOwnJob(runner, f.binding.identity, async () => { assertStartupDeadline(now, 10000, false); return frames[cursor++]; }, async ms => { now += ms; });
  assert.equal(got, own.id); assert.equal(now, 8000);
  now = 0;
  await assert.rejects(waitForOwnJob(runner, f.binding.identity, async () => { assertStartupDeadline(now, 4000, false); return { run, jobs: [] }; }, async ms => { now += ms; }), /deadline/);
  assert.equal(now, 4000);
  assert.throws(() => assertStartupDeadline(0, 4000, true), /interrupted/);
});
test('startup hard identities, malformed frames, conclusions, terminal slots and unknown roles never become wait frames', () => {
  for (const mutate of [
    f => { f.binding.api.run.status = 'cancelled'; }, f => { f.binding.api.run.status = 'completed'; },
    f => { f.binding.api.run.conclusion = 'cancelled'; }, f => { f.binding.api.run.head_sha = 'f'.repeat(40); },
    f => { f.binding.api.run.run_attempt++; }, f => { f.binding.api.run.repository.full_name = 'wrong/repository'; },
    f => { f.binding.api.run.event = 'push'; }, f => { f.binding.api.run.path = 'wrong.yml'; },
    f => { f.binding.api.jobs[0].status = 'completed'; }, f => { f.binding.api.jobs[0].conclusion = 'failure'; },
    f => { f.binding.api.jobs[0].name = 'flux3-terminal-queue / reserve (5)'; },
    f => { f.binding.api.jobs[0].name = 'flux3-terminal-queue / wrong-role'; },
    f => { f.binding.api.jobs[0].runner_name = 'unknown'; }, f => { f.binding.api.jobs[0].runner_id = 999; },
    f => { f.binding.api.jobs[0].runner_id = 0; f.binding.api.jobs[0].runner_name = 'wrong'; },
    f => { f.binding.api.jobs[0].runner_name = null; f.binding.api.jobs[0].runner_id = 999; },
    f => { f.binding.api.jobs[0].runner_id = -1; }, f => { f.binding.api.jobs[0].run_id++; },
    f => { f.binding.api.jobs[0].head_sha = 'f'.repeat(40); }, f => { f.binding.api.jobs[0].id = null; },
    f => { f.binding.api.jobs[0].name = null; }, f => { f.binding.api.jobs = null; },
  ]) {
    const f = fixture(); f.binding.api.run.status = 'queued'; mutate(f);
    assert.throws(() => assessStartupApi(f.binding.api.run, f.binding.api.jobs, f.binding.identity));
  }
});
test('one executor GET pair per 15s stays below repository rate budget; all three holders consume only signed cached frames', async () => {
  const f = fixture(); let now = 0, gets = 0;
  const refresh = createExecutorApiRefresh(async () => { gets += 2; return { ...f.binding.api, observedAt: now, ready: true }; }, () => now);
  const first = await refresh(); now = API_REFRESH_MS - 1; assert.equal(await refresh(), first); assert.equal(first.observedAt, 0);
  for (now = API_REFRESH_MS; now < 3600000; now += HEARTBEAT_MS) {
    const frame = await refresh(), proof = signed({ identity: f.binding.identity, api: frame }, f.keys[EXECUTOR].privateKey);
    for (let holder = 0; holder < 3; holder++) {
      const read = authenticated(proof, f.binding.members[EXECUTOR].publicKey);
      validateApi(read.api.run, read.api.jobs, read.identity);
    }
  }
  assert(gets >= 448 && gets <= 482, `one hour used ${gets} GETs, not four independent 1800+ request loops`);
  assert(gets < 1000); assert.equal(API_FRESH_MS, 90000); assert.equal(STALE_MS, 20000);
});
test('API errors have no automatic retries and rejected pending frames never renew coherent active proof age', async () => {
  for (const message of ['HTTP 401', 'HTTP 403', 'HTTP 500', 'malformed JSON']) {
    let calls = 0, now = 0;
    const refresh = createExecutorApiRefresh(async () => { calls++; throw Error(message); }, () => now);
    await assert.rejects(refresh(), new RegExp(message)); now = 90000;
    await assert.rejects(refresh(), new RegExp(message)); assert.equal(calls, 1);
  }
  const f = fixture(); let now = 0, latest = f.binding.api, published;
  const refresh = createExecutorApiRefresh(async () => {
    validateApi(latest.run, latest.jobs, f.binding.identity);
    return { ...latest, observedAt: now, ready: true };
  }, () => now);
  published = await refresh();
  now = 15000; latest = structuredClone(latest); latest.run.status = 'queued';
  await assert.rejects(refresh(), /in_progress/); assert.equal(published.observedAt, 0);
  now = 90001; assert(now - published.observedAt > API_FRESH_MS);
});
test('pre-member peer failure aborts startup; immediate release needs actual immutable member absence, not uncertain publication', async t => {
  const dir = await mkdtemp(path.join(os.tmpdir(), 'flux3-startup-cpu-')); t.after(() => rm(dir, { recursive: true, force: true }));
  const own = path.join(dir, 'cuda-windows-3.member.json');
  assert.equal(await unregisteredMemberAbsent(undefined, own), true);
  await writeFile(own, 'partial-not-json');
  assert.equal(await unregisteredMemberAbsent(undefined, own), false, 'existing partial publication must retain cleanup rules');
  await assert.rejects(unregisteredMemberAbsent(undefined, own, async () => { throw Object.assign(Error('denied'), { code: 'EACCES' }); }), /denied/);
  assert.equal(await unregisteredMemberAbsent({}, path.join(dir, 'missing')), false, 'known member still needs registered cleanup');
  await assertNoPeerFailure(dir, EXECUTOR);
  await writeFile(path.join(dir, 'cuda-windows-3.failure.json'), JSON.stringify({ error: 'registration failed' }));
  const f = fixture(); let waited = false;
  await assert.rejects(waitForOwnJob(EXECUTOR, f.binding.identity, async () => { await assertNoPeerFailure(dir, EXECUTOR); return undefined; }, async () => { waited = true; }), /peer failed/);
  assert.equal(waited, false, 'failure marker stops executor without waiting for a missing member');
});
test('startup mutations admitting queued readiness or ignoring peer failures violate the safety assertions', async t => {
  const source = await readFile(new URL('./flux3-terminal-queue.mjs', import.meta.url), 'utf8');
  const load = async (from, to) => {
    assert(source.includes(from));
    return import('data:text/javascript;base64,' + Buffer.from(source.replace(from, to)).toString('base64'));
  };
  const early = await load("ready: run.status === 'in_progress' && rows.length === 4", 'ready: rows.length === 4');
  const f = fixture(); f.binding.api.run.status = 'queued';
  assert.throws(() => assert.equal(early.assessStartupApi(f.binding.api.run, f.binding.api.jobs, f.binding.identity).ready, false));
  const ignored = await load('for (const peer of Object.keys(SLOTS).filter(n => n !== runner))', 'for (const peer of [])');
  const dir = await mkdtemp(path.join(os.tmpdir(), 'flux3-failure-mutation-')); t.after(() => rm(dir, { recursive: true, force: true }));
  await writeFile(path.join(dir, 'cuda-windows-3.failure.json'), '{}');
  const requiresAbort = check => assert.rejects(check(dir, EXECUTOR), /peer failed/);
  await requiresAbort(assertNoPeerFailure);
  await assert.rejects(requiresAbort(ignored.assertNoPeerFailure), /Missing expected rejection/);
});

test('explicit admission-only operation cannot call ledger, provider, model or generation callbacks', async () => {
  const calls = [];
  const forbidden = name => async () => { calls.push(name); assert.fail('diagnostic must not execute side effects'); };
  const result = await executeAdmittedQueueOperation('admission-only', { initialize: forbidden('ledger'), execute: forbidden('provider/model/generation'), finalize: forbidden('ledger audit') });
  assert.deepEqual(calls, []);
  assert.deepEqual(result, { operation: 'admission-only', status: 'DIAGNOSTIC_ONLY', generationCount: 0, campaignPermitsConsumed: 0, ledgerInitialized: false });
  assert.notEqual(result.status, 'PASSED');
});
test('normal run initializes exactly once after admission and finalizes on success or model failure', async () => {
  for (const fail of [false, true]) {
    const order = ['admitted'];
    const task = executeAdmittedQueueOperation('run', {
      initialize: async () => { assert.deepEqual(order, ['admitted']); order.push('initialize'); },
      execute: async () => { order.push('execute'); if (fail) throw Error('declared synthetic failure'); },
      finalize: async () => { order.push('finalize'); },
    });
    if (fail) await assert.rejects(task, /synthetic failure/); else assert.equal((await task).status, 'COMPLETED');
    assert.deepEqual(order, ['admitted', 'initialize', 'execute', 'finalize']);
  }
  let executed = false;
  await assert.rejects(executeAdmittedQueueOperation('run', { initialize: async () => { throw Error('init failed'); }, execute: async () => { executed = true; }, finalize: async () => assert.fail('no ledger was initialized') }), /init failed/);
  assert.equal(executed, false);
});
test('missing/malformed operation refuses before admission or execution and diagnostic never skips production admission', async () => {
  for (const operation of [undefined, null, '', 'RUN', 'diagnostic', 'admission-only\n', {}, ['run']]) {
    assert.throws(() => validateQueueOperation(operation), /explicit queue operation/);
    await assert.rejects(executeAdmittedQueueOperation(operation, { initialize: () => assert.fail(), execute: () => assert.fail(), finalize: () => assert.fail() }), /explicit queue operation/);
  }
  const source = await readFile(new URL('./flux3-terminal-queue-campaign.mjs', import.meta.url), 'utf8');
  const body = source.slice(source.indexOf('async function runCampaignInner'));
  assert(body.indexOf('validateQueueOperation(control.operation)') < body.indexOf('loadProductTargets()'));
  assert(body.indexOf("phase('admission_write')") < body.indexOf('executeAdmittedQueueOperation(control.operation'));
  assert(body.indexOf('guardHardware(admission') < body.indexOf('executeAdmittedQueueOperation(control.operation'));
  assert(body.indexOf('executeAdmittedQueueOperation(control.operation') < body.indexOf('initializeLedger('));
  assert.doesNotMatch(body.slice(0, body.indexOf('executeAdmittedQueueOperation(control.operation')), /if\s*\(\s*(?:control\.)?operation === ['"]admission-only/);
});
test('failure reporting exposes only fixed phase/code, fixed exception class and numeric whitelisted metrics', () => {
  const hidden = 'PRIVATE_SENTINEL_UNTRUSTED_PAYLOAD';
  const injected = Object.assign(Error(hidden), { code: hidden, name: hidden, stack: hidden, command: hidden, env: { arbitrary: hidden } });
  const failure = new QueuePhaseFailure(injected, hidden, { diskFreeBytes: 123, diskRequiredBytes: hidden, foreignDedicatedBytes: -1, arbitrary: hidden, physicalCapacityBytes: 456 });
  assert.deepEqual(safeQueueFailure(failure), { phase: 'startup', code: 'QUEUE_PRE_ADMISSION_FAILED' });
  assert.deepEqual(failure.report, { phase: 'startup', code: 'QUEUE_PRE_ADMISSION_FAILED', exceptionClass: 'Error', metrics: { diskFreeBytes: 123, physicalCapacityBytes: 456 } });
  assert(!JSON.stringify(failure.report).includes(hidden));
  for (const phase of QUEUE_PHASES) {
    const report = queueFailureReport(Object.assign(Error(hidden), { code: 'ENOENT' }), phase);
    assert.equal(report.phase, phase); assert.equal(report.code, 'QUEUE_FS_ENOENT');
    assert(!JSON.stringify(report).includes(hidden));
  }
  failure.report.phase = hidden; failure.report.code = hidden;
  assert.deepEqual(safeQueueFailure(failure), { phase: 'startup', code: 'QUEUE_PRE_ADMISSION_FAILED' });
  assert.deepEqual(safeQueueFailure(injected), { phase: 'startup', code: 'QUEUE_PRE_ADMISSION_FAILED' });
});
test('actual CLI failure prints only allowlisted phase and code, never exception text or environment', async () => {
  for (const mode of ['invalid-operation', 'campaign']) {
    const child = spawn(process.execPath, [fileURLToPath(new URL('./flux3-terminal-queue.mjs', import.meta.url)), mode], { env: { ...process.env, CUDA_VISIBLE_DEVICES: '-1', QUEUE_TEST_UNTRUSTED: 'PRIVATE_SENTINEL_CLI', FLUX3_QUEUE_BINDING: 'PRIVATE_SENTINEL_CLI' }, stdio: ['ignore', 'pipe', 'pipe'] });
    let text = ''; child.stdout.on('data', b => { text += b; }); child.stderr.on('data', b => { text += b; });
    const [code] = await once(child, 'exit'); assert.equal(code, 1);
    assert.deepEqual(JSON.parse(text.trim()), { phase: mode === 'campaign' ? 'queue_ownership' : 'startup', code: 'QUEUE_PRE_ADMISSION_FAILED' });
    assert.doesNotMatch(text, /PRIVATE_SENTINEL|AssertionError|stack|node:|invalid-operation/);
  }
});
test('fixed guard phase observation does not change any checks, command environments or result', async () => {
  const gpu = 'GPU-' + '1'.repeat(8) + '-' + '1'.repeat(4) + '-' + '1'.repeat(4) + '-' + '1'.repeat(4) + '-' + '1'.repeat(12);
  const exercise = async onPhase => {
    const calls = [];
    const result = await guardHardware({ backend: 'candle', gpuUuid: gpu }, [], {
      platform: 'win32', env: {}, controllerPid: 1, processTable: async () => [{ pid: 1, ppid: 0, command: 'node.exe' }], onPhase,
      exec: async (command, args, options) => {
        calls.push({ command, args, options });
        if (command === 'powershell.exe') return { stdout: gpu + ' luid=0x00000001_0x00000002' };
        if (args.includes('--query-gpu=index,uuid,pci.bus_id')) return { stdout: '1, ' + gpu + ', 00000000:C1:00.0' };
        if (args.includes('--query-gpu=memory.total,memory.free')) return { stdout: '97887, 96835' };
        if (args.includes('-x')) return { stdout: '<uuid>' + gpu + '</uuid><processes></processes>' };
        if (args.includes('--query-compute-apps=pid,gpu_uuid')) return { stdout: '' };
        assert.fail('unexpected probe');
      },
    });
    return { calls, result };
  };
  const phases = [], observed = await exercise(p => { assert(QUEUE_PHASES.includes(p)); phases.push(p); });
  assert.deepEqual(observed, await exercise(undefined));
  assert.deepEqual(phases, ['queue_ownership', 'physical_gpu_uuid', 'masked_cuda_uuid', 'gpu_process_identity', 'gpu_headroom']);
  assert.equal(observed.calls.find(c => c.command === 'powershell.exe').options.env.CUDA_VISIBLE_DEVICES, '1');
});

test('process identity failure names the exact missing field through a fixed safe code', () => {
  const row = { pid: 7, ppid: 1, startedAt: 'start', executable: 'C:/node/node.exe', command: 'node.exe', sessionId: 2 };
  assert.deepEqual(processIdentity([row], 7), { pid: 7, ppid: 1, startedAt: 'start', executable: 'C:/node/node.exe', sessionId: 2 });
  for (const [code, rows] of [
    ['QUEUE_PROCESS_ROW_MISSING', []],
    ['QUEUE_PROCESS_START_MISSING', [{ ...row, startedAt: undefined }]],
    ['QUEUE_PROCESS_EXECUTABLE_MISSING', [{ ...row, executable: undefined }]],
    ['QUEUE_PROCESS_EXECUTABLE_NOT_ABSOLUTE', [{ ...row, executable: 'node.exe' }]],
  ]) {
    let error; try { processIdentity(rows, 7); } catch (caught) { error = caught; } assert(error, code);
    assert.equal(error.safeCode, code, code);
    assert.equal(error.message, 'queue process identity unavailable', 'only the fixed message is retained');
    assert(SAFE_REASON_CODES.includes(error.safeCode));
    const report = queueFailureReport(error, 'queue_ownership');
    assert.equal(report.reason, code); assert.equal(report.code, 'QUEUE_PRE_ADMISSION_FAILED');
    assert.doesNotMatch(JSON.stringify(report), /node\.exe|C:\/|start/);
  }
  const plain = new Error('failure'); plain.safeCode = 'PRIVATE_SENTINEL';
  assert.equal(queueFailureReport(plain, 'gpu_headroom').reason, undefined, 'unknown codes never reach the report');
  assert.equal(safeQueueFailure(new QueuePhaseFailure(plain, 'gpu_headroom')).reason, undefined);
});

test('a holder concludes only on authenticated terminal cleanup; disappearance without it stays NOTPASS', async () => {
  const f = fixture(), dir = await mkdtemp(path.join(os.tmpdir(), 'flux3-queue-terminal-'));
  const sign = (await import('node:crypto')).sign;
  try {
    const containerKeys = generateKeyPairSync('rsa', { modulusLength: 2048 }), executor = f.binding.members[EXECUTOR];
    const container = { jobName: 'Local\\SceneWorksFlux3-123-1-synthetic', publicKey: containerKeys.publicKey.export({ format: 'jwk' }),
      helper: { pid: 900, ppid: 101, startedAt: 'helper', executable: 'C:/Windows/powershell.exe', sessionId: 2 },
      ackFile: path.join(dir, 'container-synthetic.ack.json'), context: { sessionId: 2, ownerSid: 'S-1-5-21-123' } };
    f.binding.container = container;
    f.table.push({ pid: 900, ppid: 101, startedAt: 'helper', executable: 'C:/Windows/powershell.exe', command: 'powershell.exe', sessionId: 2 });
    f.table.find(p => p.pid === 110).ppid = 900;
    f.binding.containerAttestation = signed({ identity: structuredClone(f.binding.identity), container }, f.keys[EXECUTOR].privateKey);
    const ackPayload = JSON.stringify({ exitCode: 0, childPid: 901, jobName: container.jobName, identity: structuredClone(f.binding.identity), cleanupVerified: true, activeProcesses: 0, kind: 'flux3-container-cleanup-ack' });
    const ack = { payload: ackPayload, signature: sign('sha256', Buffer.from(ackPayload), containerKeys.privateKey).toString('base64') };
    const cleanup = signed({ identity: structuredClone(f.binding.identity), runner: EXECUTOR, proof: JSON.parse(ackPayload) }, f.keys[EXECUTOR].privateKey);
    const terminalPulse = signed({ identity: structuredClone(f.binding.identity), runner: EXECUTOR, jobId: executor.jobId, at: NOW, apiValidUntil: NOW + API_FRESH_MS, sequence: 2, status: 'safe_to_release', containerCreated: true, cleanupProofSha256: queueDigest(cleanup) }, f.keys[EXECUTOR].privateKey);
    const write = async (name, value) => writeFile(path.join(dir, name), JSON.stringify(value) + '\n', { flag: 'w' });
    const restore = async () => { await write('container-synthetic.ack.json', ack); await write(`${EXECUTOR}.cleanup.json`, cleanup); await write(`${EXECUTOR}.pulse.json`, terminalPulse); };
    await restore();
    // Healthy ACTIVE roster is still verified exactly as before.
    assert.deepEqual(verifyRoster(f.binding, f.pulses, f.table, f.controllerPid, NOW), [200, 300, 400]);
    // The executor's rows are gone after it published authenticated terminal cleanup. That is a safe
    // terminal conclusion, and it is available ONLY while the authenticated proof is intact.
    const departed = f.table.filter(p => ![100, 101].includes(p.pid));
    assert.equal(await authenticatedTerminalCleanup(dir, f.binding, f.binding.identity), true);
    assert.throws(() => verifyRoster(f.binding, f.pulses, departed, f.controllerPid, NOW), error => error.safeCode === 'QUEUE_PROCESS_ROW_MISSING');
    for (const [label, mutate] of [
      ['peer disappeared with no cleanup envelope', async () => { await rm(path.join(dir, `${EXECUTOR}.cleanup.json`)); }],
      ['cleanup has not proven a zero process tree', async () => { await write('container-synthetic.ack.json', { ...ack, payload: ack.payload.replace('"activeProcesses":0', '"activeProcesses":1') }); }],
      ['pulse is not bound to the cleanup envelope', async () => { await write(`${EXECUTOR}.pulse.json`, { ...terminalPulse, value: { ...terminalPulse.value, cleanupProofSha256: 'e'.repeat(64) } }); }],
      ['executor never created the owned container', async () => { await write(`${EXECUTOR}.pulse.json`, { ...terminalPulse, value: { ...terminalPulse.value, containerCreated: false } }); }],
      ['forged terminal pulse', async () => { await write(`${EXECUTOR}.pulse.json`, { ...terminalPulse, signature: Buffer.from('forged').toString('base64') }); }],
    ]) {
      await restore(); await mutate();
      assert.equal(await authenticatedTerminalCleanup(dir, f.binding, f.binding.identity), false, label);
      assert.throws(() => verifyRoster(f.binding, f.pulses, departed, f.controllerPid, NOW), error => error.safeCode === 'QUEUE_PROCESS_ROW_MISSING', label);
    }
  } finally { await rm(dir, { recursive: true, force: true, maxRetries: 10, retryDelay: 100 }); }
});

test('opaque GPU headroom predicate failures carry one allowlisted reason code and keep their checks', async () => {
  const gpu = 'GPU-' + '1'.repeat(8) + '-' + '1'.repeat(4) + '-' + '1'.repeat(4) + '-' + '1'.repeat(4) + '-' + '1'.repeat(12);
  const exercise = overrides => guardHardware({ backend: 'candle', gpuUuid: gpu }, [], {
    platform: 'win32', env: {}, controllerPid: 1, processTable: async () => [{ pid: 1, ppid: 0, command: 'node.exe' }],
    exec: async (command, args) => {
      if (command === 'powershell.exe') return { stdout: gpu + ' luid=0x00000001_0x00000002' };
      if (args.includes('--query-gpu=index,uuid,pci.bus_id')) return { stdout: '1, ' + gpu + ', 00000000:C1:00.0' };
      if (args.includes('--query-gpu=memory.total,memory.free')) return { stdout: overrides.counter ?? '97887, 96835' };
      if (args.includes('-x')) return { stdout: '<uuid>' + gpu + '</uuid><processes><process_info><pid>77</pid><type>C</type></process_info></processes>' };
      if (args.includes('--query-compute-apps=pid,gpu_uuid')) return { stdout: '77, ' + gpu };
      return { stdout: '' };
    },
    ...overrides.deps,
  });
  for (const [reason, overrides] of [
    ['GPU_HEADROOM_COUNTERS_UNAVAILABLE', { counter: 'n/a' }],
    ['GPU_HEADROOM_INVALID', { counter: '97887, 999999' }],
  ]) {
    let error; try { await exercise(overrides); } catch (caught) { error = caught; }
    assert.equal(error?.safeCode, reason);
    assert(SAFE_REASON_CODES.includes(error.safeCode));
    assert.equal(queueFailureReport(error, 'gpu_headroom').reason, reason);
    assert.equal(error.message, overrides.counter === 'n/a' ? 'GPU headroom counters unavailable' : 'valid positive GPU headroom required', 'the original predicate text is preserved');
  }
  // A foreign process with unmeasurable counters keeps its original refusal, now named precisely.
  let error; try { await exercise({ deps: { processTable: async () => [{ pid: 1, ppid: 0, command: 'node.exe' }, { pid: 77, ppid: 0, command: 'game.exe', executable: 'C:/game.exe', startedAt: 's', sessionId: 2 }] } }); } catch (caught) { error = caught; }
  assert.equal(error?.safeCode, 'GPU_FOREIGN_COUNTER_INVALID');
});
