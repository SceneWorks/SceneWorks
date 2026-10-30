import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdir, mkdtemp, writeFile, readFile, rm, stat, symlink } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import os from 'node:os';
import { runInNewContext } from 'node:vm';
import path from 'node:path';
import { ARTIFACTS, SOURCE, requiredCases, validatePlan, validateDelivery, validatePeakMetrics, hashFile, verifyInputs, compareNumeric, executedTests, decodeAV, decodedDelta, validateEvidence, normalizeLoadEvent, runProcess, runCase, verifyCampaign, reclaimCaseModelStorage, caseStorageReclaim, assertCaseFreeSpace, CASE_FREE_SPACE_HEADROOM_BYTES } from './h3-finetune-terminal.mjs';
import { runScenario, pointer, isSettledJobSnapshot, ASSET_SETTLE_GRACE_MS } from './h3-finetune-terminal-api.mjs';
import { awaitPeerLane, classifyPeerLane, assertPeerCanDeliver, PEER_APPEARANCE_GRACE_MS, waitForPeerAnchors, peerAnchorCandidates, anchorArtifactBase, anchorArtifactName, anchorArtifactAttempt } from './lib/h3-peer-lane.mjs';
const sw = 'a'.repeat(40), inf = 'b'.repeat(40);
const log = 'test result: ok. 1 passed; 0 failed; 0 ignored;';
function peakObservation(bytes, method, phase) {
  const start = '2026-10-24T00:00:00.000Z', end = '2026-10-24T00:00:00.050Z';
  const cumulative = ['mlx-allocator-cumulative-high-water', 'os-rss-high-water'].includes(method);
  return { method, sampled: method.startsWith('sampled-'), intervalMs: method.startsWith('sampled-') ? 50 : null,
    bounds: [...(phase === 'loading' ? ['full-case-load-inclusive-upper-bound'] : []), ...(method === 'sampled-device-used' ? ['observed-device-global-bound'] : [])],
    samples: [{ at: start, bytes: bytes / 2 }, { at: end, bytes }], window: { start, end, phase, coverage: phase === 'loading' ? 'full-case-load-inclusive' : 'full-case' },
    attribution: { pid: process.pid, deviceId: 'synthetic-device', scope: cumulative ? 'process-lifetime' : method === 'sampled-device-used' ? 'device' : 'process', meaning: 'Synthetic CPU observation fixture, not real measurement', includesPriorAllocations: cumulative, includesOtherProcesses: method === 'sampled-device-used' },
    blindspots: method.startsWith('sampled-') ? ['Transients between samples can be missed'] : cumulative ? ['Counter includes allocations before the observation window'] : [] };
}
async function fixture(t, group = 'artifact') {
  const dir = await mkdtemp(path.join(os.tmpdir(), 'h3-terminal-fixture-'));
  t.after(() => rm(dir, { recursive: true, force: true }));
  const input = path.join(dir, 'synthetic-checkpoint'); await writeFile(input, 'not a model');
  const hash = await hashFile(input);
  const p = { version: 1, sceneworks: sw, inference: inf, source: SOURCE, output: dir, responsibleTask: 'sc-26691', nextAuthorizedAction: 'Coordinator authorizes exact next Actions case; no automatic retry.', cases: requiredCases().map(c => ({ ...c, command: ['node', 'controlled-fixture.mjs'], timeoutSeconds: 10, evidence: 'evidence.json', loadHostBudgetBytes: 1000, loadDeviceBudgetBytes: 2000, minimumControlDelta: 0.1, controlStream: 'video', referenceRevision: 'c'.repeat(40), referenceCommand: ['node', 'controlled-reference.mjs'], referenceOutput: 'reference.json', tolerance: { atol: 0.001, rtol: 0.001 }, inputs: [{ path: input, sha256: hash, role: 'checkpoint', artifact: c.artifact }, { path: input, sha256: hash, role: 'reference' }] })) };
  const c = p.cases.find(c => c.group === group);
  const e = { status: 'executed', sceneworks: sw, inference: inf, caseId: c.id, backend: c.backend, source: SOURCE, artifact: c.artifact, hardware: { device: 'synthetic CPU fixture only', driver: 'fixture' }, runtimeIdentity: { checkpointSha256: hash, planId: 'synthetic-plan', semanticDigest: `sha256:${hash}`, topology: 'fixture', quantization: 'fixture', components: [], adapters: [], operation: 'fixture', samplingPolicy: { steps: 2, schedulerShift: null }, seed: 42 }, metrics: { hostPeakBytes: 100, devicePeakBytes: 200, loadHostPeakBytes: 90, loadDevicePeakBytes: 180, observations: { host: peakObservation(100, 'sampled-process-rss', 'case'), device: peakObservation(200, 'sampled-native-allocated', 'case'), loadHost: peakObservation(90, 'sampled-process-rss', 'loading'), loadDevice: peakObservation(180, 'sampled-native-allocated', 'loading') } }, files: ['trace.json'], av: 'synthetic-av.bin', control: { intervention: 'replace conditioning', checkpointSha256: hash, seed: 42, av: 'control-av.bin' }, reference: { revision: c.referenceRevision, sha256: hash }, numeric: { candidate: 'candidate.json', reference: 'reference.json' } };
  e.testInputSha256 = hash;
  e.completedJob = { id: 'synthetic-job', status: 'completed', providerId: 'synthetic-native-provider', output: e.av };
  e.loadEvent = { event: 'h3_import_prepared_load', jobId: e.completedJob.id, provider: e.completedJob.providerId, backend: c.backend, native_backend: c.backend, seal_validated: true, cache_state: 'Cold', identity: JSON.stringify({ checkpointId: 'synthetic-checkpoint', planId: e.runtimeIdentity.planId, semanticDigest: `sha256:${hash}`, assembly: {}, sourceFiles: [{ path: input, sha256: hash }] }), prepared_sources: JSON.stringify([{ path: input, sha256: hash }]), operation: e.runtimeIdentity.operation, seed: 42, steps: 'Some(2)', scheduler_shift: 'None' };
  await writeFile(path.join(dir, 'source-catalog.json'), JSON.stringify({ sha: SOURCE.revision, siblings: ARTIFACTS.map(rfilename => ({ rfilename, lfs: { sha256: hash } })) }));
  await writeFile(path.join(dir, 'trace.json'), '{}');
  await writeFile(path.join(dir, e.av), 'synthetic primary AV');
  await writeFile(path.join(dir, e.control.av), 'synthetic control AV');
  await writeFile(path.join(dir, 'candidate.json'), '[1,2,3]');
  await writeFile(path.join(dir, 'reference.json'), '[1,2,3]');
  await writeFile(path.join(dir, 'reference.log'), 'controlled executable reference');
  await writeFile(path.join(dir, 'reference-execution.json'), JSON.stringify({ command: c.referenceCommand, revision: c.referenceRevision, outputSha256: await hashFile(path.join(dir, 'reference.json')) }));
  const save = () => writeFile(path.join(dir, c.evidence), JSON.stringify(e)); await save();
  const decode = async file => ({ sha256: await hashFile(file), streams: 'controlled process fixture, NOT real AV' });
  return { p, c, e, dir, save, decode, verify: () => validateEvidence(p, c, dir, log, decode, async () => 0.25) };
}
test('complete exact inventory includes eight frozen artifact/backend executions and every mechanism', async t => {
  const { p } = await fixture(t);
  assert.equal(validatePlan(p), p);
  assert.equal(requiredCases().filter(c => c.group === 'artifact').length, 8);
  assert.equal(new Set(ARTIFACTS).size, 4);
  for (const mutate of [p => p.cases.pop(), p => p.cases[0].artifact = 'stock.safetensors', p => p.source.revision = 'd'.repeat(40)]) {
    const copy = structuredClone(p); mutate(copy); assert.throws(() => validatePlan(copy));
  }
});
test('actual streamed content detects input drift, not filename or metadata', async t => {
  const { c } = await fixture(t); await verifyInputs(c);
  await writeFile(c.inputs[0].path, 'different checkpoint');
  await assert.rejects(verifyInputs(c), /content changed/);
});
test('numeric comparison discriminates scaling, NaN, empty and tolerance mutations', () => {
  assert.equal(compareNumeric([1, 2.001], [1, 2], { atol: 0.01, rtol: 0 }).elements, 2);
  for (const [a, b, tol] of [[[2,4],[1,2],{atol:0.01,rtol:0}], [[NaN],[1],{atol:0,rtol:0}], [[],[],{atol:0,rtol:0}], [[1],[1],{atol:-1,rtol:0}]]) assert.throws(() => compareNumeric(a,b,tol));
});
test('test execution evidence rejects ignored, zero, missing and failure summaries', () => {
  assert.equal(executedTests(log), 1);
  for (const s of ['', 'test result: ok. 0 passed; 0 failed; 0 ignored;', 'test result: ok. 1 passed; 0 failed; 1 ignored;', 'test result: ok. 1 passed; 1 failed; 0 ignored;']) assert.throws(() => executedTests(s));
});
for (const [name, mutate] of [
  ['wrong source revision', e => e.inference = sw],
  ['wrong checkpoint', e => e.runtimeIdentity.checkpointSha256 = 'f'.repeat(64)],
  ['missing sealed checkpoint identity', e => { const identity = JSON.parse(e.loadEvent.identity); identity.sourceFiles = []; e.loadEvent.identity = JSON.stringify(identity); }],
  ['unvalidated cache seal', e => e.loadEvent.seal_validated = false],
  ['CPU fallback', e => e.loadEvent.backend = 'cpu'],
  ['wrong native backend', e => e.loadEvent.native_backend = 'cuda'],
  ['wrong native job binding', e => e.loadEvent.jobId = 'unrelated-job'],
  ['changed observed sampling policy', e => e.loadEvent.steps = 'Some(99)'],
  ['wrong output binding', e => e.completedJob.output = 'unrelated-av.bin'],
  ['missing AV', e => delete e.av],
  ['missing metrics', e => delete e.metrics],
  ['residency deltas mislabeled as peaks', e => e.metrics.observations.device.method = 'pre-post-residency-delta'],
  ['missing load peak', e => delete e.metrics.loadDevicePeakBytes],
  ['loading budget exceeded', e => { e.metrics.loadHostPeakBytes = 1001; e.metrics.observations.loadHost.samples[1].bytes = 1001; }],
  ['missing runtime component hash', e => e.runtimeIdentity.components = [{ sha256: 'e'.repeat(64) }]],
  ['not actually executed', e => e.status = 'planned'],
  ['missing hardware', e => e.hardware = {}],
]) test(`receipt mutation rejects ${name}`, async t => {
  const f = await fixture(t); await f.verify(); mutate(f.e); await f.save(); await assert.rejects(f.verify());
});
// sc-26691, CUDA run 40797546305 case 0: the render completed and the validator then refused the
// product's own identity. Values below are the shapes that run's evidence.json actually carried — a
// TYPED semantic digest, bare-hex content hashes, Option-rendered sampling fields — never guessed ones.
test('evidence accepts the product-emitted typed semantic digest and refuses bare hex', async t => {
  const realEvent = {
    event: 'h3_import_prepared_load', jobId: 'job_48c3e106c8ea4437d620d838cc081ca7', provider: 'minimax_h3', backend: 'cuda:1', native_backend: 'cuda',
    seal_validated: true, cache_state: 'Cold', operation: 'reference_to_video', seed: 314159, steps: 'Some(4)', scheduler_shift: 'Some(12.0)',
    identity: JSON.stringify({ checkpointId: 'managed/real', planId: 'checkpoint-plan-4e923072a296085550518dce236b08f8b0840028cc65e9cb2769b4f353542fa6',
      semanticDigest: 'sha256:f4e8b5db8d458b58abc83650c144c3fa4815ccfa7185d28f32cabcc8eef1e23b', assembly: {},
      sourceFiles: [{ path: 'checkpoint.safetensors', sha256: 'db3ec568863db6f5ec1840210d41653afef73d4df7b0cb489c52ad458b6f8a7f' }] }),
    prepared_sources: JSON.stringify([{ path: 'checkpoint.safetensors', sha256: 'db3ec568863db6f5ec1840210d41653afef73d4df7b0cb489c52ad458b6f8a7f' }]),
  };
  const closure = normalizeLoadEvent(realEvent);
  assert.equal(closure.steps, 4); assert.equal(closure.schedulerShift, 12); assert.equal(closure.seed, 314159);
  assert.match(closure.identity.semanticDigest, /^sha256:[a-f0-9]{64}$/);
  const f = await fixture(t);
  assert.match(f.e.runtimeIdentity.semanticDigest, /^sha256:[a-f0-9]{64}$/, 'the fixture carries the product form');
  await f.verify();
  const bare = f.e.runtimeIdentity.semanticDigest.slice('sha256:'.length);
  f.e.runtimeIdentity.semanticDigest = bare;
  const identity = JSON.parse(f.e.loadEvent.identity); identity.semanticDigest = bare; f.e.loadEvent.identity = JSON.stringify(identity);
  await f.save();
  await assert.rejects(f.verify(), /typed sha256:<64 hex> commitment/);
});
test('numeric case executes scale-sensitive array assertion and pinned reference revision', async t => {
  const f = await fixture(t, 'numeric'); const result = await f.verify(); assert.equal(result.numeric.elements, 3);
  await writeFile(path.join(f.dir, 'candidate.json'), '[2,4,6]'); await assert.rejects(f.verify(), /numeric mismatch/);
  await writeFile(path.join(f.dir, 'candidate.json'), '[1,2,3]'); f.e.reference.revision = sw; await f.save(); await assert.rejects(f.verify(), /reference revision drift/);
  await assert.rejects(validateEvidence(f.p, f.c, f.dir, 'test result: ok. 0 passed; 0 failed; 1 ignored;', f.decode), /ignored/);
});
test('conditioning controls reject dropped conditioning and changed seed/checkpoint', async t => {
  const f = await fixture(t, 'capability'); await f.verify();
  await assert.rejects(validateEvidence(f.p, f.c, f.dir, log, f.decode, async () => 0), /ignored/);
  f.e.control.seed = 43; await f.save(); await assert.rejects(f.verify());
  f.e.control.seed = 42; f.e.control.checkpointSha256 = '1'.repeat(64); await f.save(); await assert.rejects(f.verify());
});
test('AV verifier fully decodes both streams and rejects metadata-only/missing audio', async t => {
  const f = await fixture(t); const calls = [];
  const command = async (bin, args) => { calls.push({ bin, args }); return { stdout: JSON.stringify({ streams: [{ codec_type: 'video', nb_read_frames: '4', duration: '1' }, { codec_type: 'audio', nb_read_frames: '4', duration: '1' }] }) }; };
  await decodeAV(path.join(f.dir, f.e.av), command);
  assert.equal(calls[1].bin, 'ffmpeg'); assert.ok(calls[1].args.includes('0:a:0')); assert.ok(calls[1].args.includes('-xerror'));
  await assert.rejects(decodeAV(path.join(f.dir, f.e.av), async () => ({ stdout: '{"streams":[{"codec_type":"video","nb_read_frames":"4","duration":"1"}]}' })), /audio/);
});
test('decoded scale-sensitive intervention metric is computed from media process bytes', async () => {
  const command = async (_bin, args) => ({ stdout: Buffer.from(args.includes('primary') ? [0, 0, 0] : [255, 255, 255]) });
  assert.equal(await decodedDelta('primary', 'control', 'video', command), 1);
  assert.equal(await decodedDelta('primary', 'primary', 'video', command), 0);
});
test('controlled subprocess executes, propagates failures and refuses log overwrite', async t => {
  const { dir } = await fixture(t);
  const options = { cwd: dir, env: process.env, logPath: path.join(dir, 'process.log'), timeoutSeconds: 2 };
  await runProcess([process.execPath, '-e', 'console.log("actually ran")'], options);
  assert.match(await readFile(options.logPath, 'utf8'), /actually ran/);
  await assert.rejects(runProcess([process.execPath, '-e', 'process.exit(0)'], options), /EEXIST/);
  await assert.rejects(runProcess([process.execPath, '-e', 'process.exit(7)'], { ...options, logPath: path.join(dir, 'failed.log') }), /failed/);
  await assert.rejects(runProcess([process.execPath, '-e', 'setTimeout(()=>{},100)'], { ...options, timeoutSeconds: 0.01, logPath: path.join(dir, 'timeout.log') }), /timeout/);
});
// A wedged case process (its drain never returns, its live children keep it alive) once held the
// step open for 6.5 h (run 40797515762): the timer only logged H3_CASE_TIMEOUT. The harness now
// terminates the process it spawned, by its own child PID, after the deadline plus a grace.
const isAlive = pid => { try { process.kill(pid, 0); return true; } catch { return false; } };
test('a case process that ignores its deadline is terminated by its own PID within the bound', async t => {
  const { dir } = await fixture(t);
  // Ignores SIGTERM where signals exist (POSIX), so the SIGKILL escalation is exercised there too.
  const script = "process.on('SIGTERM',()=>console.log('ignoring SIGTERM'));console.log('pid '+process.pid);setInterval(()=>{},1000);";
  const started = Date.now(), logPath = path.join(dir, 'wedged.log');
  await assert.rejects(runProcess([process.execPath, '-e', script], { cwd: dir, env: process.env, logPath, timeoutSeconds: 0.2, graceSeconds: 0.3, killGraceSeconds: 0.5 }), /terminated its own case process \d+/);
  const elapsed = Date.now() - started, log = await readFile(logPath, 'utf8');
  assert.ok(elapsed >= 500 && elapsed < 10000, `bounded termination took ${elapsed} ms`);
  assert.match(log, /H3_CASE_TIMEOUT/); assert.match(log, /H3_CASE_TERMINATED: case process \d+/);
  const pid = Number(/pid (\d+)/.exec(log)[1]); assert.equal(isAlive(pid), false, 'the case process is gone');
  if (process.platform !== 'win32') assert.match(log, /H3_CASE_KILLED/);
});
test('a case process exiting while its bound is read never arms another escalation timer', async t => {
  const { dir } = await fixture(t), logPath = path.join(dir, 'race.log');
  let release;
  // The bound read is still in flight when the process exits; escalate() must notice.
  const readBound = () => new Promise(resolve => { release = () => resolve(JSON.stringify({ kind: 'h3-case-shutdown-bound', pid: 0, deadlineMs: Date.now() + 3600000 })); });
  const started = Date.now();
  const run = runProcess([process.execPath, '-e', 'setTimeout(()=>{},400)'], { cwd: dir, env: process.env, logPath, timeoutSeconds: 0.05, graceSeconds: 0.05, killGraceSeconds: 0.5, shutdownBoundPath: path.join(dir, 'bound.json'), readBound });
  await new Promise(resolve => setTimeout(resolve, 200)); assert.ok(release, 'escalation reached the bound read');
  await assert.rejects(run, /case exceeded timeout/); release();
  assert.ok(Date.now() - started < 5000, 'run-case is never held open by an escalation timer');
  assert.doesNotMatch(await readFile(logPath, 'utf8'), /H3_CASE_TERMINATED/);
});
test('a clean exit just before the deadline is not misreported while stdio drains', async t => {
  const { dir } = await fixture(t);
  // A detached grandchild keeps the inherited pipes open past the deadline; the case process itself exited cleanly first.
  const script = "require('child_process').spawn(process.execPath,['-e','setTimeout(()=>{},1500)'],{stdio:'inherit',detached:true}).unref();";
  await runProcess([process.execPath, '-e', script], { cwd: dir, env: process.env, logPath: path.join(dir, 'clean.log'), timeoutSeconds: 0.6 });
  assert.doesNotMatch(await readFile(path.join(dir, 'clean.log'), 'utf8'), /H3_CASE_TIMEOUT/);
});
test('on Windows the last-resort kill is TerminateProcess without a meaningless SIGKILL grace', async t => {
  const { dir } = await fixture(t), logPath = path.join(dir, 'win.log');
  await assert.rejects(runProcess([process.execPath, '-e', 'setInterval(()=>{},1000)'], { cwd: dir, env: process.env, logPath, timeoutSeconds: 0.1, graceSeconds: 0.1, killGraceSeconds: 0.2, platform: 'win32' }), /terminated its own case process/);
  const log = await readFile(logPath, 'utf8'); assert.match(log, /TerminateProcess of the harness's own child PID/); assert.doesNotMatch(log, /SIGKILL/);
});
test('case and lane deadlines stay inside the Actions job timeout and leave the shutdown window', async () => {
  const { caseDeadlines, laneWorkDeadline, CASE_SHUTDOWN_RESERVE_MS, ACTIONS_JOB_TIMEOUT_MS, LANE_FINALIZE_RESERVE_MS, CASE_KILL_GRACE_SECONDS } = await import('./h3-finetune-terminal.mjs');
  const { DEFAULT_CANCEL_DRAIN_SECONDS, postDrainShutdownMs } = await import('./h3-finetune-terminal-runtime.mjs');
  assert.ok(CASE_SHUTDOWN_RESERVE_MS >= DEFAULT_CANCEL_DRAIN_SECONDS * 1000 + postDrainShutdownMs() + CASE_KILL_GRACE_SECONDS * 1000, 'the reserve fits a default drain and the bounded owned shutdown');
  const workflow = await readFile(new URL('../.github/workflows/h3-finetune-terminal.yml', import.meta.url), 'utf8');
  assert.equal(Number(/\n    timeout-minutes: (\d+)\n/.exec(workflow)[1]) * 60000, ACTIONS_JOB_TIMEOUT_MS, 'ACTIONS_JOB_TIMEOUT_MS mirrors the terminal job timeout');
  // A first dispatch: lane start == job start (+ the setup steps), so the 24 h budget ends with the job timeout.
  const start = Date.parse('2026-10-29T00:00:00Z'), budget = { deadline: new Date(start + 86400000).toISOString() };
  const work = laneWorkDeadline(budget, { jobStartedAt: new Date(start + 60000).toISOString() });
  assert.equal(work, start + 86400000 - LANE_FINALIZE_RESERVE_MS); assert.ok(work < start + ACTIONS_JOB_TIMEOUT_MS - 30 * 60000, 'the always() evidence steps keep their reserve');
  // A later dispatch of the same lane: its own job timeout is later; the persisted budget governs.
  assert.equal(laneWorkDeadline(budget, { jobStartedAt: new Date(start + 20 * 3600000).toISOString() }), start + 86400000 - LANE_FINALIZE_RESERVE_MS);
  // No lane budget (a single case): the job timeout alone.
  assert.equal(laneWorkDeadline(null, { jobStartedAt: new Date(start).toISOString() }), start + ACTIONS_JOB_TIMEOUT_MS - LANE_FINALIZE_RESERVE_MS);
  // A budget-capped case gets its whole shutdown window between soft and hard.
  const capped = caseDeadlines({ timeoutSeconds: 7200, now: work - 30 * 60000, workDeadline: work });
  assert.equal(capped.hard, work); assert.equal(capped.hard - capped.soft, CASE_SHUTDOWN_RESERVE_MS);
  const roomy = caseDeadlines({ timeoutSeconds: 7200, now: start, workDeadline: work });
  assert.equal(roomy.soft, start + 7200000); assert.equal(roomy.hard, work, 'a cancelled job may take the worker budget up to the lane work deadline');
  const source = await readFile(new URL('./h3-finetune-terminal.mjs', import.meta.url), 'utf8');
  assert.ok(source.includes('H3_CASE_HARD_DEADLINE_MS:String(caseHardDeadline)') && source.includes('hardDeadlineMs: caseHardDeadline'), 'runCase hands the same hard deadline to the case process and to runProcess');
});
// Real CUDA attempt 1, run 40903166111: lane budget from 15:35:56, job start 15:35:51, own anchors
// published 16:58:40. Reference anchor (case 1) 2687313 ms with steps 53558/53581/53799/54018 ms;
// prompt anchor (case 15) 857867 ms with 17871/17968/18236/18747 ms. Remaining step counts are the
// plan's declared samplings (50-step: cases 2, 4, 20, 25; 8-step: 17, 18, 21; the rest 4).
test('the peer-anchor wait is sized from real measured per-case work and lets CUDA wait for a ~3 h MLX anchor', async () => {
  const { peerAnchorWaitDeadline, anchorTimingFromLogs, estimateCaseMs, remainingLaneCases, CASE_SHUTDOWN_RESERVE_MS, LANE_FINALIZE_RESERVE_MS } = await import('./h3-finetune-terminal.mjs');
  const { OPERATIONAL_CASE_DEFS } = await import('./h3-finetune-terminal-reduced.mjs');
  const stepLog = ms => ms.map((elapsed, i) => JSON.stringify({ message: 'minimax-h3: denoise step done', step: i + 1, total: ms.length, elapsed_ms: elapsed })).join('\n');
  const timings = {
    reference: anchorTimingFromLogs({ wallMs: 2687313, steps: 4, logText: stepLog([53558, 53581, 53799, 54018]) }),
    prompt: anchorTimingFromLogs({ wallMs: 857867, steps: 4, logText: stepLog([17871, 17968, 18236, 18747]) }),
  };
  assert.equal(timings.reference.basis, 'measured-step-split'); assert.equal(timings.reference.fixedMs, 2687313 - 214956);
  const steps = { 2: 50, 4: 50, 17: 8, 18: 8, 20: 50, 21: 8, 25: 50 };
  const p = { cases: OPERATIONAL_CASE_DEFS.flatMap(def => ['cuda', 'mlx'].map(backend => ({ ...def, backend, id: `${backend}.reduced.${def.slug}`, ...(def.kind === 'actual-artifact-probe' ? {} : { declaredSourceBindings: [{ sampling: { steps: steps[def.index] ?? 4 } }] }) }))) };
  assert.equal(remainingLaneCases(p, 'cuda').length, 23, 'selected backend: every remaining case');
  assert.equal(remainingLaneCases(p, 'cuda', { displaced: true }).length, 16, 'displaced backend omits the seven broad-matrix cases');
  // Per-case, not the longest anchor applied blindly: a 4-step hybrid case costs its own anchor, a
  // 50-step reference case its fixed part plus 50 measured steps.
  const byIndex = index => estimateCaseMs(p.cases.find(c => c.backend === 'cuda' && c.index === index), timings);
  assert.equal(Math.round(byIndex(16)), 857867); assert.equal(Math.round(byIndex(5)), 2687313);
  assert.equal(Math.round(byIndex(2)), Math.round(timings.reference.fixedMs + 50 * timings.reference.stepMs));
  const budget = { deadline: '2026-10-30T15:35:56.000Z' }, published = Date.parse('2026-10-29T16:58:40Z');
  const wait = peerAnchorWaitDeadline(p, 'cuda', budget, { jobStartedAt: '2026-10-29T15:35:51Z', timings });
  assert.equal(wait.workDeadline, Date.parse('2026-10-30T15:35:51Z') - LANE_FINALIZE_RESERVE_MS);
  assert.equal(wait.deadline, wait.workDeadline - wait.remainingMs - CASE_SHUTDOWN_RESERVE_MS);
  assert.ok(Math.abs(wait.remainingMs / 3600000 - 15.10) < 0.01, `remaining CUDA work ${(wait.remainingMs / 3600000).toFixed(2)} h`);
  assert.equal(new Date(wait.deadline).toISOString().slice(0, 16), '2026-10-29T23:24');
  const waitHours = (wait.deadline - published) / 3600000;
  assert.ok(waitHours > 6 && waitHours < 7, `CUDA may wait ${waitHours.toFixed(2)} h for the MLX anchors`);
  assert.ok(wait.deadline - published > 3 * 3600000, 'a ~3 h MLX anchor publication fits');
  // The formula this replaces (remaining x (longest anchor + 20 min)) would have refused at once.
  assert.ok(published > wait.workDeadline - 23 * (2687313 + CASE_SHUTDOWN_RESERVE_MS));
  // An engine without per-step timing is charged pessimistically (whole wall per step), labelled.
  const flat = anchorTimingFromLogs({ wallMs: 4 * 600000, steps: 4 });
  assert.equal(flat.stepMs, 600000); assert.equal(flat.fixedMs, 0); assert.match(flat.basis, /no per-step timing/);
  assert.throws(() => peerAnchorWaitDeadline(p, 'cuda', budget, {}), /measured anchor timings/);
});
test('own anchor timings come from the retained anchor case receipts and worker logs', async t => {
  const { ownAnchorTimings } = await import('./h3-finetune-terminal.mjs');
  const output = await mkdtemp(path.join(os.tmpdir(), 'sc26691-anchor-timing-')); t.after(() => rm(output, { recursive: true, force: true }));
  const cases = [[1, 'artifact-0-ordered-mixed-baseline-ref-turbo', '2026-10-29T15:57:52.037Z', '2026-10-29T16:42:39.350Z', [53558, 53581, 53799, 54018]], [15, 'real-hybrid-text', '2026-10-29T16:44:20.489Z', '2026-10-29T16:58:38.356Z', [17871, 17968, 18236, 18747]]];
  const p = { output, cases: cases.map(([index, slug]) => ({ index, slug, backend: 'cuda', kind: 'full-av', id: `cuda.reduced.${slug}`, declaredSourceBindings: [{ sampling: { steps: 4 } }] })) };
  for (const [index, slug, started, completed, steps] of cases) {
    const dir = path.join(output, `cuda.reduced.${slug}`); await mkdir(dir);
    await writeFile(path.join(dir, 'execution.json'), JSON.stringify({ status: 'passed', started, completed }));
    // The real worker line shape, interleaved with unrelated events.
    await writeFile(path.join(dir, 'worker-1.log'), [JSON.stringify({ event: 'artifact_seal_hash_progress' }), ...steps.map((elapsed_ms, i) => JSON.stringify({ message: 'minimax-h3: denoise step done', step: i + 1, total: 4, elapsed_ms, span: { name: 'h3_import_execution' } }))].join('\n') + '\n');
    void index;
  }
  const timings = await ownAnchorTimings(p, 'cuda');
  assert.equal(timings.reference.wallMs, 2687313); assert.equal(timings.reference.stepMs, 53739); assert.equal(timings.prompt.wallMs, 857867);
  assert.equal(timings.prompt.basis, 'measured-step-split');
});
test('the case process exits deterministically when owned handles outlive its bounded shutdown', async t => {
  const { dir } = await fixture(t);
  const runtime = new URL('./h3-finetune-terminal-runtime.mjs', import.meta.url).href;
  const script = `import(${JSON.stringify(runtime)}).then(({scheduleDeterministicExit})=>{setInterval(()=>{},1000);process.exitCode=3;scheduleDeterministicExit(200);});`;
  const started = Date.now();
  await assert.rejects(runProcess([process.execPath, '-e', script], { cwd: dir, env: process.env, logPath: path.join(dir, 'linger.log'), timeoutSeconds: 20 }), /case process failed/);
  assert.ok(Date.now() - started < 15000); assert.match(await readFile(path.join(dir, 'linger.log'), 'utf8'), /H3_CASE_FORCED_EXIT: owned handles outlived the bounded shutdown; exiting with 3/);
});
test('a recorded drain bound extends the post-deadline window for the same child only, capped by the hard deadline', async t => {
  const { dir } = await fixture(t);
  const boundPath = path.join(dir, 'shutdown-bound.json');
  // The child records the worker-advertised drain bound (1.5 s out) for ITS OWN pid, then wedges.
  const script = `require('fs').writeFileSync(${JSON.stringify(boundPath)},JSON.stringify({kind:'h3-case-shutdown-bound',pid:process.pid,jobId:'j',source:'worker-advertised-stall-budget',deadlineMs:Date.now()+1500}));setInterval(()=>{},1000);`;
  let started = Date.now();
  await assert.rejects(runProcess([process.execPath, '-e', script], { cwd: dir, env: process.env, logPath: path.join(dir, 'bounded.log'), timeoutSeconds: 0.1, graceSeconds: 0.1, killGraceSeconds: 0.5, shutdownBoundPath: boundPath }), /terminated its own case process/);
  assert.ok(Date.now() - started >= 1400, 'honoured the recorded drain bound');
  assert.match(await readFile(path.join(dir, 'bounded.log'), 'utf8'), /within its recorded bound \(worker-advertised-stall-budget\)/);
  // A bound for a foreign pid is ignored: termination follows the plain grace.
  const foreign = `require('fs').writeFileSync(${JSON.stringify(boundPath)},JSON.stringify({kind:'h3-case-shutdown-bound',pid:1,deadlineMs:Date.now()+60000}));setInterval(()=>{},1000);`;
  started = Date.now();
  await assert.rejects(runProcess([process.execPath, '-e', foreign], { cwd: dir, env: process.env, logPath: path.join(dir, 'foreign.log'), timeoutSeconds: 0.1, graceSeconds: 0.2, killGraceSeconds: 0.5, shutdownBoundPath: boundPath }), /terminated/);
  assert.ok(Date.now() - started < 10000);
  // The hard (lane) deadline caps any recorded bound.
  const greedy = `require('fs').writeFileSync(${JSON.stringify(boundPath)},JSON.stringify({kind:'h3-case-shutdown-bound',pid:process.pid,source:'worker-advertised-stall-budget',deadlineMs:Date.now()+3600000}));setInterval(()=>{},1000);`;
  started = Date.now();
  await assert.rejects(runProcess([process.execPath, '-e', greedy], { cwd: dir, env: process.env, logPath: path.join(dir, 'capped.log'), timeoutSeconds: 0.1, graceSeconds: 0.2, killGraceSeconds: 0.5, shutdownBoundPath: boundPath, hardDeadlineMs: Date.now() + 1000 }), /terminated/);
  assert.ok(Date.now() - started < 10000, 'the lane deadline caps the recorded bound');
  assert.match(await readFile(path.join(dir, 'capped.log'), 'utf8'), /capped by the hard deadline/);
});
test('missing executions cannot yield campaign acceptance; local native run is refused', async t => {
  const { p } = await fixture(t); await assert.rejects(verifyCampaign(p), /ENOENT/);
  if (process.env.GITHUB_ACTIONS !== 'true') await assert.rejects(runCase(p, p.cases[0].id, '.', '.'), /Actions-only/);
});
const response = (value, status = 200) => ({ status, text: async () => JSON.stringify(value) });
const scenario = () => ({ baseUrl: 'http://127.0.0.1:43210', caseId: 'mlx.generalization.ltx-add-select-generate', timeoutSeconds: 5, steps: [
  { id: 'import', method: 'POST', path: '/api/v1/models/import', expectedStatus: 201, body: { name: 'Imported H3', sourcePath: '/private-fixture/model.safetensors', baseModel: 'minimax_h3', minimaxH3Partition: 'ref2va' }, assert: [{ pointer: '/id', equals: 'import-job' }] },
  { id: 'installed', waitJob: { $capture: 'import/id' }, assert: [{ pointer: '/result/modelId', equals: 'imported-model' }] },
  { id: 'generate', method: 'POST', path: '/api/v1/video/jobs', expectedStatus: 201, body: { projectId: 'private-project', model: { $capture: 'installed/result/modelId' }, mode: 'reference_to_video', prompt: 'A person waves', referenceAssetIds: ['private-ref'] }, assert: [{ pointer: '/payload/model', equals: 'imported-model' }] },
  { id: 'generated', waitJob: { $capture: 'generate/id' }, assert: [{ pointer: '/status', equals: 'completed' }] },
] });
test('real app sequence uses existing import/select/video job endpoints and captured imported model', async () => {
  const calls = [];
  const results = [response({ id: 'import-job' }, 201), response({ status: 'completed', result: { modelId: 'imported-model' } }), response({ id: 'video-job', payload: { model: 'imported-model' } }, 201), response({ status: 'completed' })];
  const result = await runScenario(scenario(), { fetchImpl: async (url, options) => { calls.push([url.pathname, options]); return results.shift(); } });
  assert.equal(calls[0][0], '/api/v1/models/import'); assert.equal(calls[2][0], '/api/v1/video/jobs');
  assert.equal(JSON.parse(calls[2][1].body).model, 'imported-model');
  assert.equal(result.transcript.filter(r => r.assertion).length, 4);
});
test('API failure/timeout cancels actual submitted job, not the GPU process', async () => {
  const s = scenario(); s.steps = s.steps.slice(0, 2); const calls = [];
  await assert.rejects(runScenario(s, { fetchImpl: async (url, options) => { calls.push([url.pathname, options.method]); return calls.length === 1 ? response({ id: 'import-job' }, 201) : response({ status: 'failed' }); } }), /did not complete/);
  assert.deepEqual(calls.at(-1), ['/api/v1/jobs/import-job/cancel', 'POST']);
  let time = 0;
  await assert.rejects(runScenario(s, { now: () => time, sleep: async ms => { time += ms; }, fetchImpl: async url => url.pathname.endsWith('/import') ? response({ id: 'import-job' }, 201) : response({ status: 'running' }) }), /timeout/);
});
// Trimmed from CUDA campaign 40874652324, cuda.reduced.real-hybrid-text: the harness captured
// revision 82 of the completed hybrid video job (raw worker `assetWrites`, before the API persisted
// them) and failed "completed actual result asset IDs required"; drained revision 83 carried them.
const hybridCompletedUnpersisted = { id: 'job_dbeee8fb5a82661c1c2caee94a062c5d', type: 'video_generate', status: 'completed', projectId: 'project_c3805fe54b838fc219747e2e5ff5e7ec', payload: { model: 'imported_sc26691_hybrid_single_checkpoint' }, revision: 82,
  result: { adapter: 'candle_minimax_h3', model: 'imported_sc26691_hybrid_single_checkpoint', expectedCount: 1, generationSetId: 'genset_322d2a47096d4f64afe6f91ecba1fc5b', generationSet: { id: 'genset_322d2a47096d4f64afe6f91ecba1fc5b', count: 1, mode: 'text_to_video' },
    assetWrites: [{ assetId: 'asset_4a13efad5e514debb937506901b123af', type: 'video', mimeType: 'video/mp4', mediaPath: 'assets/videos/genset_322d2a47096d4f64afe6f91ecba1fc5b/2026-10-29_imported-sc26691-hybrid-single-checkpoint_a-red-cube-slides-right-across-a-white-tab.mp4' }] } };
const hybridCompletedPersisted = { ...hybridCompletedUnpersisted, revision: 83,
  result: { adapter: 'candle_minimax_h3', model: 'imported_sc26691_hybrid_single_checkpoint', expectedCount: 1, generationSetId: 'genset_322d2a47096d4f64afe6f91ecba1fc5b', assetIds: ['asset_4a13efad5e514debb937506901b123af'],
    assets: [{ id: 'asset_4a13efad5e514debb937506901b123af', type: 'video', projectId: 'project_c3805fe54b838fc219747e2e5ff5e7ec', generationSetId: 'genset_322d2a47096d4f64afe6f91ecba1fc5b', file: { mimeType: 'video/mp4', width: 1088, height: 544, fps: 24, frameCount: 124, hasAudio: true, path: 'assets/videos/genset_322d2a47096d4f64afe6f91ecba1fc5b/2026-10-29_imported-sc26691-hybrid-single-checkpoint_a-red-cube-slides-right-across-a-white-tab.mp4' } }] } };
test('a completed job is captured only after the API persisted its reported assetWrites', async () => {
  assert.equal(isSettledJobSnapshot(hybridCompletedUnpersisted), false);
  assert.equal(isSettledJobSnapshot(hybridCompletedPersisted), true);
  assert.equal(isSettledJobSnapshot({ status: 'completed', result: { modelId: 'imported-model' } }), true, 'an import result has nothing to persist');
  assert.equal(isSettledJobSnapshot({ status: 'completed', result: { assetWrites: [] } }), true);
  assert.equal(isSettledJobSnapshot({ status: 'failed', result: { assetWrites: [{ type: 'video' }] } }), true, 'only a completed job waits on persistence');
  const s = scenario(); s.steps = [{ id: 'generated', waitJob: 'job_dbeee8fb5a82661c1c2caee94a062c5d', assert: [{ pointer: '/status', equals: 'completed' }] }];
  const snapshots = [{ ...hybridCompletedUnpersisted, status: 'running' }, hybridCompletedUnpersisted, hybridCompletedUnpersisted, hybridCompletedPersisted];
  const result = await runScenario(s, { sleep: async () => {}, fetchImpl: async () => response(snapshots.shift()) });
  assert.equal(snapshots.length, 0);
  assert.equal(result.captures.generated.revision, 83);
  assert.deepEqual(result.captures.generated.result.assetIds, ['asset_4a13efad5e514debb937506901b123af']);
  // A handoff that never lands fails on its own bounded grace, naming the cause — not as a pass,
  // and not after the whole case deadline.
  let time = 0, cancels = 0;
  await assert.rejects(runScenario({ ...s, timeoutSeconds: 3600 }, { now: () => time, sleep: async ms => { time += ms; }, fetchImpl: async url => { if (url.pathname.endsWith('/cancel')) { cancels++; return response({}, 200); } return response(hybridCompletedUnpersisted); } }), /assetWrites were never persisted/);
  assert.ok(time >= ASSET_SETTLE_GRACE_MS && time < ASSET_SETTLE_GRACE_MS + 5000);
  assert.equal(cancels, 1);
});
test('a timed-out status read is retried before the deadline; a timed-out submission is not', async () => {
  // The CUDA campaign's status poll outlasted 30s while a checkpoint copy saturated the jobs
  // database's volume; the healthy case failed on that one read.
  const timeout = () => Object.assign(new Error('The operation was aborted due to timeout'), { name: 'TimeoutError' });
  const s = scenario(); s.steps = s.steps.slice(0, 2); let reads = 0;
  const result = await runScenario(s, { sleep: async () => {}, fetchImpl: async url => {
    if (url.pathname.endsWith('/import')) return response({ id: 'import-job' }, 201);
    if (++reads === 1) throw timeout();
    return response({ status: 'completed', result: { modelId: 'imported-model' } });
  } });
  assert.equal(reads, 2); assert.equal(result.captures.installed.status, 'completed');
  await assert.rejects(runScenario(s, { fetchImpl: async () => { throw timeout(); } }), /timeout/);
  let time = 0;
  await assert.rejects(runScenario(s, { now: () => time, sleep: async ms => { time += ms; }, fetchImpl: async url => {
    if (url.pathname.endsWith('/import')) return response({ id: 'import-job' }, 201);
    time += 6000; throw timeout();
  } }), /timeout/);
});
test('API assertions fail on dropped selected-model propagation and external services', async () => {
  const s = scenario(); s.baseUrl = 'https://unrelated.example'; await assert.rejects(runScenario(s));
  s.baseUrl = 'http://localhost:43210'; s.steps = [s.steps[0]];
  await assert.rejects(runScenario(s, { fetchImpl: async () => response({ id: 'wrong-job' }, 201) }));
  assert.throws(() => pointer({}, '/missing'), /missing/);
});
test('every long macOS campaign step holds a no-sleep assertion bound to its own shell', async () => {
  // Run 40903116430: the nax-macos MacBook Pro slept mid model-import; logging and every timer
  // froze for 5 h and the 7200 s case timeout never fired. Each long step keeps the machine awake
  // for exactly its own lifetime: `caffeinate -w $$` exits with the step's shell, and its detached
  // stdio holds no step pipe open, so no assertion can outlive the step (or the job).
  const workflow = await readFile(new URL('../.github/workflows/h3-finetune-terminal.yml', import.meta.url), 'utf8');
  const guard = 'if [ "${RUNNER_OS:-}" = "macOS" ]; then caffeinate -dimsu -w "$$" </dev/null >/dev/null 2>&1 & fi';
  const steps = workflow.split(/\n      - name: /).slice(1).map(block => ({ name: block.split('\n')[0], block }));
  for (const name of ['Bootstrap the reduced terminal campaign when requested', 'Ensure selected assets and conditioning in this terminal route', 'Derive explicit dense source variants from the actually staged raw snapshot', 'Extract the pinned public row as mono 32-kHz speech and retain attribution', 'Execute or reverify the reduced operational cases', 'Choose the measured matrix backend and finish this lane']) {
    const step = steps.find(s => s.name === name); assert.ok(step, name);
    assert.ok(step.block.includes(guard), `${name} lacks the step-bound caffeinate guard`);
    assert.ok(step.block.indexOf(guard) < step.block.indexOf("<<'NODE'") || !step.block.includes("<<'NODE'"), `${name}: the guard must precede the long-running work`);
  }
  // Never an unbound assertion that could leak past the step.
  for (const line of workflow.split('\n').filter(l => l.includes('caffeinate ') && !l.trim().startsWith('#'))) assert.equal(line.trim(), guard);
});
test('both terminal lanes select a reviewed CPython 3.12 instead of installing one', async () => {
  const workflow = await readFile(new URL('../.github/workflows/h3-finetune-terminal.yml', import.meta.url), 'utf8');
  // setup-python cannot install on either self-hosted platform here. On macOS it writes to the
  // HOSTED runner's tool cache, "/Users/runner", which does not exist on a self-hosted Mac whose
  // home is the operator's, so the job dies before any model work. On Windows the same action
  // removes the existing install and its HKLM registry entries, and the runner account cannot do
  // the registry part ("Requested registry access is not allowed") — and it deletes the
  // tool-cache interpreter on the way out. Both lanes therefore select an interpreter ALREADY on
  // the runner, through each platform's own verified selector in this repository.
  assert.doesNotMatch(workflow, /actions\/setup-python@/);
  assert.match(workflow, /id: speech_python_windows/);
  // Windows provisions the organization's hash-pinned portable CPython 3.12.10 (no installer,
  // no registry writes) and verifies it with the same selector the StarVector lane uses.
  assert.match(workflow, /Install-StarVectorWindowsPythonPackage -DestinationRoot \$pythonRoot/);
  assert.match(workflow, /Invoke-StarVectorWindowsPythonLock -LockPath/);
  assert.match(workflow, /Select-StarVectorBootstrapPython -CandidatePaths @\(\$installed\) -ExpectedVersion @\(3, 12, 10\)/);
  assert.match(workflow, /id: speech_python_macos/);
  assert.match(workflow, /select-starvector-macos-python\.mjs select \/opt\/homebrew\/bin\/python3\.12/);
  assert.match(workflow, /SPEECH_BASE_PYTHON: \$\{\{ steps\.speech_python_windows\.outputs\.python-path \|\| steps\.speech_python_macos\.outputs\.python-path \}\}/);
});
test('the Windows terminal lane selects the exact listener the request names', async () => {
  const workflow = await readFile(new URL('../.github/workflows/h3-finetune-terminal.yml', import.meta.url), 'utf8');
  const job = workflowJob(workflow, 'terminal');
  // Two approved full-weights CUDA listeners both match the generic label set, so the job could
  // land on the one the request did NOT name — and the exact-runner assertion then refuses. Each
  // listener carries its own name as a label, so the requested runner selects itself.
  const runsOn = /^    runs-on: (.+)$/m.exec(job)[1];
  assert.deepEqual(workflowExpression(runsOn, { backend: 'cuda', expected_runner: 'cuda-windows-2' }), [
    'self-hosted', 'windows', 'cuda', 'real-weights', 'cuda-windows-2',
  ]);
  assert.deepEqual(workflowExpression(runsOn, { backend: 'cuda', expected_runner: 'cuda-windows' }), [
    'self-hosted', 'windows', 'cuda', 'real-weights', 'cuda-windows',
  ]);
  // The macOS lane needs only the label exactly one macOS listener carries.
  assert.deepEqual(workflowExpression(runsOn, { backend: 'mlx', expected_runner: 'nax-macos' }), [
    'self-hosted', 'macOS', 'ARM64', 'real-weights',
  ]);
});

test('the bootstrap output root shares a volume with the staged model files', async () => {
  const workflow = await readFile(new URL('../.github/workflows/h3-finetune-terminal.yml', import.meta.url), 'utf8');
  // The runtime HARDLINKS staged model files into each case's data directory, and a hardlink
  // cannot cross volumes. With the output root in RUNNER_TEMP and the staged root on the weights
  // volume, the first real case failed with EXDEV. The output root is therefore derived from the
  // same per-run private staging root, as a sibling (outside the stage's own run-root cleanup).
  assert.match(workflow, /const stagedRoot = privateRunRoot\(/);
  assert.match(workflow, /const outputRoot = `\$\{stagedRoot\}-bootstrap-\$\{env\.GITHUB_SHA\}`;/);
  assert.doesNotMatch(workflow, /const outputRoot = path\.join\(env\.RUNNER_TEMP/);
  // Same device, by construction, on both platforms.
  const { privateRunRoot } = await import('./h3-finetune-terminal-stage.mjs');
  const mac = privateRunRoot({ backend: 'mlx', platform: 'darwin', env: { HF_HOME: '/Volumes/Models/huggingface' }, runId: '7', attempt: '1' });
  assert.equal(mac, path.join('/Volumes/Models/huggingface', 'sc26691-private', '7-1'));
  const win = privateRunRoot({ backend: 'cuda', platform: 'win32', env: {}, runId: '7', attempt: '1' });
  assert.equal(win, 'E:\\huggingface\\sc26691-private\\7-1');
});

test('the Windows terminal lane activates the MSVC host compiler for nvcc before any build', async () => {
  const workflow = await readFile(new URL('../.github/workflows/h3-finetune-terminal.yml', import.meta.url), 'utf8');
  const job = workflowJob(workflow, 'terminal');
  // A fresh step has no MSVC environment, so the native build failed with
  // "nvcc fatal: Cannot find compiler 'cl.exe' in PATH". The lane loads the same BuildTools
  // MSVC 14.44 the Windows Candle lanes use and exports it to the rest of the job.
  const activate = job.indexOf('Activate the MSVC host compiler for nvcc');
  assert.ok(activate >= 0, 'the Windows lane must activate MSVC');
  assert.ok(activate < job.indexOf('Execute or reverify the reduced operational cases'), 'before any native build');
  assert.ok(job.includes('2022\\BuildTools\\VC\\Auxiliary\\Build\\vcvars64.bat'));
  assert.ok(job.includes('NVCC_CCBIN=%VCToolsInstallDir%bin\\Hostx64\\x64'));
  assert.ok(job.includes('if not exist "%NVCC_CCBIN%\\cl.exe" exit /b 1'), 'a missing host compiler must refuse loudly');
  assert.ok(job.includes('$env:GITHUB_ENV'), 'the MSVC environment must reach the later bash/node build steps');
  // Git Bash's coreutils link.exe shadows MSVC's inside bash steps, so the linker is named exactly.
  assert.ok(job.includes('CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER=%NVCC_CCBIN%\\link.exe'));
  assert.ok(job.includes("'NVCC_CCBIN','CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER')"), 'the exact linker must be exported to the later build steps');
});

test('the Windows terminal lane selects Git Bash before its first bash step', async () => {
  const workflow = await readFile(new URL('../.github/workflows/h3-finetune-terminal.yml', import.meta.url), 'utf8');
  const job = workflowJob(workflow, 'terminal');
  // A self-hosted Windows listener's own bash resolution falls through to the WSL relay, which has
  // no /bin/bash there, so the lane died in its very first step. Git for Windows' bash IS on the
  // box and has to be put on the PATH before any bash step runs — the same selection the
  // inference Windows native-terminal lane makes.
  const selectIndex = job.indexOf('Select Git Bash');
  const firstBashIndex = job.indexOf('shell: bash');
  assert.ok(selectIndex >= 0, 'the Windows lane must select Git Bash');
  assert.ok(firstBashIndex >= 0, 'the lane runs bash steps');
  assert.ok(selectIndex < firstBashIndex, 'Git Bash must be selected before the first bash step');
  assert.ok(job.includes('if not exist "C:\\Program Files\\Git\\bin\\bash.exe" exit /b 1'), 'a missing Git Bash must be a loud refusal');
  assert.ok(job.includes('echo C:\\Program Files\\Git\\bin>>"%GITHUB_PATH%"'), 'Git Bash must be put on the PATH');
  assert.ok(job.includes("inputs.backend == 'cuda'"), 'the selection is the Windows lane only');
});

test('the macOS terminal lane reads and writes its staged set on the reviewed-weights volume', async () => {
  const workflow = await readFile(new URL('../.github/workflows/h3-finetune-terminal.yml', import.meta.url), 'utf8');
  // The runner service carries no HF_HOME, so the staging probe would otherwise read a
  // different hub than the one the reviewed weights are seeded into, and the lane's private
  // copy of the staged set would land on the runner's smaller internal disk instead of the
  // volume that holds it.
  assert.match(workflow, /HF_HOME: \$\{\{ inputs\.backend == 'mlx' && '\/Volumes\/Models\/huggingface' \|\| '' \}\}/);
  // The macOS lane must still be the one that probes that hub: the Windows lane's paths are
  // absolute on E: and must not be redirected by the macOS volume.
  assert.match(workflow, /\["self-hosted","macOS","ARM64","real-weights"\]/);
});

test('manual workflow is execution-only and uses existing runner/token infrastructure without global locks', async () => {
  const workflow = await readFile(new URL('../.github/workflows/h3-finetune-terminal.yml', import.meta.url), 'utf8');
  assert.match(workflow, /workflow_call:/); assert.doesNotMatch(workflow,/^  workflow_dispatch:/m,'only the existing registered terminal routes may call asset ensure'); assert.doesNotMatch(workflow, /^  (push|pull_request|schedule):/m);
  assert.match(workflow, /\[\"self-hosted\",\"macOS\",\"ARM64\",\"real-weights\"\]/); assert.match(workflow, /real-weights/); assert.match(workflow, /cancel-in-progress: false/);
  assert.match(workflow, /SCENEWORKS_INFERENCE_READ_TOKEN/); assert.match(workflow, /stationary_pair_authorized/);
  assert.doesNotMatch(workflow, /registration_only|pkill|flock|gpu.token/i);
});
const workflowJob = (text, name) => {
  const start=text.indexOf(`\n  ${name}:\n`); assert.ok(start>=0,`missing ${name} job`);
  const next=text.slice(start+1).search(/\n  [a-z][\w-]*:\n/); return text.slice(start,next<0?undefined:start+1+next);
};
const workflowExpression = (expression, inputs, github={}) => runInNewContext(expression.trim().replace(/^\$\{\{\s*|\s*\}\}$/g,''),{
  inputs,github:{event_name:'workflow_dispatch',run_id:'unit_fake_101',run_attempt:'1',ref:'refs/heads/unit_fake',...github},fromJSON:JSON.parse,toJSON:JSON.stringify,format:(template,...values)=>template.replace(/\{(\d+)\}/g,(_,n)=>String(values[Number(n)])),
},{timeout:100});
test('registered Candle entrypoint positively routes only explicit H3 selections to the same-ref reusable job',async()=>{
  const router=await readFile(new URL('../.github/workflows/server-candle-linux.yml',import.meta.url),'utf8');
  const callee=await readFile(new URL('../.github/workflows/h3-finetune-terminal.yml',import.meta.url),'utf8');
  const job=workflowJob(router,'h3-terminal'),selected=/^    if: (.+)$/m.exec(job)[1],authorization=/^      stationary_pair_authorized: (.+)$/m.exec(job)[1];
  assert.match(job,/uses: \.\/.github\/workflows\/h3-finetune-terminal\.yml/);assert.doesNotMatch(job,/uses:.*@main|registration_only|activation/);assert.match(callee,/workflow_call:/);
  for(const operation of ['standard','','source','provision','readiness','campaign','diagnostic-candle-1b','h3-import-mlx','h3-import-cuda','h3-import-both','h3-admission-proof','h3-ui-proof','h3-stage-mlx','h3-stage-cuda','h3-terminal','unknown'])assert.equal(workflowExpression(selected,{starvector_terminal_operation:operation}),false,operation);
  const choices=/starvector_terminal_operation:[\s\S]*?options: \[([^\]]+)\]/.exec(router)[1].split(',').map(v=>v.trim());
  for(const operation of ['h3-terminal-mlx','h3-terminal-cuda']){
    assert.equal(choices.filter(v=>v===operation).length,1);
    const inputs={starvector_terminal_operation:operation,campaign_run_id:JSON.stringify({stationary_pair_authorized:true})};assert.equal(workflowExpression(selected,inputs),true);assert.equal(workflowExpression(authorization,inputs),true);
    for(const event of ['push','pull_request','merge_group','schedule'])assert.equal(workflowExpression(selected,inputs,{event_name:event}),false);
    const backend=/^      backend: (.+)$/m.exec(job)[1];assert.equal(workflowExpression(backend,inputs),operation.endsWith('-mlx')?'mlx':'cuda');
    for(const value of [false,null,1,'true','false'])assert.equal(workflowExpression(authorization,{...inputs,campaign_run_id:JSON.stringify({stationary_pair_authorized:value})}),false);
  }
  const native=workflowJob(callee,'terminal'),nativeIf=/^    if: (.+)$/m.exec(native)[1];
  for(const event of ['push','pull_request','merge_group','schedule'])assert.equal(workflowExpression(nativeIf,{stationary_pair_authorized:true,backend:'mlx'},{event_name:event}),false);
  assert.equal(workflowExpression(nativeIf,{stationary_pair_authorized:false,backend:'mlx'}),false);assert.equal(workflowExpression(nativeIf,{stationary_pair_authorized:true,backend:'unknown'}),false);
  for(const backend of ['mlx','cuda'])assert.equal(workflowExpression(nativeIf,{stationary_pair_authorized:true,backend}),true);
});
test('H3 bridge preserves all standing dispatch keys, old flow selection and isolated concurrency groups',async()=>{
  const router=await readFile(new URL('../.github/workflows/server-candle-linux.yml',import.meta.url),'utf8');
  const header=router.slice(router.indexOf('  workflow_dispatch:'),router.indexOf('\npermissions:'));
  const keys=[...header.matchAll(/^      ([a-z][\w_]+):$/gm)].map(m=>m[1]).sort();
  assert.deepEqual(keys,['starvector_terminal_operation','inference_revision','inference_preflight_run_id','inference_preflight_artifact','mac_app_data_source','mac_hf_home_source','windows_app_data_source','windows_hf_home_source','permanent_pin','campaign_run_id'].sort());
  const rootGroup=/^  group: (.+)$/m.exec(router)[1],cancel=/^  cancel-in-progress: (.+)$/m.exec(router)[1],build=/^    if: (.+)$/m.exec(workflowJob(router,'build-candle'))[1];
  for(const operation of ['standard','']){assert.equal(workflowExpression(build,{starvector_terminal_operation:operation}),true);assert.equal(workflowExpression(cancel,{starvector_terminal_operation:operation}),true);assert.match(workflowExpression(rootGroup,{starvector_terminal_operation:operation}),/^server-candle-linux-/);}
  for(const operation of ['source','provision','readiness','campaign','diagnostic-candle-1b']){assert.equal(workflowExpression(rootGroup,{starvector_terminal_operation:operation}),'starvector-terminal-bridge');assert.equal(workflowExpression(cancel,{starvector_terminal_operation:operation}),false);assert.equal(workflowExpression(build,{starvector_terminal_operation:operation}),false);}
  for(const operation of ['h3-terminal-mlx','h3-terminal-cuda']){const inputs={starvector_terminal_operation:operation};assert.equal(workflowExpression(cancel,inputs),false);assert.equal(workflowExpression(build,inputs),false);assert.notEqual(workflowExpression(rootGroup,inputs),workflowExpression(rootGroup,inputs,{run_id:'unit_fake_102',run_attempt:'1'}));assert.match(workflowExpression(rootGroup,inputs),/^h3-terminal-bridge-/);}
  for(const operation of ['h3-import-mlx','h3-import-cuda','h3-import-both']){const inputs={starvector_terminal_operation:operation};assert.equal(workflowExpression(cancel,inputs),false);assert.equal(workflowExpression(build,inputs),false);assert.equal(workflowExpression(rootGroup,inputs),'h3-import-native-unit_fake_101');assert.equal(workflowExpression(rootGroup,inputs,{run_id:'unit_fake_102'}),'h3-import-native-unit_fake_102');}
  const cpuInputs={starvector_terminal_operation:'h3-admission-proof'};assert.equal(workflowExpression(rootGroup,cpuInputs),'h3-admission-cpu-unit_fake_101');assert.equal(workflowExpression(rootGroup,cpuInputs,{run_id:'unit_fake_102'}),'h3-admission-cpu-unit_fake_102');assert.equal(workflowExpression(cancel,cpuInputs),false);assert.equal(workflowExpression(build,cpuInputs),false);
  for(const operation of ['h3-import-mlx','h3-import-cuda','h3-import-both','h3-admission-proof','h3-ui-proof','h3-terminal-mlx','h3-terminal-cuda'])for(const event of ['push','pull_request','merge_group','schedule'])assert.equal(workflowExpression(rootGroup,{starvector_terminal_operation:operation},{event_name:event}),'starvector-terminal-bridge');
  for(const operation of ['unknown','h3-import','h3-terminal']){const inputs={starvector_terminal_operation:operation};assert.equal(workflowExpression(build,inputs),false);assert.equal(workflowExpression(rootGroup,inputs),'starvector-terminal-bridge');}
  for(const [name,file] of [['source-closure','starvector-terminal-source.yml'],['provision','starvector-terminal-provision.yml'],['readiness','starvector-terminal-readiness.yml'],['campaign','starvector-terminal.yml']])assert.ok(workflowJob(router,`starvector-${name}`).includes(`uses: ./.github/workflows/${file}`));
});
test('H3 caller forwards only explicit request fields and the standing read token without duplicate execution logic',async()=>{
  const router=await readFile(new URL('../.github/workflows/server-candle-linux.yml',import.meta.url),'utf8'),job=workflowJob(router,'h3-terminal');
  for(const field of ['sceneworks_sha','inference_sha','expected_runner','plan_path','output_path','case_id'])assert.ok(new RegExp(field+': \\$\\{\\{ fromJSON\\(inputs\\.campaign_run_id\\)\\.'+field+'( \\|\\| \'\')? \\}\\}').test(job),field);
  assert.doesNotMatch(job,/runs-on:|run-case|cargo |permanent_pin|inference_revision|secrets: inherit/);
  assert.match(job,/SCENEWORKS_INFERENCE_READ_TOKEN: \$\{\{ secrets\.SCENEWORKS_INFERENCE_READ_TOKEN \|\| secrets\.INFERENCE_READ_TOKEN \}\}/);
  const callee=await readFile(new URL('../.github/workflows/h3-finetune-terminal.yml',import.meta.url),'utf8');assert.match(callee,/SCENEWORKS_INFERENCE_READ_TOKEN:\n        required: false/);assert.match(callee,/group: h3-finetune-terminal-\$\{\{ github.run_id \}\}-\$\{\{ github.run_attempt \}\}/);
});
test('terminal reuses native device queues and binds manual CUDA physical GPU 1 without changing ordinary queues',async()=>{
  const workflow=await readFile(new URL('../.github/workflows/h3-finetune-terminal.yml',import.meta.url),'utf8'),job=workflowJob(workflow,'terminal');
  const group=/^      group: (.+)$/m.exec(job)[1];assert.equal(workflowExpression(group,{backend:'cuda'}),'windows-candle-gpu-real-weights');assert.equal(workflowExpression(group,{backend:'mlx'}),'h3-import-proofs-nax-macos');assert.match(job,/cancel-in-progress: false/);
  for(const [key,value]of [['CUDA_DEVICE_ORDER','PCI_BUS_ID'],['CUDA_VISIBLE_DEVICES','1'],['SCENEWORKS_GPU_ID','1']]){const expression=new RegExp(`^          ${key}: (.+)$`,'m').exec(job)[1];assert.equal(workflowExpression(expression,{backend:'cuda'}),value);}
});
test('actual reusable preflight rejects source-ref drift before product execution without a main activation prerequisite',async t=>{
  const workflow=await readFile(new URL('../.github/workflows/h3-finetune-terminal.yml',import.meta.url),'utf8'),code=/node --input-type=module -e '\n([\s\S]*?)\n          '/.exec(workflow)?.[1];assert.ok(code);
  assert.doesNotMatch(code,/workflow.*register|default.*branch|gh workflow|activation/);
  const dir=await mkdtemp(path.join(os.tmpdir(),'h3-dispatch-unit-fake-'));t.after(()=>rm(dir,{recursive:true,force:true}));const file=path.join(dir,'plan.json');await writeFile(file,JSON.stringify({sceneworks:sw,inference:inf,output:dir,cases:[{id:'mlx.artifact.0',backend:'mlx'}]}));
  const env={...process.env,PLAN:file,OUTPUT:dir,CASE_ID:'mlx.artifact.0',BACKEND:'mlx',EXPECTED_RUNNER:'unit_fake_runner',RUNNER_NAME:'unit_fake_runner',SW_SHA:sw,INF_SHA:inf,GITHUB_SHA:sw};
  const good=spawnSync(process.execPath,['--input-type=module','-e',code],{env,encoding:'utf8'});assert.equal(good.status,0,good.stderr);
  const bad=spawnSync(process.execPath,['--input-type=module','-e',code],{env:{...env,GITHUB_SHA:inf},encoding:'utf8'});assert.equal(bad.status,1);assert.match(bad.stderr,/caller workflow commit/);
});
test('terminal compiler selection derives only the checked-out root pin without executing Rust locally',async t=>{
  const workflow=await readFile(new URL('../.github/workflows/h3-finetune-terminal.yml',import.meta.url),'utf8');
  const reader=workflow.slice(workflow.indexOf('      - name: Read the checked-out root Rust pin'),workflow.indexOf('      - name: Prepare the root-pinned compiler'));
  assert.match(reader,/working-directory: sceneworks/);
  const code=/node --input-type=module -e '\n([\s\S]*?)\n          '/.exec(reader)?.[1];assert.ok(code);
  const dir=await mkdtemp(path.join(os.tmpdir(),'h3-toolchain-unit-fake-'));t.after(()=>rm(dir,{recursive:true,force:true}));
  const rootPin=/^channel = "([^"]+)"$/m.exec(await readFile(new URL('../rust-toolchain.toml',import.meta.url),'utf8'))[1];
  const output=path.join(dir,'environment'),paths=path.join(dir,'paths'),cargoHome=path.join(dir,'cargo');
  const run=()=>spawnSync(process.execPath,['--input-type=module','-e',code],{cwd:dir,env:{...process.env,GITHUB_ENV:output,GITHUB_PATH:paths,CARGO_HOME:cargoHome},encoding:'utf8'});
  for(const pin of [rootPin,'9.8.7']){
    await writeFile(path.join(dir,'rust-toolchain.toml'),`[toolchain]\nchannel = "${pin}"\n`);await writeFile(output,'');await writeFile(paths,'');
    const result=run();assert.equal(result.status,0,result.stderr);assert.equal(await readFile(output,'utf8'),`RUSTUP_TOOLCHAIN=${pin}\n`);assert.equal(await readFile(paths,'utf8'),path.join(cargoHome,'bin')+'\n');
  }
  for(const source of ['[toolchain]\nchannel = "stable"\n','[toolchain]\nchannel = "1.98"\n','[toolchain]\n','channel = "9.8.7"\nchannel = "9.8.6"\n']){
    await writeFile(path.join(dir,'rust-toolchain.toml'),source);await writeFile(output,'');assert.notEqual(run().status,0);assert.equal(await readFile(output,'utf8'),'');
  }
  const prepare=workflow.slice(workflow.indexOf('      - name: Prepare the root-pinned compiler'),workflow.indexOf('      - name: Execute or reverify'));
  assert.match(prepare,/rustup toolchain install "\$RUSTUP_TOOLCHAIN" --profile minimal --no-self-update/);assert.match(prepare,/rustup run "\$RUSTUP_TOOLCHAIN" rustc --version/);assert.doesNotMatch(prepare,/rustup (?:default|override)|1\.97\.1/);
});
test('source UI proof is explicit hosted Chromium only, independent of model pin and campaign authorization',async()=>{
  const router=await readFile(new URL('../.github/workflows/server-candle-linux.yml',import.meta.url),'utf8'),job=workflowJob(router,'h3-ui-proof');
  const {PLAYWRIGHT_VERSION}=await import('../apps/web/verify/hosted-browser/guard.mjs');assert.ok(job.includes(`playwright@${PLAYWRIGHT_VERSION}`));assert.match(job,/SC26687_EXPECTED_HEAD: \$\{\{ github.sha \}\}/);
  const choices=/starvector_terminal_operation:[\s\S]*?options: \[([^\]]+)\]/.exec(router)[1].split(',').map(v=>v.trim());assert.equal(choices.filter(v=>v==='h3-ui-proof').length,1);
  const selected=/^    if: (.+)$/m.exec(job)[1];
  for(const operation of [undefined,'','unknown',...choices])for(const event of ['push','pull_request','merge_group','schedule','workflow_dispatch'])assert.equal(workflowExpression(selected,{starvector_terminal_operation:operation},{event_name:event}),event==='workflow_dispatch'&&operation==='h3-ui-proof');
  for(const name of [...router.slice(router.indexOf('\njobs:')).matchAll(/^  ([a-z][\w-]*):$/gm)].map(m=>m[1]).filter(v=>v!=='h3-ui-proof'))assert.equal(workflowExpression(/^    if: (.+)$/m.exec(workflowJob(router,name))[1],{starvector_terminal_operation:'h3-ui-proof'}),false,`${name} refuses UI proof`);
  assert.match(job,/runs-on: ubuntu-latest/);assert.match(job,/timeout-minutes: 20/);assert.match(job,/ref: \$\{\{ github.sha \}\}/);assert.match(job,/fetch-depth: 0/);assert.match(job,/persist-credentials: false/);
  for(const text of ['test "$RUNNER_ENVIRONMENT" = github-hosted','test "$GITHUB_ACTIONS" = true','SC26687_HOSTED_BROWSER_PROOF: "1"','npm ci --prefix apps/web','playwright@1.59.1','SC26687_PLAYWRIGHT_ROOT','sc26687-playwright-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}','PLAYWRIGHT_BROWSERS_PATH','resolvePlaywrightEntry','"$SC26687_PLAYWRIGHT_ROOT/node_modules/.bin/playwright" install --with-deps chromium','playwright-resolution.log','node apps/web/verify/hosted-browser/run-hosted-browser.mjs','set -euo pipefail','source-inputs.sha256','if-no-files-found: error'])assert.ok(job.includes(text),text);
  assert.doesNotMatch(job,/--prefix apps\/web playwright|npm install --no-save --package-lock=false --prefix apps\/web|npx --no-install playwright/,'Playwright must not mutate the apps/web npm graph');
  const driver=await readFile(new URL('../apps/web/verify/hosted-browser/run-hosted-browser.mjs',import.meta.url),'utf8'),guard=await readFile(new URL('../apps/web/verify/hosted-browser/guard.mjs',import.meta.url),'utf8');
  assert.match(driver,/resolvePlaywrightEntry\(\{ root: process\.env\.SC26687_PLAYWRIGHT_ROOT \}\)/);assert.match(driver,/loadPlaywrightCommonJs\(\{ entry: playwrightTools\.entry \}\)/);assert.doesNotMatch(driver,/await import\("playwright"\)|pathToFileURL\(playwrightTools\.entry\)|require\.resolve\("playwright\/package\.json"\)|NODE_PATH/);
  assert.match(guard,/export function resolvePlaywrightEntry/);assert.match(guard,/export function resolveBrowsersPath/);assert.match(guard,/export function loadPlaywrightCommonJs/);assert.doesNotMatch(guard,/NODE_PATH/);
  assert.doesNotMatch(job,/playwright@[~^]|permanent_pin|stationary_pair_authorized|run-case|cargo |self-hosted|CUDA_VISIBLE_DEVICES|HF_HOME|secrets: inherit/);
  assert.match(job,/if: \$\{\{ always\(\) \}\}/);assert.match(job,/tee "\$SC26687_ARTIFACT_DIR\/browser-proof.log"/);
  const rootGroup=/^  group: (.+)$/m.exec(router)[1],cancel=/^  cancel-in-progress: (.+)$/m.exec(router)[1],inputs={starvector_terminal_operation:'h3-ui-proof'};
  assert.equal(workflowExpression(rootGroup,inputs),'h3-ui-hosted-unit_fake_101-1');assert.equal(workflowExpression(rootGroup,inputs,{run_id:'unit_fake_102'}),'h3-ui-hosted-unit_fake_102-1');assert.equal(workflowExpression(rootGroup,inputs,{run_attempt:'2'}),'h3-ui-hosted-unit_fake_101-2');assert.equal(workflowExpression(cancel,inputs),false);
});
test('CPU bridge provisions its pinned private interpreter before model assets and records the observed closure',async()=>{
  const source=await readFile(new URL('../.github/workflows/h3-finetune-terminal.yml',import.meta.url),'utf8'),start=source.indexOf('      - name: Prepare private CPU-only conditioning and source-variant tools'),assets=source.indexOf('      - name: Ensure selected assets and conditioning in this terminal route');
  assert.ok(start>0&&start<assets,'import/help failure must precede massive model downloads');const setup=source.slice(start,assets);
  for(const token of ['torch==2.9.1','https://download.pytorch.org/whl/cpu','H3_PROBE_PYTHON','cpu-tools-freeze.txt','cpu-tools.json','cpu-bridge-help.txt','--help','sys.version_info[:2]==(3,12)','"-I","-B"'])assert.ok(setup.includes(token),token);
  // The bridge's interpreter is a PINNED CPython 3.12 SELECTED from the runner (each platform's
  // own verified selector), not one an action installs: neither platform can install one here.
  assert.ok(source.includes('Install-StarVectorWindowsPythonPackage'));
  assert.ok(source.includes('ExpectedVersion @(3, 12, 10)'));
  assert.ok(source.includes('select-starvector-macos-python.mjs select'));
  assert.doesNotMatch(source,/actions\/setup-python@/);assert.doesNotMatch(setup,/--no-deps|PYTHONPATH|einops|diffusers/);assert.ok(setup.includes('numericProof:false'));
  const adapter=await readFile(new URL('./h3-finetune-terminal-artifact-probe.mjs',import.meta.url),'utf8');assert.ok(adapter.includes("process.env.H3_PROBE_PYTHON"));assert.ok(adapter.includes("const producerCommand=[python,'-I','-B'"));
});
test('asset staging is folded into the terminal delivery and runs the reduced 25/backend inventory',async()=>{
  const router=await readFile(new URL('../.github/workflows/server-candle-linux.yml',import.meta.url),'utf8'),callee=await readFile(new URL('../.github/workflows/h3-finetune-terminal.yml',import.meta.url),'utf8');
  const choices=/starvector_terminal_operation:[\s\S]*?options: \[([^\]]+)\]/.exec(router)[1].split(',').map(v=>v.trim());
  assert.deepEqual(choices,['standard','source','provision','readiness','campaign','diagnostic-candle-1b','h3-import-mlx','h3-import-cuda','h3-import-both','h3-admission-proof','h3-ui-proof','h3-terminal-mlx','h3-terminal-cuda']);
  for(const value of ['h3-stage-mlx','h3-stage-cuda'])assert.equal(choices.filter(v=>v===value).length,0,`${value} selector removed`);
  assert.doesNotMatch(router,/\n  h3-stage:\n/);assert.doesNotMatch(router,/h3-assets-stage-/);
  const caller=workflowJob(router,'h3-terminal');assert.ok(caller.includes('campaign_request: ${{ inputs.campaign_run_id }}'));
  const terminal=workflowJob(callee,'terminal');
  const ensure=callee.indexOf('      - name: Ensure selected assets and conditioning in this terminal route');
  const resolve=callee.indexOf('      - name: Resolve the reduced operational case set for this backend');
  const validate=callee.indexOf('      - name: Validate explicit inputs before any product execution');
  const run=callee.indexOf('      - name: Execute or reverify the reduced operational cases');
  assert.ok(ensure>0&&resolve>ensure&&validate>resolve&&run>validate,'ensure -> reduced-case resolve -> validate -> run ordering');
  const ensureSlice=callee.slice(ensure,resolve);
  for(const token of ['runStage','h3-finetune-terminal-stage.mjs','prepareReferenceFixtures','receipt.identityControls','expected-assets-not-integrity-proof.json','stage-binding.json','bound-plan.json','blocked_by_allowed_roots','H3_PLAN_EFFECTIVE','license_authority','set -euo pipefail'])assert.ok(ensureSlice.includes(token),token);
  assert.doesNotMatch(ensureSlice,/run-case|acknowledged:\s*true|licenseAcknowledged:\s*true/,'staging must not synthesize license consent or execute cases');
  for(const token of ['pyarrow==19.0.1','h3-finetune-terminal-speech.py','h3-finetune-terminal-float-variants.py'])assert.ok(callee.includes(token),token);
  const resolveSlice=callee.slice(resolve,validate);
  for(const token of ['h3-finetune-terminal.mjs','executionCases','validatePlan','case-ids.json','25','18'])assert.ok(resolveSlice.includes(token),token);
  const runSlice=callee.slice(run);
  for(const token of ['H3_TERMINAL_CASES','run-case','H3_TERMINAL_SKIP_RUN','H3_PLAN_EFFECTIVE'] )assert.ok(runSlice.includes(token),token);
  assert.doesNotMatch(runSlice,/CASE_ID" "\$GITHUB_WORKSPACE\/sceneworks" "\$GITHUB_WORKSPACE\/inference"\n/, 'single-case invocation replaced by the reduced loop');
  assert.match(callee,/if: (?:\$\{\{ )?always\(\)/);
  assert.match(callee,/h3-terminal-stage-\$\{\{ inputs\.backend \}\}/);
  assert.match(callee,/if-no-files-found: warn/);
});
test('bootstrap request branches before the first plan read and skips restaging when prepared',async()=>{
  const router=await readFile(new URL('../.github/workflows/server-candle-linux.yml',import.meta.url),'utf8'),callee=await readFile(new URL('../.github/workflows/h3-finetune-terminal.yml',import.meta.url),'utf8');
  const header=callee.slice(callee.indexOf('  workflow_call:'),callee.indexOf('permissions:'));
  for(const field of ['plan_path','output_path'])assert.match(header,new RegExp(field+':\\n\\s+required: false\\n\\s+type: string'),field);
  const cpu=callee.indexOf('      - name: Prepare private CPU-only conditioning and source-variant tools');
  const bootstrap=callee.indexOf('      - name: Bootstrap the reduced terminal campaign when requested');
  const ensure=callee.indexOf('      - name: Ensure selected assets and conditioning in this terminal route');
  assert.ok(cpu>0&&bootstrap>cpu&&ensure>bootstrap,'CPU Torch preflight -> bootstrap -> ensure ordering');
  const bootSlice=callee.slice(bootstrap,ensure);
  for(const token of ['h3-finetune-terminal-bootstrap.mjs','bootstrapTerminal','request.bootstrap','sc-26691-h3-cases-v1','reduced-v1','componentSelections','legalAuthority','H3_PLAN_EFFECTIVE','H3_OUTPUT_EFFECTIVE','H3_BOOTSTRAP_PREPARED','GITHUB_OUTPUT','privateRunRoot','sc26691-private','GITHUB_RUN_ATTEMPT','H3_SCENEWORKS_SHA','H3_INFERENCE_SHA','set -euo pipefail'])assert.ok(bootSlice.includes(token),token);
  assert.match(bootSlice,/prepared:\s*true|prepared,\s*true/);assert.match(bootSlice,/proof:\s*false|proof,\s*false/);
  assert.doesNotMatch(bootSlice,/h3-finetune-terminal\.mjs"?\s+"?run-case|run-case/, 'bootstrap only prepares; it does not execute cases');
  const ensureSlice=callee.slice(ensure,callee.indexOf('      - name: Derive explicit dense source variants'));
  assert.ok(ensureSlice.indexOf('H3_BOOTSTRAP_PREPARED')>=0&&ensureSlice.indexOf('H3_BOOTSTRAP_PREPARED')<ensureSlice.indexOf('readFile('),'bootstrap branch must precede the first plan read');
  for(const token of ['external plan_path is required when request.bootstrap is absent','stageRequested && !bootstrapPrepared','binding.status="bootstrap-prepared"','H3_OUTPUT_EFFECTIVE','H3_INFERENCE_SHA'])assert.ok(ensureSlice.includes(token),token);
  assert.doesNotMatch(ensureSlice,/prepareIdentityControls\(/,'bootstrap path must not redo identity controls');
  const choices=/starvector_terminal_operation:[\s\S]*?options: \[([^\]]+)\]/.exec(router)[1].split(',').map(v=>v.trim());
  assert.equal(choices.length,13);assert.doesNotMatch(router,/\n  h3-bootstrap:\n/);
  assert.match(callee,/timeout-minutes: 1440/);assert.match(callee,/budgetSeconds:86400/);
  assert.match(callee,/H3_OUTPUT_EFFECTIVE/);
});
test('ordinary check selects the CPU-only preparation tests rather than a terminal/model command',async()=>{
  const p=JSON.parse(await readFile(new URL('../package.json',import.meta.url),'utf8'));
  for(const name of ['h3-finetune-terminal.test.mjs','h3-finetune-terminal-cases.test.mjs','h3-finetune-terminal-runtime.test.mjs'])assert.ok(p.scripts.check.includes(`scripts/${name}`));
  assert.doesNotMatch(p.scripts.check,/h3-finetune-terminal(?:-runtime)?\.mjs\s+(?:run-case|api|numeric)|gh workflow run/);
});
test('delivery records reject old CI, wrong main pin and unresolved limitations', async t => {
  const { p } = await fixture(t);
  p.delivery = { requiredCI: [{ headSha: sw, conclusion: 'success', url: 'https://example.test/ci' }], featureEndReview: { scope: 'epic-requirements-and-production-delta', sceneworks: sw, inference: inf, unresolvedFindings: 0, url: 'https://example.test/review' }, inferenceMain: { sha: inf, url: 'https://example.test/inference-main' }, sceneworksMain: { sha: 'e'.repeat(40), sourceCommit: sw, inferencePin: inf, url: 'https://example.test/sw-main' }, shortcutReadbacks: [26679,26691].map(id => ({ id, observedState: 'In Progress', readAt: '2026-10-24T00:00:00Z', url: `https://app.shortcut.com/trefry/${id}` })), limitations: [] };
  validateDelivery(p);
  for (const mutate of [d => d.requiredCI[0].headSha = 'f'.repeat(40), d => d.sceneworksMain.inferencePin = sw, d => d.limitations.push('Missing CUDA AV')]) {
    const copy = structuredClone(p); mutate(copy.delivery); assert.throws(() => validateDelivery(copy));
  }
  const prior=structuredClone(p);
  prior.delivery.featureEndReview.sceneworks='c'.repeat(40);prior.delivery.featureEndReview.inference='d'.repeat(40);
  assert.throws(()=>validateDelivery(prior),/focused late-delta/);
  prior.delivery.lateDeltaReview={scope:'late-pin-generated-repair-delta',baseSceneworks:prior.delivery.featureEndReview.sceneworks,baseInference:prior.delivery.featureEndReview.inference,sceneworks:sw,inference:inf,unresolvedFindings:0,url:'https://example.test/late-review'};
  validateDelivery(prior);
  for(const mutate of [d=>d.featureEndReview.scope='pin-only',d=>d.featureEndReview.unresolvedFindings=1,d=>d.featureEndReview.sceneworks='not-a-sha',d=>d.lateDeltaReview.baseSceneworks=sw,d=>d.lateDeltaReview.baseInference=inf,d=>d.lateDeltaReview.sceneworks='e'.repeat(40),d=>d.lateDeltaReview.inference='f'.repeat(40),d=>d.lateDeltaReview.unresolvedFindings=1,d=>d.lateDeltaReview.scope='second-whole-branch-review',d=>delete d.lateDeltaReview.url]){
    const copy=structuredClone(prior);mutate(copy.delivery);assert.throws(()=>validateDelivery(copy));
  }
});
test('honest sampled CUDA process/device-used peaks do not require native allocator highwater', async t => {
  const f = await fixture(t);
  for (const method of ['sampled-process-cuda-used', 'sampled-device-used']) {
    f.e.metrics.observations.device = peakObservation(200, method, 'case');
    f.e.metrics.observations.loadDevice = peakObservation(180, method, 'loading');
    validatePeakMetrics(f.e.metrics, { ...f.c, backend: 'cuda' });
  }
});
test('MLX cumulative allocator highwater is accepted honestly as process-global, not phase-local', async t => {
  const f = await fixture(t);
  f.e.metrics.observations.device = peakObservation(200, 'mlx-allocator-cumulative-high-water', 'case');
  f.e.metrics.observations.loadDevice = peakObservation(180, 'mlx-allocator-cumulative-high-water', 'loading');
  validatePeakMetrics(f.e.metrics, f.c);
  f.e.metrics.observations.loadDevice.attribution.includesPriorAllocations = false;
  assert.throws(() => validatePeakMetrics(f.e.metrics, f.c), /not window-local/);
});
test('sampled peak receipts reject missing cadence/attribution/blindspots, invented maxima and acquisition-only windows', async t => {
  const f = await fixture(t);
  for (const mutate of [
    o => delete o.intervalMs, o => o.attribution = {}, o => o.blindspots = [], o => o.bounds = [],
    o => o.samples = [], o => o.samples[1].bytes = 201,
    o => o.window.coverage = 'source-acquisition-through-provider-handle',
    o => o.samples[1].at = '2026-10-24T00:00:01Z',
  ]) {
    const metrics = structuredClone(f.e.metrics); mutate(metrics.observations.loadDevice);
    assert.throws(() => validatePeakMetrics(metrics, f.c));
  }
});
test('newly prepared native derivatives are verified by bytes without invented pre-run hashes', async t => {
  const f = await fixture(t);
  const derivative = path.join(f.dir, 'native-derivative'); await writeFile(derivative, 'actual new derivative');
  f.e.loadEvent.prepared_sources = JSON.stringify([{ path: derivative, sha256: await hashFile(derivative) }]);
  await f.save(); await f.verify();
  await writeFile(derivative, 'drifted derivative'); await assert.rejects(f.verify(), /prepared source content changed/);
});
test('a declared input read in place is recognized in the product verbatim spelling', { skip: process.platform !== 'win32' }, async t => {
  // A linked checkpoint is read below its library root, which the API reports verbatim
  // (`\\?\E:\...`, campaign 40874652324 library-roots). It is a declared input outside task state.
  const f = await fixture(t);
  const outside = await mkdtemp(path.join(os.tmpdir(), 'h3-linked-')); t.after(() => rm(outside, { recursive: true, force: true }));
  const linked = path.join(outside, 'unrelated-name.safetensors'); await writeFile(linked, 'linked checkpoint bytes');
  const sha256 = await hashFile(linked); f.c.inputs.push({ role: 'checkpoint', path: linked, sha256 });
  f.e.loadEvent.prepared_sources = JSON.stringify([{ path: `\\\\?\\${linked}`, sha256 }]);
  await f.save(); await f.verify();
});
test('replaced executable reference output cannot pass numerical fidelity', async t => {
  const f = await fixture(t, 'numeric'); await f.verify();
  await writeFile(path.join(f.dir, 'reference.json'), '[2,4,6]');
  await writeFile(path.join(f.dir, 'candidate.json'), '[2,4,6]');
  await assert.rejects(f.verify(), /reference output was replaced/);
});
test('renamed stock weights cannot impersonate a frozen HF artifact with internally consistent receipts', async t => {
  const f = await fixture(t); await f.verify();
  const catalog = JSON.parse(await readFile(path.join(f.dir, 'source-catalog.json'), 'utf8'));
  catalog.siblings[0].lfs.sha256 = 'f'.repeat(64);
  await writeFile(path.join(f.dir, 'source-catalog.json'), JSON.stringify(catalog));
  await assert.rejects(f.verify(), /not the frozen source artifact/);
});
test('generation reclassification cannot bypass mandatory AV or metrics', async t => {
  const f = await fixture(t); f.c.group = 'lifecycle'; delete f.e.av; delete f.e.metrics; await f.save();
  assert.throws(() => validatePlan(f.p)); await assert.rejects(f.verify(), /reclassification/);
});
test('refusal lifecycle and arithmetic fixtures need executed input proof, not invented successful load receipts', async t => {
  for (const group of ['lifecycle', 'numeric']) {
    const f = await fixture(t, group);
    delete f.e.runtimeIdentity; delete f.e.completedJob; delete f.e.loadEvent; delete f.e.metrics;
    await f.save(); await f.verify();
    delete f.e.testInputSha256; await f.save(); await assert.rejects(f.verify(), /test input identity/);
  }
});
test('whole-lane budget persists its original deadline across cases and dispatches',async t=>{
  const {p}=await fixture(t);const {laneBudget}=await import('./h3-finetune-terminal.mjs');const start=Date.parse('2026-10-25T00:00:00Z');
  const first=await laneBudget(p,'mlx',{now:start});assert.equal(first.remainingSeconds,86400);
  const resumed=await laneBudget(p,'mlx',{now:start+3600000,startedAt:'2026-10-26T00:00:00Z'});assert.equal(resumed.remainingSeconds,82800);assert.equal(resumed.startedAt,first.startedAt);
  await assert.rejects(laneBudget(p,'mlx',{now:start+86400001}),/original 24-hour/);
});
test('inline source binding rewrites the actual scenario and input hash together without creating consent',async t=>{
  const {p,c,dir}=await fixture(t);p.proof_kind='unit_fake';
  const {mkdir}=await import('node:fs/promises'),{bindStagedCampaignPlan}=await import('./h3-finetune-terminal-cases.mjs');
  const root=path.join(dir,'sc26691-private','101-1');await mkdir(root,{recursive:true});const source=c.inputs[0],target=path.join(root,'model.bin');await writeFile(target,await readFile(source.path));
  const scenarioPath=path.join(dir,'original-scenario.json'),scenario={caseId:c.id,steps:[{body:{sourcePath:source.path,prompt:source.path,licenseAcknowledged:false}}]};await writeFile(scenarioPath,JSON.stringify(scenario));
  c.runtime={scenario:scenarioPath,port:43210,gpuId:'unit_fake'};c.inputs.push({path:scenarioPath,sha256:await hashFile(scenarioPath),role:'scenario'});c.declaredSourceBindings=[{licenseAuthority:{kind:'unit_fake',acknowledged:false,reference:'NOT real approval'}}];
  const receiptPath=path.join(root,'h3-stage-receipt.json');await writeFile(receiptPath,JSON.stringify({proof_kind:'unit_fake',status:'completed',files:[{status:'completed',finalPath:target,expectedBytes:11,actualBytes:11,expectedSha256:source.sha256,observedSha256:source.sha256}]}));
  const result=await bindStagedCampaignPlan(p,{receiptPath,root,backend:'mlx',allowUnitFake:true}),bound=result.plan.cases.find(v=>v.id===c.id),actual=JSON.parse(await readFile(bound.runtime.scenario,'utf8'));
  assert.equal(actual.steps[0].body.sourcePath,target);assert.equal(actual.steps[0].body.prompt,source.path,'a prompt equal to a filesystem path remains literal creative text');assert.equal(actual.steps[0].body.licenseAcknowledged,false);assert.deepEqual(bound.declaredSourceBindings,c.declaredSourceBindings);
  assert.equal(bound.inputs.find(v=>v.role==='scenario').sha256,await hashFile(bound.runtime.scenario));assert.deepEqual(JSON.parse(await readFile(scenarioPath,'utf8')),scenario);assert.equal(result.plan.proof_kind,'unit_fake');
});
test('three individual source mutations are killed by executed assertions, not syntax errors', async t => {
  const { dir } = await fixture(t);
  const original = await readFile(new URL('./h3-finetune-terminal.mjs', import.meta.url), 'utf8');
  for (const file of ['h3-finetune-terminal-api.mjs', 'h3-finetune-terminal-reduced.mjs', 'h3-finetune-terminal.test.mjs']) await writeFile(path.join(dir, file), await readFile(new URL(file, import.meta.url)));
  await mkdir(path.join(dir, 'lib'), { recursive: true });
  await writeFile(path.join(dir, 'lib', 'h3-campaign-build-profile.mjs'), await readFile(new URL('./lib/h3-campaign-build-profile.mjs', import.meta.url)));
  await writeFile(path.join(dir, 'lib', 'windows-verbatim-path.mjs'), await readFile(new URL('./lib/windows-verbatim-path.mjs', import.meta.url)));
  await writeFile(path.join(dir, 'lib', 'h3-peer-lane.mjs'), await readFile(new URL('./lib/h3-peer-lane.mjs', import.meta.url)));
  for (const [needle, pattern] of [
    ['assert.equal(e.inference, p.inference);', 'receipt mutation rejects wrong source revision'],
    ["assert.equal(+m[3], 0, 'ignored tests are not evidence');", 'test execution evidence rejects ignored'],
    ['assert.ok(Math.abs(v - r) <= tolerance.atol + tolerance.rtol * Math.abs(r), `numeric mismatch at ${i}`);', 'numeric comparison discriminates scaling'],
  ]) {
    assert.equal(original.split(needle).length, 2, 'mutation must reach exactly one guard');
    const mutated = path.join(dir, 'h3-finetune-terminal.mjs');
    await writeFile(mutated, original.replace(needle, '/* source mutation: removed guard */'));
    assert.equal(spawnSync(process.execPath, ['--check', mutated]).status, 0, 'non-compiling mutation is invalid');
    const childEnv = { ...process.env }; delete childEnv.NODE_TEST_CONTEXT;
    const run = spawnSync(process.execPath, ['--test', `--test-name-pattern=${pattern}`, path.join(dir, 'h3-finetune-terminal.test.mjs')], { encoding: 'utf8', env: childEnv });
    assert.equal(run.status, 1, `mutation survived: ${pattern}\n${run.stdout}\n${run.stderr}`);
    assert.match(run.stdout + run.stderr, /Missing expected (exception|rejection)/, 'must fail the discriminating assertion');
  }
});
// sc-26691 per-case storage reclaim. The layout mirrors CUDA run 40855384972's case-0 service root
// (data/models/{imports,resolved}, data/cache/minimax-h3-assemblies/<id>/{transformer,transformer_ref},
// data/projects/.../loras/imports) and its verbatim prepared_sources spelling.
async function reclaimFixture(t, { status = 'passed' } = {}) {
  const base = await mkdtemp(path.join(os.tmpdir(), 'sc26691-reclaim-'));
  t.after(() => rm(base, { recursive: true, force: true }));
  const output = path.join(base, '40855384972-1-bootstrap-ae69ef7b2928922370528f832c062c628f1a76b4');
  const c = { id: 'cuda.reduced.artifact-0-ordered-mixed-baseline-ref-turbo', evidence: 'evidence.json', inputs: [] };
  const p = { output };
  const service = path.join(output, 'services', c.id, 'data');
  const files = {
    imported: path.join(service, 'models', 'imports', 'imported_sc26691_artifact_0_ref2va_v2_int8', 'diffusion_pytorch_model.safetensors'),
    resolved: path.join(service, 'models', 'resolved', 'entries', 'bundle.safetensors'),
    ref: path.join(service, 'cache', 'minimax-h3-assemblies', '8ddcf6cab3958cccf6dc7d63c1483046', 'transformer_ref', 'diffusion_pytorch_model-00001-of-00014.safetensors'),
    kept: path.join(service, 'cache', 'minimax-h3-assemblies', '8ddcf6cab3958cccf6dc7d63c1483046', 'transformer', 'config.json'),
    lora: path.join(service, 'projects', 'h3-reduced-terminal-proof.sceneworks', 'loras', 'imports', 'sc26691_ref2v_turbo_4step_36bbcdc78d8d', 'minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors'),
  };
  for (const [name, file] of Object.entries(files)) { await mkdir(path.dirname(file), { recursive: true }); await writeFile(file, `unit fake ${name}`); }
  const prepared = await Promise.all(Object.values(files).map(async file => ({ path: process.platform === 'win32' ? `\\\\?\\${file}` : file, sha256: await hashFile(file) })));
  const loadEvent = { message: 'Actual native load', event: 'h3_import_prepared_load', jobId: 'job_81cbc014f0cc7171e185068d2e5888e3', provider: 'candle', backend: 'cuda', native_backend: 'cuda', seal_validated: true, cache_state: 'Cold', identity: { checkpointId: 'managed/imported_sc26691_artifact_0_ref2va_v2_int8', planId: 'p', semanticDigest: `sha256:${'a'.repeat(64)}`, assembly: { family: 'minimax-h3', partition: 'ref2va', components: {}, componentClosure: [] }, sourceFiles: [] }, prepared_sources: prepared, operation: 'reference_to_video', seed: 1, steps: 4, scheduler_shift: null, level: 'info', reportedAt: '2026-10-28T03:12:44.101Z' };
  const dir = path.join(output, c.id);
  await mkdir(dir, { recursive: true });
  await writeFile(path.join(dir, 'evidence.json'), JSON.stringify({ loadEvent }));
  await writeFile(path.join(dir, 'execution.json'), JSON.stringify({ caseId: c.id, status }));
  return { base, output, p, c, dir, files, prepared };
}
const exists = file => stat(file).then(() => true, error => { if (error.code === 'ENOENT') return false; throw error; });
test('a passed case reclaims only its own import, resolved bundle and transformer_ref copies', async t => {
  const prev = [process.env.GITHUB_RUN_ID, process.env.GITHUB_RUN_ATTEMPT];
  process.env.GITHUB_RUN_ID = '40855384972'; process.env.GITHUB_RUN_ATTEMPT = '1';
  t.after(() => { if (prev[0] === undefined) delete process.env.GITHUB_RUN_ID; else process.env.GITHUB_RUN_ID = prev[0]; if (prev[1] === undefined) delete process.env.GITHUB_RUN_ATTEMPT; else process.env.GITHUB_RUN_ATTEMPT = prev[1]; });
  const f = await reclaimFixture(t);
  const sibling = path.join(f.base, 'staging', 'model.safetensors'); await mkdir(path.dirname(sibling), { recursive: true }); await writeFile(sibling, 'shared staging');
  const record = await reclaimCaseModelStorage(f.p, f.c, f.dir);
  assert.equal(record.removed.length, 3);
  for (const name of ['imported', 'resolved', 'ref']) assert.equal(await exists(f.files[name]), false, `${name} reclaimed`);
  for (const name of ['kept', 'lora']) assert.equal(await exists(f.files[name]), true, `${name} kept`);
  assert.equal(await exists(sibling), true, 'nothing outside the case service root is touched');
  assert.deepEqual(record.preparedFiles.map(v => v.sha256).sort(), f.prepared.filter((_, i) => i < 3).map(v => v.sha256).sort());
  const retained = await caseStorageReclaim(f.p, f.c, f.dir);
  for (const [i, input] of f.prepared.entries()) assert.equal(await retained.removedFile(input), i < 3, 'only removed, recorded files are excused from re-hashing');
  assert.equal(await retained.removedFile({ ...f.prepared[0], sha256: 'f'.repeat(64) }), false, 'a different hash is never excused');
  // Idempotent on resume; a later change to the passed receipt unbinds the record.
  assert.deepEqual(await reclaimCaseModelStorage(f.p, f.c, f.dir), record);
  await writeFile(path.join(f.dir, 'execution.json'), JSON.stringify({ caseId: f.c.id, status: 'passed', changed: true }));
  await assert.rejects(caseStorageReclaim(f.p, f.c, f.dir), /not bound to this passed execution/);
});
test('storage reclaim refuses non-passed cases, other runs\' roots and symlinked targets', async t => {
  const prev = [process.env.GITHUB_RUN_ID, process.env.GITHUB_RUN_ATTEMPT];
  t.after(() => { if (prev[0] === undefined) delete process.env.GITHUB_RUN_ID; else process.env.GITHUB_RUN_ID = prev[0]; if (prev[1] === undefined) delete process.env.GITHUB_RUN_ATTEMPT; else process.env.GITHUB_RUN_ATTEMPT = prev[1]; });
  process.env.GITHUB_RUN_ID = '40855384972'; process.env.GITHUB_RUN_ATTEMPT = '1';
  const failed = await reclaimFixture(t, { status: 'failed' });
  await assert.rejects(reclaimCaseModelStorage(failed.p, failed.c, failed.dir), /only a recorded passed case/);
  assert.equal(await exists(failed.files.imported), true);
  process.env.GITHUB_RUN_ID = '40855384999';
  const other = await reclaimFixture(t);
  await assert.rejects(reclaimCaseModelStorage(other.p, other.c, other.dir), /confined to this run's own private root/);
  assert.equal(await exists(other.files.imported), true);
  process.env.GITHUB_RUN_ID = '40855384972';
  const linked = await reclaimFixture(t), outside = path.join(linked.base, 'shared-hf-cache');
  await mkdir(outside, { recursive: true }); await writeFile(path.join(outside, 'blob'), 'shared');
  const resolved = path.dirname(path.dirname(linked.files.resolved)); await rm(resolved, { recursive: true });
  await symlink(outside, resolved, 'junction');
  await assert.rejects(reclaimCaseModelStorage(linked.p, linked.c, linked.dir), /not a plain directory|escapes the case service root/);
  assert.equal(await exists(path.join(outside, 'blob')), true, 'a linked target never deletes what it points at');
  assert.equal(await exists(path.join(linked.dir, 'storage-reclaim.json')), false, 'no record when the guard refuses');
});
test('per-case free-space preflight refuses below 2x checkpoint plus headroom', async () => {
  const c = { id: 'cuda.reduced.artifact-4', inputs: [{ role: 'checkpoint', path: 'E:\\staging\\x.safetensors' }], runtime: { scenario: 'unit_fake' } };
  const checkpoint = 34004507622, statImpl = async () => ({ size: checkpoint }), bsize = 4096;
  const need = 2 * checkpoint + CASE_FREE_SPACE_HEADROOM_BYTES;
  assert.deepEqual(await assertCaseFreeSpace({ output: 'E:\\run' }, c, { statImpl, statfsImpl: async () => ({ bsize, bavail: Math.ceil(need / bsize) }) }), { free: Math.ceil(need / bsize) * bsize, need });
  await assert.rejects(assertCaseFreeSpace({ output: 'E:\\run' }, c, { statImpl, statfsImpl: async () => ({ bsize, bavail: Math.floor(need / bsize) - 1 }) }), /insufficient free space for cuda\.reduced\.artifact-4: .* GiB free on the run volume, need >= 79\.3 GiB/);
  assert.equal(await assertCaseFreeSpace({ output: 'E:\\run' }, { ...c, runtime: {} }, { statImpl, statfsImpl: async () => ({ bsize, bavail: 0 }) }), null, 'cases that import nothing are not gated');
});
// sc-26691 adaptive peer fail-fast. Rows are the real Actions API rows of the paired dispatches
// 40855384972 (CUDA) and 40855343040 (MLX) at ae69ef7b2 (trimmed to the fields the policy reads).
const PEER_SHA = 'ae69ef7b2928922370528f832c062c628f1a76b4';
const realRun = (id, status, conclusion) => ({ id, name: 'Server Candle (Linux/NVIDIA)', event: 'workflow_dispatch', status, conclusion, head_sha: PEER_SHA, run_attempt: 1, path: '.github/workflows/server-candle-linux.yml', workflow_id: 297197519 });
const realJobs = (backend, status, conclusion) => [
  { id: backend === 'mlx' ? 123384796089 : 123384930458, name: 'h3-terminal / terminal', status, conclusion, labels: backend === 'mlx' ? ['self-hosted', 'macOS', 'ARM64', 'real-weights'] : ['self-hosted', 'windows', 'cuda', 'real-weights', 'cuda-windows'], runner_name: backend === 'mlx' ? 'nax-macos' : 'cuda-windows' },
  { id: 123384934525, name: 'h3-import-mlx', status: 'completed', conclusion: 'skipped', labels: ['self-hosted', 'macOS', 'ARM64', 'rw-llm'], runner_name: null },
  { id: 123384996509, name: 'h3-import-cuda', status: 'completed', conclusion: 'skipped', labels: ['self-hosted', 'Windows', 'X64', 'cuda'], runner_name: null },
];
function fakeGh({ runs, jobs, artifacts = [] }) {
  const calls = [];
  const gh = args => {
    const url = args.find(a => a.startsWith('repos/')); calls.push(url);
    const runArtifacts = /runs\/(\d+)\/artifacts/.exec(url)?.[1];
    if (runArtifacts) return JSON.stringify([{ artifacts: (typeof artifacts === 'function' ? artifacts() : artifacts).filter(a => String(a.workflow_run?.id) === runArtifacts) }]);
    if (typeof runs === 'function') { if (url.includes('/actions/workflows/')) return JSON.stringify([{ workflow_runs: runs() }]); }
    if (url.includes('/actions/workflows/')) { assert.match(url, new RegExp(`head_sha=${PEER_SHA}`)); return JSON.stringify([{ workflow_runs: runs }]); }
    const id = /runs\/(\d+)\/attempts\/\d+\/jobs/.exec(url)?.[1]; assert.ok(id, url); return JSON.stringify([{ jobs: (typeof jobs === 'function' ? jobs() : jobs)[id] ?? [] }]);
  };
  return { gh, calls };
}
test('adaptive CUDA lane proceeds when the paired MLX terminal job is queued or running', async () => {
  for (const status of ['queued', 'in_progress']) {
    const { gh } = fakeGh({ runs: [realRun(40855384972, 'in_progress', null), realRun(40855343040, status, null)], jobs: { 40855343040: realJobs('mlx', status, null) } });
    const result = await awaitPeerLane({ gh, repository: 'SceneWorks/SceneWorks', headSha: PEER_SHA, selfRunId: 40855384972, peerBackend: 'mlx', anchorName: 'x', startedAt: 0, now: () => 0, sleep: async () => assert.fail('no wait') });
    assert.equal(result.state, 'alive'); assert.equal(result.run.id, 40855343040);
  }
  // Its own (CUDA) job never counts as the MLX peer.
  assert.equal(classifyPeerLane({ runs: [realRun(40855343040, 'in_progress', null)], jobsByRun: new Map([['40855343040', realJobs('cuda', 'in_progress', null)]]), selfRunId: 1, headSha: PEER_SHA, peerBackend: 'mlx' }).state, 'absent');
});
test('adaptive lane fails within the grace window when the peer is absent or already ended', async () => {
  const sleeps = [];
  let clock = 0;
  const absent = fakeGh({ runs: [realRun(40855384972, 'in_progress', null)], jobs: {} });
  await assert.rejects(awaitPeerLane({ gh: absent.gh, repository: 'SceneWorks/SceneWorks', headSha: PEER_SHA, selfRunId: 40855384972, peerBackend: 'mlx', anchorName: 'x', startedAt: 0, now: () => clock, sleep: async ms => { sleeps.push(ms); clock += ms; } }), /no paired mlx terminal run was dispatched at this SceneWorks SHA within 10 minutes/);
  assert.ok(clock <= PEER_APPEARANCE_GRACE_MS + 20000, 'fails in minutes, not after a long anchor wait');
  // The real cancelled MLX peer of this pair: refused once the grace has passed.
  const ended = fakeGh({ runs: [realRun(40855384972, 'in_progress', null), realRun(40855343040, 'completed', 'cancelled')], jobs: { 40855343040: realJobs('mlx', 'completed', 'cancelled') } });
  clock = 0;
  await assert.rejects(awaitPeerLane({ gh: ended.gh, repository: 'SceneWorks/SceneWorks', headSha: PEER_SHA, selfRunId: 40855384972, peerBackend: 'mlx', anchorName: 'x', startedAt: 0, now: () => clock, sleep: async ms => { clock += ms; } }), /run 40855343040 \(job 123384796089\) already ended as cancelled without publishing inline anchors/);
  // In the wait loop (hours after start) a failed peer is refused on the first check.
  assert.throws(() => assertPeerCanDeliver(classifyPeerLane({ runs: [realRun(40855343040, 'completed', 'failure')], jobsByRun: new Map([['40855343040', realJobs('mlx', 'completed', 'failure')]]), selfRunId: 40855384972, headSha: PEER_SHA, peerBackend: 'mlx' }), { peerBackend: 'mlx', now: 6 * 3600000, startedAt: 0, anchorsPublished: false }), /already ended as failure/);
  // A peer that succeeded without anchors never ran the adaptive matrix.
  assert.throws(() => assertPeerCanDeliver(classifyPeerLane({ runs: [realRun(40855343040, 'completed', 'success')], jobsByRun: new Map([['40855343040', realJobs('mlx', 'completed', 'success')]]), selfRunId: 40855384972, headSha: PEER_SHA, peerBackend: 'mlx' }), { peerBackend: 'mlx', now: 6 * 3600000, startedAt: 0, anchorsPublished: false }), /finished successfully without publishing inline anchors/);
});
test('published peer anchors satisfy the preflight even when the peer has since ended', async () => {
  const anchors = { id: 9, name: 'a-attempt-1', expired: false, created_at: '2026-10-29T10:40:00Z', workflow_run: { id: 40855343040, head_sha: PEER_SHA } };
  const { gh, calls } = fakeGh({ runs: [realRun(40855384972, 'in_progress', null), realRun(40855343040, 'completed', 'failure')], jobs: { 40855343040: realJobs('mlx', 'completed', 'failure') }, artifacts: [anchors, { ...anchors, id: 10, workflow_run: { id: 40855384972, head_sha: PEER_SHA } }] });
  const result = await awaitPeerLane({ gh, repository: 'SceneWorks/SceneWorks', headSha: PEER_SHA, selfRunId: 40855384972, peerBackend: 'mlx', anchorName: 'a', startedAt: 0, now: () => 9e9, sleep: async () => assert.fail('no wait') });
  assert.equal(result.state, 'anchors'); assert.equal(result.anchors.id, 9, 'never its own run\'s artifact');
  assert.ok(!calls.some(url => url.includes('/runs/40855384972/')), 'its own run is never queried');
});
// The anchor wait (sc-26691) is bounded by the peer's liveness and this lane's own work deadline,
// not a fixed six hours. Hour-scale clocks are virtual.
const HOUR = 3600000;
const anchorBase = anchorArtifactBase({ backend: 'mlx', sceneworks: PEER_SHA, inference: 'b'.repeat(40), evidence: 'c'.repeat(40) });
const anchorArtifact = (id, attempt, createdAt, runId = 40855343040) => ({ id, name: anchorArtifactName(anchorBase, attempt), expired: false, created_at: createdAt, workflow_run: { id: runId, head_sha: PEER_SHA } });
const waitArgs = (gh, clock, extra) => ({ gh, repository: 'SceneWorks/SceneWorks', name: anchorBase, headSha: PEER_SHA, selfRunId: 40855384972, peerBackend: 'mlx', startedAt: 0, now: () => clock.t, sleep: async ms => { clock.t += ms; }, download: async artifact => `dir-${artifact.id}`, ...extra });
const livePair = () => [realRun(40855384972, 'in_progress', null), realRun(40855343040, 'in_progress', null)];
test('the anchor wait continues past six hours while the peer lane is alive', async () => {
  const clock = { t: 0 }, published = anchorArtifact(41, 1, '2026-10-29T18:00:00Z');
  const { gh } = fakeGh({ runs: livePair(), jobs: { 40855343040: realJobs('mlx', 'in_progress', null) }, artifacts: () => clock.t >= 7.5 * HOUR ? [published] : [] });
  const seen = [];
  const result = await waitForPeerAnchors(waitArgs(gh, clock, { deadline: 24 * HOUR, validate: async (artifact, directory) => { seen.push([artifact.id, directory]); return { decision: 'cuda' }; } }));
  assert.equal(result.artifact.id, 41); assert.equal(result.artifact.runAttempt, 1); assert.deepEqual(result.result, { decision: 'cuda' });
  assert.ok(clock.t >= 7.5 * HOUR, 'waited beyond the old six-hour cap for a live peer'); assert.deepEqual(seen, [[41, 'dir-41']]);
});
test('the anchor wait stops as soon as the peer lane is observed failed', async () => {
  const clock = { t: 0 };
  const { gh } = fakeGh({ runs: () => [realRun(40855384972, 'in_progress', null), realRun(40855343040, clock.t >= 3 * HOUR ? 'completed' : 'in_progress', clock.t >= 3 * HOUR ? 'failure' : null)], jobs: () => ({ 40855343040: clock.t >= 3 * HOUR ? realJobs('mlx', 'completed', 'failure') : realJobs('mlx', 'in_progress', null) }) });
  await assert.rejects(waitForPeerAnchors(waitArgs(gh, clock, { deadline: 24 * HOUR, validate: async () => assert.fail('nothing to validate') })), /already ended as failure without publishing inline anchors/);
  assert.ok(clock.t >= 3 * HOUR && clock.t <= 3 * HOUR + 5 * 60000, `stopped ${clock.t - 3 * HOUR} ms after the peer failed (within one backed-off poll)`);
});
test('the anchor wait is bounded by its deadline, never a fixed cap, and backs off while nothing changes', async () => {
  const clock = { t: 0 };
  const { gh, calls } = fakeGh({ runs: livePair(), jobs: { 40855343040: realJobs('mlx', 'in_progress', null) } });
  await assert.rejects(waitForPeerAnchors(waitArgs(gh, clock, { deadline: 19 * HOUR, validate: async () => assert.fail('nothing to validate') })), /reached its peer-anchor wait deadline 1970-01-01T19:00:00.000Z \(its work deadline minus the time its remaining cases need\) while the paired mlx lane was still able to deliver/);
  assert.equal(clock.t, 19 * HOUR, 'the deadline is the bound, exactly');
  // 19 h of an unchanged live peer: the poll backs off to 5 min, so the Actions API budget holds.
  assert.ok(calls.length < 19 * 12 * 3 + 30, `${calls.length} gh api calls in 19 h`);
  await assert.rejects(waitForPeerAnchors(waitArgs(gh, { t: 0 }, { deadline: undefined, validate: async () => {} })), /work deadline/);
  await assert.rejects(waitForPeerAnchors(waitArgs(gh, { t: 0 }, { deadline: HOUR, download: undefined, validate: async () => {} })), /separate download/);
});
test('rerun attempts publish distinct artifacts and the newest valid one is selected', async () => {
  assert.equal(anchorArtifactName(anchorBase, 2), `${anchorBase}-attempt-2`); assert.throws(() => anchorArtifactName(anchorBase, 0));
  assert.equal(anchorArtifactAttempt(`${anchorBase}-attempt-3`, anchorBase), 3);
  for (const foreign of [anchorBase, `${anchorBase}-attempt-x`, `${anchorBase}-attempt-03`, `h3-anchors-cuda-${PEER_SHA}-attempt-1`]) assert.equal(anchorArtifactAttempt(foreign, anchorBase), null, foreign);
  // The same peer run, attempts 1 and 2 (no 409: distinct names); an expired one, and this lane's own, never count.
  const attempt1 = anchorArtifact(51, 1, '2026-10-29T10:00:00Z'), attempt2 = anchorArtifact(52, 2, '2026-10-29T14:00:00Z');
  const { gh } = fakeGh({ runs: [realRun(40855384972, 'in_progress', null), { ...realRun(40855343040, 'in_progress', null), run_attempt: 2 }], jobs: { 40855343040: realJobs('mlx', 'in_progress', null) }, artifacts: [attempt1, attempt2, { ...anchorArtifact(53, 3, '2026-10-29T15:00:00Z'), expired: true }, anchorArtifact(54, 1, '2026-10-29T16:00:00Z', 40855384972)] });
  assert.deepEqual((await peerAnchorCandidates({ gh, repository: 'SceneWorks/SceneWorks', name: anchorBase, headSha: PEER_SHA, selfRunId: 40855384972, peerBackend: 'mlx' })).map(a => [a.id, a.runAttempt]), [[52, 2], [51, 1]]);
  // Newest first; a candidate that fails VALIDATION falls through to the next-newest, and is not retried.
  const clock = { t: 0 }, tried = [], logs = [];
  const result = await waitForPeerAnchors(waitArgs(gh, clock, { deadline: 24 * HOUR, log: line => logs.push(line), validate: async artifact => { tried.push(artifact.id); if (artifact.runAttempt === 2) throw new Error('anchor packet attempt does not match its artifact'); return { attempt: artifact.runAttempt }; } }));
  assert.deepEqual(tried, [52, 51]); assert.equal(result.artifact.id, 51); assert.deepEqual(result.result, { attempt: 1 });
  assert.ok(logs.some(line => /H3_PEER_ANCHORS_REJECTED: artifact 52/.test(line)));
  assert.equal(clock.t, 0, 'no wait when valid anchors are already published');
});
test('a failed anchor download is retried with backoff, never treated as an invalid packet', async () => {
  const clock = { t: 0 }, published = anchorArtifact(61, 1, '2026-10-29T10:00:00Z');
  const { gh } = fakeGh({ runs: livePair(), jobs: { 40855343040: realJobs('mlx', 'in_progress', null) }, artifacts: [published] });
  const directories = [], logs = [];
  let downloads = 0;
  const result = await waitForPeerAnchors(waitArgs(gh, clock, { deadline: 24 * HOUR, log: line => logs.push(line),
    download: async () => { const directory = `fresh-${++downloads}`; directories.push(directory); if (downloads <= 4) throw new Error('gh run download: HTTP 502'); return directory; },
    validate: async (artifact, directory) => ({ directory }) }));
  assert.equal(result.artifact.id, 61); assert.equal(result.directory, 'fresh-5'); assert.equal(new Set(directories).size, 5, 'a fresh directory per try');
  assert.ok(clock.t > 0, 'backed off between tries'); assert.equal(logs.filter(line => /H3_PEER_ANCHORS_DOWNLOAD_RETRY: artifact 61/.test(line)).length, 4);
  assert.ok(!logs.some(line => /REJECTED/.test(line)));
});
test('transient Actions API failures are retried; completed runs are cached; only a long outage ends the wait', async () => {
  const clock = { t: 0 };
  const flaky = fakeGh({ runs: livePair(), jobs: { 40855343040: realJobs('mlx', 'in_progress', null) }, artifacts: () => clock.t >= HOUR ? [anchorArtifact(71, 1, '2026-10-29T10:00:00Z')] : [] });
  let n = 0; const gh = args => { if (++n % 3 === 0) throw new Error('HTTP 502: Bad Gateway'); return flaky.gh(args); };
  const result = await waitForPeerAnchors(waitArgs(gh, clock, { deadline: 24 * HOUR, validate: async () => 'ok' }));
  assert.equal(result.artifact.id, 71, 'one 5xx no longer kills the lane');
  // A completed peer run's jobs and artifacts are fetched once, however many polls follow.
  const done = fakeGh({ runs: [realRun(40855384972, 'in_progress', null), realRun(40855343040, 'completed', 'success'), realRun(40855340000, 'queued', null)], jobs: { 40855343040: realJobs('mlx', 'completed', 'success') } });
  const { PeerLaneObserver } = await import('./lib/h3-peer-lane.mjs');
  const observer = new PeerLaneObserver({ gh: done.gh, repository: 'SceneWorks/SceneWorks', headSha: PEER_SHA, selfRunId: 40855384972, peerBackend: 'mlx', sleep: async () => {} });
  for (let i = 0; i < 5; i++) await observer.observe({ name: anchorBase });
  assert.equal(done.calls.filter(url => url.includes('/runs/40855343040/')).length, 2, 'completed run: jobs and artifacts once each');
  // Every call failing: the outage limit ends the wait with the cause, not the first error.
  const down = { t: 0 };
  await assert.rejects(waitForPeerAnchors(waitArgs(() => { throw new Error('HTTP 503'); }, down, { deadline: 24 * HOUR, validate: async () => {} })), /Actions API stayed unreachable for \d+ minutes/);
  assert.ok(down.t >= 30 * 60000 && down.t < 2 * HOUR);
});
test('anchors come only from the paired peer run, never from a finished earlier pair at the same SHA', async () => {
  const stale = anchorArtifact(81, 1, '2026-10-28T10:00:00Z', 40855300000), live = anchorArtifact(82, 1, '2026-10-29T12:00:00Z');
  const clock = { t: 0 };
  const { gh } = fakeGh({ runs: [realRun(40855300000, 'completed', 'success'), realRun(40855384972, 'in_progress', null), realRun(40855343040, 'in_progress', null)], jobs: { 40855300000: realJobs('mlx', 'completed', 'success'), 40855343040: realJobs('mlx', 'in_progress', null) }, artifacts: () => clock.t >= 2 * HOUR ? [stale, live] : [stale] });
  const result = await waitForPeerAnchors(waitArgs(gh, clock, { deadline: 24 * HOUR, validate: async artifact => artifact.id }));
  assert.equal(result.artifact.id, 82); assert.ok(clock.t >= 2 * HOUR, 'the stale pair\'s anchors were not taken');
  // A newer dispatch whose jobs do not exist yet may be the real peer: nothing is taken meanwhile.
  const { pairedPeerRun } = await import('./lib/h3-peer-lane.mjs');
  const jobsByRun = new Map([['40855343040', realJobs('mlx', 'completed', 'success')], ['40855390000', []]]);
  assert.equal(pairedPeerRun({ runs: [realRun(40855343040, 'completed', 'success'), realRun(40855390000, 'queued', null)], jobsByRun, selfRunId: 1, headSha: PEER_SHA, peerBackend: 'mlx' }), null);
  assert.equal(pairedPeerRun({ runs: [realRun(40855343040, 'completed', 'success'), realRun(40855390000, 'completed', 'cancelled')], jobsByRun, selfRunId: 1, headSha: PEER_SHA, peerBackend: 'mlx' }).id, 40855343040);
});
test('a dispatch whose jobs are not created yet is a possible peer only while active', () => {
  assert.equal(classifyPeerLane({ runs: [realRun(40855343040, 'queued', null)], jobsByRun: new Map([['40855343040', []]]), selfRunId: 1, headSha: PEER_SHA, peerBackend: 'mlx' }).state, 'alive');
  assert.equal(classifyPeerLane({ runs: [realRun(40855343040, 'completed', 'cancelled')], jobsByRun: new Map([['40855343040', []]]), selfRunId: 1, headSha: PEER_SHA, peerBackend: 'mlx' }).state, 'absent');
  assert.equal(classifyPeerLane({ runs: [{ ...realRun(40855343040, 'queued', null), head_sha: 'c85f515525819c8ddc87af09e53dd9b829c9d150' }], jobsByRun: new Map(), selfRunId: 1, headSha: PEER_SHA, peerBackend: 'mlx' }).state, 'absent', 'a run at another SHA is never the peer');
});
