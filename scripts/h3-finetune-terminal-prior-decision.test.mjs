// sc-26691 (2026-11-01): a lane runs an explicit case subset at a new SceneWorks SHA on a DECLARED
// prior measured matrix decision, and one backend's evidence split across two runs is re-validated
// as two attributed segments. Fixtures are the REAL retained anchor packets of runs 41213935677
// (cuda) / 41213914294 (mlx) and the decision fields of run 41213935677's retained adapter plan.
import test from 'node:test';
import assert from 'node:assert/strict';
import { cp, mkdir, mkdtemp, readFile, rm, writeFile, appendFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { SOURCE, adoptPriorMatrixDecision, executionCases, executionPlanDigest, verifyMatrixAdmission, verifySplitBackendEvidence, PRIOR_DECISION_KIND } from './h3-finetune-terminal.mjs';
import { operationalCases } from './h3-finetune-terminal-reduced.mjs';
import { inferencePinAncestry, loadPriorDecisionEvidence, loadPriorLaneEvidence, peerPacketArtifactId, reducedEvidenceArtifact, retainedDecisionPlan, retainedLanePlan } from './lib/h3-prior-decision.mjs';

const FIXTURE = fileURLToPath(new URL('./fixtures/sc26691-prior-decision/', import.meta.url));
const PRIOR_SW = '4c3625898b20859f57ca38986ed80b1c427cdfdb', PIN = '354b41ccaccdc14afccff0d0fb1ae1b0be04836e', EVIDENCE = '0bada8f7287f414c30f77eb1d29ef4a544eb8ffc';
const NEW_SW = 'b'.repeat(40), RUN = '41213935677', PEER_RUN = '41213914294';
// The MLX lane's re-pin (sc-26691): inference a43724b8 descends from run 41213935677's 354b41cc. The
// first-parent log is the REAL `git log --first-parent --format='%H %s' 354b41cc..a43724b8`.
const NEW_PIN = 'a43724b82fbae35c9227cef4bcdabba6dd1da12c';
const FIRST_PARENT_LOG = [
  'a43724b82fbae35c9227cef4bcdabba6dd1da12c Merge pull request #1405 from SceneWorks/sc-26691-keyframe-contract',
  '70dbe4f41819fcec206737f5d73b428aaed37b6e Merge pull request #1402 from SceneWorks/codex/sc-26850-krea-paired-comparison',
  '3c52eef46c902d296ce84d77a65e32fe126b32d5 Merge pull request #1404 from SceneWorks/feature/sc-27980-generic-load-progress',
  '4c137be754ed38d4c5a2195327a515725d4b31d3 Merge pull request #1403 from SceneWorks/feature/sc-27980-generic-load-progress',
  'fb8948a629dee07211c3ae9333d8dd885a68039a Merge pull request #1401 from SceneWorks/feature/sc-26850-cuda-training-acceleration',
  '35159482adf6c29649c6d97daac3dea51c637411 Merge pull request #1400 from SceneWorks/feature/sc-27980-generic-load-progress',
  '39219b9e6550bc33f340ca2533c2d3422a5579a6 Merge pull request #1399 from SceneWorks/feature/sc-26850-cuda-training-acceleration',
  '69c25a1d87f761464cb200802dfbb04125e9f54c Merge pull request #1398 from SceneWorks/feature/sc-26850-cuda-training-acceleration',
  '54ad274dd4fd28b02faece88f268aa2a2ddaf86b Merge pull request #1396 from SceneWorks/feature/sc-26850-cuda-training-acceleration',
  '932ddff6ece44c5a4ba80c17307815785cf3af0e Merge pull request #1395 from SceneWorks/claude/ep27252-test-helpers',
  '7e0a34e4e774750cc2a2eee098a5d7ae2347bebb Merge pull request #1394 from SceneWorks/feature/sc-26850-cuda-training-acceleration',
];
const descends = async ({ prior, own }) => ({ prior, own, descends: true, firstParentLog: FIRST_PARENT_LOG });
const REMAINING = ['real-hybrid-first-last', 'real-hybrid-reference', 'real-independent-fl2va', 'real-hybrid-text-repeat', 'numeric-pruned-artifact', 'numeric-w4a8-artifact', 'lifecycle-generation-source-prep-cancel'].map(slug => `cuda.reduced.${slug}`);

async function workspace(t) {
  const root = await mkdtemp(path.join(os.tmpdir(), 'sc26691-prior-decision-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  for (const b of ['cuda', 'mlx']) await cp(path.join(FIXTURE, `anchors-${b}`), path.join(root, `anchors-${b}`), { recursive: true });
  const decisionPlan = JSON.parse(await readFile(path.join(FIXTURE, 'decision-plan.json'), 'utf8'));
  // The 18 cases run 41213935677 actually passed: every cuda case but the 7 it never reached.
  const evidence = path.join(root, 'evidence');
  for (const c of operationalCases().filter(c => c.backend === 'cuda' && !REMAINING.includes(c.id))) {
    await mkdir(path.join(evidence, c.id), { recursive: true });
    await writeFile(path.join(evidence, c.id, 'execution.json'), JSON.stringify({ caseId: c.id, status: 'passed', runId: RUN, runAttempt: '1', sceneworks: PRIOR_SW, inference: PIN }));
  }
  const plan = { version: 1, inventory: 'reduced-v1', sceneworks: NEW_SW, inference: PIN, source: SOURCE, nativeEvidence: { ref: 'main', sha: EVIDENCE, productPin: PIN }, output: root, cases: operationalCases() };
  const packets = { cuda: { directory: path.join(root, 'anchors-cuda'), runId: RUN, runAttempt: '1' }, mlx: { directory: path.join(root, 'anchors-mlx'), runId: PEER_RUN, runAttempt: '1' } };
  return { root, decisionPlan, evidence, plan, packets, decisionRun: { runId: RUN, attempt: '1', backend: 'cuda' } };
}
const adopt = (w, overrides = {}) => adoptPriorMatrixDecision(w.plan, { backend: 'cuda', decisionRun: w.decisionRun, decisionPlan: w.decisionPlan, packets: w.packets, priorEvidenceRoot: w.evidence, caseIds: REMAINING, changedFiles: ['scripts/h3-finetune-terminal.mjs'], ...overrides });

test('the CUDA remaining-cases lane adopts run 41213935677\'s measured decision and runs exactly the 7 cases it never reached', async t => {
  const w = await workspace(t), record = await adopt(w);
  assert.equal(record.kind, PRIOR_DECISION_KIND);
  assert.deepEqual(record.decisionRun, { runId: RUN, attempt: '1', backend: 'cuda' });
  assert.deepEqual(record.anchorPackets, { mlx: { runId: PEER_RUN, runAttempt: '1', retainedDirectory: 'anchors-mlx' }, cuda: { runId: RUN, runAttempt: '1', retainedDirectory: 'anchors-cuda' } });
  // The packets' native witnesses are REUSED (same native evidence SHA), and that reuse is recorded.
  assert.deepEqual(record.reusedNativeWitnesses.map(v => [v.backend, v.nativeEvidenceSha, v.runRef, v.packet.runId, v.receipt.path, v.logs.map(l => l.path)]),
    [['mlx', EVIDENCE, `actions:${PEER_RUN}/1`, PEER_RUN, 'anchors-mlx/witness-0.json', ['anchors-mlx/witness-0-0.log']], ['cuda', EVIDENCE, `actions:${RUN}/1`, RUN, 'anchors-cuda/witness-0.json', ['anchors-cuda/witness-0-0.log']]]);
  for (const v of record.reusedNativeWitnesses) assert.match(v.receipt.sha256, /^[0-9a-f]{64}$/);
  assert.equal(record.decision.selectedBackend, 'cuda'); assert.equal(record.decision.displacedBackend, 'mlx');
  assert.deepEqual(record.decision, w.decisionPlan.matrixDecision, 'recomputed from the retained anchors, equal to the retained decision');
  assert.equal(record.allocation.length, 25); assert.equal(record.priorPassedCaseIds.length, 18);
  assert.deepEqual(record.subsetCaseIds, REMAINING); assert.deepEqual(record.notCovered, []);
  assert.equal(record.caseDefinitionBasis, 'same-case'); assert.deepEqual(record.caseDefinitionDifferences, [], 'run 41213935677\'s retained definitions match today\'s');
  assert.deepEqual(record.sourceShaDifference, { from: PRIOR_SW, to: NEW_SW, files: ['scripts/h3-finetune-terminal.mjs'] });
  assert.equal(w.plan.matrixAnchors.length, 4);
  assert.deepEqual(w.plan.proofIndex.map(p => p.backend).sort(), ['cuda', 'mlx'], 'both lanes\' retained native witnesses come with their packets');
  await verifyMatrixAdmission(w.plan);
  // With no explicit subset the default is the allocation minus what already passed: the same 7.
  const fresh = await workspace(t);
  assert.deepEqual((await adopt(fresh, { caseIds: null })).subsetCaseIds, REMAINING);
});
test('the MLX lane adopts the same decision for its full displaced allocation; the cuda run\'s passed cases never count for it', async t => {
  const w = await workspace(t), record = await adopt(w, { backend: 'mlx', caseIds: null });
  assert.equal(record.allocation.length, 18); assert.deepEqual(record.priorPassedCaseIds, []);
  assert.deepEqual(record.subsetCaseIds, executionCases(w.plan).filter(c => c.backend === 'mlx').map(c => c.id));
  assert.ok(!record.subsetCaseIds.includes('mlx.reduced.real-drop-images'), 'a broad case omitted on the displaced backend stays omitted');
  // The retained cuda plan defines no mlx case: the mlx lane is compared with each same-slug cuda counterpart.
  assert.equal(record.caseDefinitionBasis, 'peer-backend-counterpart'); assert.deepEqual(record.caseDefinitionDifferences, []);
});
test('a prior-decision subset refuses cases that already passed, cases outside the allocation and drifted evidence', async t => {
  await assert.rejects(adopt(await workspace(t), { caseIds: ['cuda.reduced.real-drop-audio'] }), /already PASSED in run 41213935677/);
  await assert.rejects(adopt(await workspace(t), { backend: 'mlx', caseIds: ['mlx.reduced.real-drop-images'] }), /not in the mlx lane's allocation/);
  await assert.rejects(adopt(await workspace(t), { changedFiles: undefined }), /git difference/);
  const pin = await workspace(t); pin.plan.inference = 'c'.repeat(40);
  await assert.rejects(adopt(pin), /measured at inference 354b41cc\w+, not this lane's c{40}; adopting it across a pin change needs the lane's inference ancestry/);
  const log = await workspace(t); await appendFile(path.join(log.packets.mlx.directory, 'witness-0-0.log'), 'tampered\n');
  await assert.rejects(adopt(log), /retained mlx anchor file witness-0-0.log changed/);
  const wall = await workspace(t); wall.decisionPlan.matrixAnchors[0].wallMs += 1;
  await assert.rejects(adopt(wall), /not the anchors the prior decision was measured from/);
  const decision = await workspace(t); decision.decisionPlan.matrixDecision.reason = 'edited';
  await assert.rejects(adopt(decision), /recomputed from the retained anchors differs/);
  const swapped = await workspace(t); swapped.packets.cuda.runAttempt = '2';
  await assert.rejects(adopt(swapped), /decision run's own anchor packet must come from that attempt/);
  const twice = await workspace(t); await adopt(twice);
  await assert.rejects(adopt(twice), /adopted once/);
  // The admission check keeps holding the plan to the recorded difference.
  const drift = await workspace(t); await adopt(drift); drift.plan.priorMatrixDecision.sourceShaDifference.to = 'd'.repeat(40);
  await assert.rejects(verifyMatrixAdmission(drift.plan), /Expected values to be strictly equal/);
});
// The MLX lane at a43724b8 adopts run 41213935677's decision, measured at 354b41cc (same native SHA).
const repinned = async t => { const w = await workspace(t); w.plan.inference = NEW_PIN; w.plan.nativeEvidence.productPin = NEW_PIN; return w; };
test('a lane at a DESCENDANT inference pin adopts the prior decision, validating the packets at their own pin and recording the change', async t => {
  const w = await repinned(t), asked = [];
  const record = await adopt(w, { backend: 'mlx', caseIds: null, inferenceAncestry: async v => { asked.push(v); return descends(v); } });
  assert.deepEqual(asked, [{ prior: PIN, own: NEW_PIN }]);
  assert.deepEqual(record.inferenceChange, { prior: PIN, own: NEW_PIN, descends: true, firstParentLog: FIRST_PARENT_LOG });
  assert.match(record.meaning, /ran at inference inferenceChange\.prior and were validated against that pin/);
  assert.equal(record.subsetCaseIds.length, 18); assert.deepEqual(record.decision, w.decisionPlan.matrixDecision);
  // The packets (productPin 354b41cc) and anchors were checked against THEIR pin, never the lane's.
  for (const b of ['mlx', 'cuda']) assert.equal(JSON.parse(await readFile(path.join(w.packets[b].directory, 'anchors.json'), 'utf8')).productPin, PIN);
  assert.ok(w.plan.matrixAnchors.every(anchor => anchor.pinSha === PIN)); assert.equal(w.plan.inference, NEW_PIN);
  await verifyMatrixAdmission(w.plan);
  // The admission check holds the plan to the recorded change.
  const own = await repinned(t); await adopt(own, { backend: 'mlx', caseIds: null, inferenceAncestry: descends }); own.plan.priorMatrixDecision.inferenceChange.own = 'c'.repeat(40);
  await assert.rejects(verifyMatrixAdmission(own.plan), /recorded inference change does not end at this plan's pin/);
  const dropped = await repinned(t); await adopt(dropped, { backend: 'mlx', caseIds: null, inferenceAncestry: descends }); delete dropped.plan.priorMatrixDecision.inferenceChange;
  await assert.rejects(verifyMatrixAdmission(dropped.plan), /Expected values to be strictly equal/, 'without the record the anchors must be at the plan\'s own pin');
  // Same pin: no ancestry is asked for and no inferenceChange is recorded.
  const same = await workspace(t), samePin = await adopt(same, { inferenceAncestry: () => assert.fail('same-pin adoption never asks for ancestry') });
  assert.equal('inferenceChange' in samePin, false);
});
test('a pin change is refused when the lane pin does not descend, when the native SHA changed, or when a packet is not at its own recorded pin', async t => {
  await assert.rejects(adopt(await repinned(t), { backend: 'mlx', caseIds: null, inferenceAncestry: async ({ prior, own }) => ({ prior, own, descends: false, firstParentLog: [] }) }), /a43724b8\w+ does not descend from the prior decision's pin 354b41cc\w+; the decision is not adopted across it/);
  await assert.rejects(adopt(await repinned(t), { backend: 'mlx', caseIds: null, inferenceAncestry: async ({ prior, own }) => ({ prior, own, descends: true, firstParentLog: [] }) }), /first-parent inference log between the two pins must be recorded/);
  await assert.rejects(adopt(await repinned(t), { backend: 'mlx', caseIds: null, inferenceAncestry: async ({ own }) => ({ prior: 'c'.repeat(40), own, descends: true, firstParentLog: FIRST_PARENT_LOG }) }), /ancestry answered for another prior pin/);
  const native = await repinned(t); native.plan.nativeEvidence.sha = 'e'.repeat(40);
  await assert.rejects(adopt(native, { backend: 'mlx', caseIds: null, inferenceAncestry: descends }), /must share this lane's native evidence SHA/);
  // The retained decision plan names another (descending) pin than its packets were measured at:
  // the packets are validated against the decision's recorded pin, so they are refused.
  const packet = await repinned(t); packet.decisionPlan.inference = 'd'.repeat(40);
  await assert.rejects(adopt(packet, { backend: 'mlx', caseIds: null, inferenceAncestry: descends }), /retained mlx anchors use another inference pin/);
});
test('inference ancestry is read from a real git checkout: descendant, non-descendant and unknown pins', async t => {
  const { execFileSync } = await import('node:child_process');
  const repo = await mkdtemp(path.join(os.tmpdir(), 'sc26691-inference-ancestry-')); t.after(() => rm(repo, { recursive: true, force: true }));
  const git = args => execFileSync('git', ['-c', 'user.name=t', '-c', 'user.email=t@example.invalid', '-c', 'commit.gpgsign=false', ...args], { cwd: repo, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] });
  git(['init', '-q', '-b', 'main']);
  const commit = subject => { git(['commit', '-q', '--allow-empty', '-m', subject]); return git(['rev-parse', 'HEAD']).trim(); };
  const base = commit('base pin');
  git(['checkout', '-q', '-b', 'side']); const side = commit('side work');
  git(['checkout', '-q', 'main']); const direct = commit('direct change');
  git(['merge', '-q', '--no-ff', '-m', 'Merge side', 'side']); const own = git(['rev-parse', 'HEAD']).trim();
  const r = inferencePinAncestry({ git, prior: base, own });
  assert.equal(r.descends, true); assert.deepEqual(r.firstParentLog, [`${own} Merge side`, `${direct} direct change`], 'first-parent only: the merged side commit is not listed');
  assert.ok(!r.firstParentLog.some(line => line.startsWith(side)));
  git(['checkout', '-q', '-b', 'other', base]); const other = commit('unrelated');
  assert.deepEqual(inferencePinAncestry({ git, prior: other, own }), { prior: other, own, descends: false, firstParentLog: [] });
  assert.throws(() => inferencePinAncestry({ git, prior: 'f'.repeat(40), own }), /cannot establish inference ancestry/);
  assert.throws(() => inferencePinAncestry({ git, prior: own, own }), /two different pins/);
});
test('a subset case whose workload changed since the decision run is refused; other definition changes are recorded', async t => {
  const steps = await workspace(t); steps.plan.cases.find(c => c.id === REMAINING[0]).backendSteps = { cuda: 8, mlx: 8 };
  await assert.rejects(adopt(steps), /subset case workload changed since run 41213935677.*cuda\.reduced\.real-hybrid-first-last \(backendSteps\)/);
  const absent = await workspace(t); absent.decisionPlan.cases = absent.decisionPlan.cases.filter(c => c.id !== REMAINING[1]);
  await assert.rejects(adopt(absent), /real-hybrid-reference \(\(not defined in the prior plan\)\)/);
  const peer = await workspace(t); peer.plan.cases.find(c => c.id === 'mlx.reduced.real-hybrid-reference').conditioning = 'first-last';
  await assert.rejects(adopt(peer, { backend: 'mlx', caseIds: null }), /mlx\.reduced\.real-hybrid-reference \(conditioning\)/, 'a cross-backend counterpart is compared too');
  // A prior-passed case's definition change and a subset case's input bytes are recorded, not refused.
  const recorded = await workspace(t);
  recorded.plan.cases.find(c => c.id === 'cuda.reduced.real-drop-audio').acceptance = { ac: ['edited'], e: [] };
  recorded.plan.cases.find(c => c.id === REMAINING[2]).inputs = [{ role: 'checkpoint', sha256: 'f'.repeat(64), path: '/x' }];
  const record = await adopt(recorded);
  assert.deepEqual(record.caseDefinitionDifferences, [
    { id: 'cuda.reduced.real-drop-audio', subset: false, workloadFields: ['acceptance'], otherFields: [] },
    { id: REMAINING[2], subset: true, workloadFields: [], otherFields: ['inputIdentities'] },
  ].sort((a, b) => record.allocation.indexOf(a.id) - record.allocation.indexOf(b.id)));
  const missing = await workspace(t); delete missing.decisionPlan.cases;
  await assert.rejects(adopt(missing), /must carry its case definitions/);
});
test('the retained evidence is located by the declared run alone: its reduced artifact, its own packet, and the peer packet its anchors name', async t => {
  const w = await workspace(t), work = path.join(w.root, 'fetched'), downloads = [];
  const ownName = `h3-anchors-cuda-${PRIOR_SW}-${PIN}-${EVIDENCE}-attempt-1`, peerName = `h3-anchors-mlx-${PRIOR_SW}-${PIN}-${EVIDENCE}-attempt-1`;
  const gh = args => {
    const route = args[1];
    if (route === `repos/SceneWorks/SceneWorks/actions/runs/${RUN}`) return JSON.stringify({ status: 'completed', repository: { full_name: 'SceneWorks/SceneWorks' } });
    if (route.startsWith(`repos/SceneWorks/SceneWorks/actions/runs/${RUN}/artifacts`)) return JSON.stringify({ artifacts: [{ name: `h3-terminal-cuda-reduced-${RUN}-1`, expired: false }, { name: `h3-terminal-stage-cuda-${PRIOR_SW}-${RUN}-1` }, { name: ownName, expired: false }] });
    if (route === 'repos/SceneWorks/SceneWorks/actions/artifacts/13051908009') return JSON.stringify({ name: peerName, expired: false, workflow_run: { id: Number(PEER_RUN) } });
    throw new Error(`unexpected gh ${args.join(' ')}`);
  };
  const download = async ({ runId, name, directory }) => {
    downloads.push([runId, name]);
    if (name.startsWith('h3-terminal-cuda-reduced-')) { await cp(w.evidence, directory, { recursive: true }); await writeFile(path.join(directory, 'cuda.reduced.real-drop-audio', 'adapter-plan.json'), JSON.stringify(w.decisionPlan)); }
    else await cp(path.join(FIXTURE, name === ownName ? 'anchors-cuda' : 'anchors-mlx'), directory, { recursive: true });
  };
  const loaded = await loadPriorDecisionEvidence({ gh, download, repository: 'SceneWorks/SceneWorks', runId: RUN, attempt: '1', workDir: work });
  assert.deepEqual(downloads, [[RUN, `h3-terminal-cuda-reduced-${RUN}-1`], [RUN, ownName], [PEER_RUN, peerName]]);
  assert.deepEqual(loaded.decisionRun, { runId: RUN, attempt: '1', backend: 'cuda' });
  assert.equal(loaded.packets.mlx.runId, PEER_RUN); assert.equal(loaded.packets.mlx.runAttempt, '1');
  const record = await adoptPriorMatrixDecision(w.plan, { backend: 'cuda', ...loaded, caseIds: REMAINING, changedFiles: [] });
  assert.deepEqual(record.subsetCaseIds, REMAINING);
  assert.equal(peerPacketArtifactId(w.decisionPlan, 'mlx'), '13051908009');
  assert.throws(() => reducedEvidenceArtifact([{ name: `h3-terminal-cuda-reduced-${RUN}-1`, expired: true }], RUN, '1'), /expired/);
  assert.throws(() => reducedEvidenceArtifact([], RUN, '1'), /exactly one reduced case-evidence artifact/);
  const empty = path.join(w.root, 'no-decision'); await mkdir(path.join(empty, 'cuda.reduced.x'), { recursive: true });
  await assert.rejects(retainedDecisionPlan(empty, 'cuda'), /no retained cuda case plan carries a measured matrix decision/);
});
// Two segments of one backend: the prior run's passed cases at 4c3625898 and the subset run at a new SHA.
async function splitFixture(t, { duplicate = false, omit = null, retainPackets = true, toSubset = null, toPrior = null, pinChange = false, dropChangeRecord = false } = {}) {
  const w = await workspace(t), decided = { ...w.plan, sceneworks: PRIOR_SW, matrixDecision: w.decisionPlan.matrixDecision, matrixAnchors: w.decisionPlan.matrixAnchors };
  const subsetPlan = structuredClone(w.plan);
  if (pinChange) { subsetPlan.inference = NEW_PIN; subsetPlan.nativeEvidence.productPin = NEW_PIN; }
  await adopt({ ...w, plan: subsetPlan }, pinChange ? { inferenceAncestry: descends } : {});
  if (dropChangeRecord) delete subsetPlan.priorMatrixDecision.inferenceChange;
  const segments = [{ label: 'prior', caseRoot: path.join(w.root, 'seg-prior'), runId: RUN, attempt: '1' }, { label: 'subset', caseRoot: path.join(w.root, 'seg-subset'), runId: '99', attempt: '1' }];
  const write = async (segment, plan, id, runId) => {
    const dir = path.join(segment.caseRoot, id); await mkdir(dir, { recursive: true });
    await writeFile(path.join(dir, 'adapter-plan.json'), JSON.stringify(plan));
    await writeFile(path.join(dir, 'execution.json'), JSON.stringify({ caseId: id, status: 'passed', runId, runAttempt: '1', sceneworks: plan.sceneworks, inference: plan.inference, planSha256: executionPlanDigest(plan), verified: { files: [] } }));
  };
  for (const c of executionCases(decided).filter(c => c.backend === 'cuda' && !REMAINING.includes(c.id))) if (c.id !== toSubset) await write(segments[0], decided, c.id, RUN);
  for (const id of REMAINING) if (id !== omit && id !== toPrior) await write(segments[1], subsetPlan, id, '99');
  if (toSubset) await write(segments[1], subsetPlan, toSubset, '99');
  if (toPrior) await write(segments[0], decided, toPrior, RUN);
  if (duplicate) await write(segments[1], subsetPlan, 'cuda.reduced.real-drop-audio', '99');
  // The subset run retains its prior-decision anchor packets (output-relative) beside its cases.
  if (retainPackets) for (const b of ['cuda', 'mlx']) await cp(path.join(w.root, `anchors-${b}`), path.join(segments[1].caseRoot, `anchors-${b}`), { recursive: true });
  return { segments, verified: [] };
}
test('a split readback attributes each case to its own run and SHA, flags cross-segment controls, and reports a gap as a gap', async t => {
  const f = await splitFixture(t), verified = [];
  const verifyCase = async (plan, c) => { verified.push([c.id, plan.sceneworks]); };
  const controls = async (plan, backend, { caseDir }) => {
    assert.ok(caseDir({ id: 'cuda.reduced.real-hybrid-text' }).includes('seg-prior')); assert.ok(caseDir({ id: 'cuda.reduced.real-hybrid-text-repeat' }).includes('seg-subset'));
    return [{ caseId: 'cuda.reduced.real-hybrid-prompt-change', baselineCaseId: 'cuda.reduced.real-hybrid-text', repeatedCaseId: 'cuda.reduced.real-hybrid-text-repeat' }, { caseId: 'cuda.reduced.real-drop-images', baselineCaseId: 'cuda.reduced.artifact-0-ordered-mixed-baseline-ref-turbo', repeatedCaseId: 'cuda.reduced.real-mixed-repeat' }];
  };
  const r = await verifySplitBackendEvidence({ backend: 'cuda', segments: f.segments }, { verifyCase, controls });
  assert.equal(r.status, 'split-backend-content-reverified'); assert.equal(r.singleShaBackendReadback, false);
  // Roles come from the prior-decision record, so manifest order does not change the result.
  const reversed = await verifySplitBackendEvidence({ backend: 'cuda', segments: [...f.segments].reverse() }, { verifyCase: async () => {}, controls });
  assert.deepEqual(reversed, r);
  assert.deepEqual(r.segments.map(s => s.role), ['decision-run', 'prior-decision-subset']);
  assert.equal(verified.length, 25); assert.deepEqual(r.missing, []);
  assert.deepEqual(r.segments.map(s => [s.label, s.sceneworks, s.cases.length]), [['prior', [PRIOR_SW], 18], ['subset', [NEW_SW], 7]]);
  assert.deepEqual(r.crossCaseControls.map(v => v.crossSegment), [true, false]);
  assert.equal(r.priorMatrixDecision.decisionRun.runId, RUN);
  // The subset segment ran no native witnesses of its own: the reuse is reported and re-hashed.
  assert.deepEqual(r.reusedNativeWitnesses.map(v => [v.segment, v.backend, v.nativeEvidenceSha, v.reverifiedFromRetainedFiles]), [['subset', 'mlx', EVIDENCE, true], ['subset', 'cuda', EVIDENCE, true]]);
  const tampered = await splitFixture(t); await appendFile(path.join(tampered.segments[1].caseRoot, 'anchors-cuda', 'witness-0-0.log'), 'x');
  await assert.rejects(verifySplitBackendEvidence({ backend: 'cuda', segments: tampered.segments }, { verifyCase: async () => {}, controls }), /reused witness native-encoded-surface-cuda-\w+ retained file anchors-cuda\/witness-0-0\.log changed or is missing/);
  await assert.rejects(verifySplitBackendEvidence({ backend: 'cuda', segments: (await splitFixture(t, { retainPackets: false })).segments }, { verifyCase: async () => {}, controls }), /ENOENT/);
  const gap = await verifySplitBackendEvidence({ backend: 'cuda', segments: (await splitFixture(t, { omit: REMAINING[0] })).segments }, { verifyCase: async () => {}, controls });
  assert.equal(gap.status, 'split-backend-incomplete'); assert.deepEqual(gap.missing, [REMAINING[0]]); assert.equal(gap.crossCaseControls, null);
  await assert.rejects(verifySplitBackendEvidence({ backend: 'cuda', segments: (await splitFixture(t, { duplicate: true })).segments }, { verifyCase: async () => {}, controls }), /passed in more than one segment/);
});
test('a split readback across a recorded inference change reports each segment\'s and case\'s own pin; an undeclared change is refused', async t => {
  const controls = async () => [], seen = [];
  const r = await verifySplitBackendEvidence({ backend: 'cuda', segments: (await splitFixture(t, { pinChange: true })).segments }, { verifyCase: async (plan, c) => { seen.push([c.id, plan.inference]); }, controls });
  assert.equal(r.status, 'split-backend-content-reverified');
  assert.deepEqual(r.segments.map(s => [s.label, s.role, s.sceneworks, s.inference, s.cases.length]), [['prior', 'decision-run', [PRIOR_SW], [PIN], 18], ['subset', 'prior-decision-subset', [NEW_SW], [NEW_PIN], 7]]);
  assert.deepEqual(r.inferenceChange, { prior: PIN, own: NEW_PIN, descends: true, firstParentLog: FIRST_PARENT_LOG }); assert.equal(r.inference, NEW_PIN);
  for (const row of r.cases) assert.equal(row.inference, row.segment === 'subset' ? NEW_PIN : PIN, `${row.id} reports the pin it ran at`);
  // Each case was re-validated against its OWN plan's pin.
  assert.equal(seen.length, 25); for (const [id, pin] of seen) assert.equal(pin, REMAINING.includes(id) ? NEW_PIN : PIN);
  // Same-pin readback reports one pin per segment and no change.
  const same = await verifySplitBackendEvidence({ backend: 'cuda', segments: (await splitFixture(t)).segments }, { verifyCase: async () => {}, controls });
  assert.equal(same.inferenceChange, null); assert.deepEqual(same.segments.map(s => s.inference), [[PIN], [PIN]]);
  await assert.rejects(verifySplitBackendEvidence({ backend: 'cuda', segments: (await splitFixture(t, { pinChange: true, dropChangeRecord: true })).segments }, { verifyCase: async () => {}, controls }), /segments use different inference pins and the prior-decision record declares no inference change/);
});
test('a split readback is tied to the prior-decision record: its decision run, subset and prior-passed cases', async t => {
  const none = async () => {}, controls = async () => [];
  const other = await splitFixture(t); other.segments[0].runId = '41213935678';
  await assert.rejects(verifySplitBackendEvidence({ backend: 'cuda', segments: other.segments }, { verifyCase: none, controls }), /executed in run 41213935677, not segment prior/);
  // A segment whose cases genuinely ran in another run than the record's decision run is refused.
  const wrongRun = await splitFixture(t);
  const { readdir } = await import('node:fs/promises');
  for (const id of await readdir(wrongRun.segments[0].caseRoot)) {
    const file = path.join(wrongRun.segments[0].caseRoot, id, 'execution.json'), row = JSON.parse(await readFile(file, 'utf8'));
    await writeFile(file, JSON.stringify({ ...row, runId: '41213935678' }));
  }
  wrongRun.segments[0].runId = '41213935678';
  await assert.rejects(verifySplitBackendEvidence({ backend: 'cuda', segments: wrongRun.segments }, { verifyCase: none, controls }), /is run 41213935678, not the decision run 41213935677/);
  await assert.rejects(verifySplitBackendEvidence({ backend: 'cuda', segments: (await splitFixture(t, { toSubset: 'cuda.reduced.real-drop-audio' })).segments }, { verifyCase: none, controls }), /cuda\.reduced\.real-drop-audio ran in the subset segment but is not in its declared subsetCaseIds/);
  await assert.rejects(verifySplitBackendEvidence({ backend: 'cuda', segments: (await splitFixture(t, { toPrior: REMAINING[0] })).segments }, { verifyCase: none, controls }), /real-hybrid-first-last from the decision run is not among the record's priorPassedCaseIds/);
  const three = await splitFixture(t);
  await assert.rejects(verifySplitBackendEvidence({ backend: 'cuda', segments: [...three.segments, { ...three.segments[0], label: 'extra' }] }, { verifyCase: none, controls }), /exactly two run segments/);
});
test('the terminal workflow adopts a declared prior decision before resolving cases and then exchanges no anchors', async () => {
  const workflow = await readFile(new URL('../.github/workflows/h3-finetune-terminal.yml', import.meta.url), 'utf8');
  const adopt = workflow.indexOf('- name: Adopt the declared prior measured matrix decision'), resolve = workflow.indexOf('- name: Resolve the reduced operational case set for this backend');
  assert.ok(adopt > 0 && adopt < resolve, 'adoption runs after staging/bootstrap and before the case set is resolved');
  assert.match(workflow, /if: \$\{\{ fromJSON\(inputs\.campaign_request\)\.prior_matrix_decision != null \}\}/);
  assert.match(workflow, /&& fromJSON\(inputs\.campaign_request\)\.prior_matrix_decision == null \}\}/, 'the peer-liveness preflight is skipped: there is no peer');
  assert.match(workflow, /const record = await adoptPriorMatrixDecision\(plan, \{backend: env\.H3_STAGE_BACKEND/);
  assert.match(workflow, /execFileSync\("git", \["diff", "--name-only", `\$\{loaded\.decisionPlan\.sceneworks\}\.\.\$\{plan\.sceneworks\}`\]/);
  assert.match(workflow, /if \(plan\.priorMatrixDecision\) \{ ids = plan\.priorMatrixDecision\.subsetCaseIds;/);
  assert.match(workflow, /request\.adaptive_matrix!==false&&!plan\.priorMatrixDecision;/);
  assert.match(workflow, /path\.join\(env\.H3_STAGE_EVIDENCE, "prior-matrix-decision\.json"\)/, 'the record is retained with the always-uploaded stage evidence');
  assert.match(workflow, /\.\.\.Object\.values\(plan\.priorMatrixDecision\.anchorPackets\)\.map\(v=>path\.join\(plan\.output,v\.retainedDirectory\)\)/, 'the reused witnesses\' packets are retained with the case evidence');
  assert.match(workflow, /reusedFromPriorDecision:witnesses\.reuse\?\?null/, 'the lane logs a reused witness as reused');
  assert.match(workflow, /inferencePinAncestry\(\{git: inferenceGit, prior, own\}\)/, 'pin-change ancestry is read from the lane\'s own inference checkout');
  assert.match(workflow, /inferenceRoot = path\.join\(env\.GITHUB_WORKSPACE, "inference"\)/);
  assert.match(workflow, /changedFiles, inferenceAncestry, laneRun: lane\?\.laneRun \?\? null\}\)/);
});
test('explicit case ids are authoritative: refused without a prior decision instead of silently dropped', async () => {
  const workflow = await readFile(new URL('../.github/workflows/h3-finetune-terminal.yml', import.meta.url), 'utf8');
  assert.match(workflow, /if\(adaptive&&explicitSubset\)throw new Error\(/, 'the Execute step refuses an explicit subset the adaptive matrix would ignore');
  assert.match(workflow, /if\(Array\.isArray\(request\.case_ids\)&&request\.case_ids\.length>2\)throw new Error\("explicit case_ids cannot run under the adaptive matrix/, 'the peer preflight fails fast before staging');
  const { spawnSync } = await import('node:child_process');
  const script = fileURLToPath(new URL('../docs/testing/sc-26691-campaign/dispatch.sh', import.meta.url));
  const r = spawnSync('bash', [script, '--ack', 'a'.repeat(40)], { encoding: 'utf8', env: { ...process.env, CASE_IDS: 'cuda.reduced.real-hybrid-reference', PRIOR_MATRIX_RUN: '', DRY_RUN: '1' } });
  assert.equal(r.status, 2, r.stderr); assert.match(r.stderr, /REFUSING: CASE_IDS needs PRIOR_MATRIX_RUN/);
});

// sc-26691 (2026-11-01): the MLX lane continues ITS OWN run. Run 41331644188 (nax-macos-2, at ae148c29a,
// inference a43724b8, on cuda run 41213935677's decision) passed 15 of its 18 allocated cases, then
// refused numeric-pruned-artifact (device proof); numeric-w4a8-artifact and the lifecycle case never ran.
// The decision run is the cuda lane's, so the "already passed" rule must read the MLX lane's own run.
const LANE_RUN = '41331644188', LANE_SW = 'ae148c29ad533079cc5bdfde78c8d2ddd220ab83';
const MLX_PASSED = ['artifact-0-ordered-mixed-baseline-ref-turbo', 'artifact-1-reference', 'artifact-2-lokr-ref-turbo-baseline', 'artifact-3-opaque-renamed-linked-source', 'real-drop-audio', 'real-drop-lokr', 'real-drop-lora', 'real-hybrid-first-last', 'real-hybrid-image', 'real-hybrid-last-only', 'real-hybrid-reference', 'real-hybrid-text', 'real-independent-fl2va', 'real-lokr-repeat', 'real-mixed-repeat'].map(slug => `mlx.reduced.${slug}`);
const MLX_REMAINING = ['numeric-pruned-artifact', 'numeric-w4a8-artifact', 'lifecycle-generation-source-prep-cancel'].map(slug => `mlx.reduced.${slug}`);
async function laneFixture(t) {
  const w = await repinned(t), lanePlan = structuredClone(w.plan); lanePlan.sceneworks = LANE_SW;
  await adopt({ ...w, plan: lanePlan }, { backend: 'mlx', caseIds: null, inferenceAncestry: descends });
  const laneRoot = path.join(w.root, 'lane-evidence');
  const write = async (id, status, plan = lanePlan) => {
    const dir = path.join(laneRoot, id); await mkdir(dir, { recursive: true });
    await writeFile(path.join(dir, 'adapter-plan.json'), JSON.stringify(plan));
    await writeFile(path.join(dir, 'execution.json'), JSON.stringify({ caseId: id, status, runId: LANE_RUN, runAttempt: '1', sceneworks: LANE_SW, inference: NEW_PIN, planSha256: executionPlanDigest(plan), verified: { files: [] } }));
  };
  for (const id of MLX_PASSED) await write(id, 'passed');
  await write('mlx.reduced.numeric-pruned-artifact', 'failed');
  const laneRun = { runId: LANE_RUN, attempt: '1', evidenceRoot: laneRoot, plan: lanePlan };
  const continueLane = async (overrides = {}) => { const plan = structuredClone(w.plan); const record = await adopt({ ...w, plan }, { backend: 'mlx', caseIds: null, inferenceAncestry: descends, laneRun, ...overrides }); return { record, plan }; };
  return { w, lanePlan, laneRoot, laneRun, write, continueLane };
}
const withLaneRecord = (f, patch) => ({ laneRun: { ...f.laneRun, plan: { ...f.lanePlan, priorMatrixDecision: patch === undefined ? undefined : { ...f.lanePlan.priorMatrixDecision, ...patch } } } });
test('the MLX lane continues its own run 41331644188 on run 41213935677\'s decision: exactly the 3 cases it never passed', async t => {
  const f = await laneFixture(t), { record, plan } = await f.continueLane();
  assert.equal(executionCases(plan).filter(c => c.backend === 'mlx').length, 18, 'the MLX allocation on the prior decision is 18 cases');
  assert.deepEqual(record.subsetCaseIds, MLX_REMAINING);
  assert.deepEqual(record.priorPassedCaseIds, [], 'the cuda decision run carries none of the mlx lane\'s passes');
  assert.deepEqual(record.priorLaneRun, { runId: LANE_RUN, attempt: '1', sceneworks: LANE_SW, inference: NEW_PIN, passedCaseIds: record.allocation.filter(id => MLX_PASSED.includes(id)) });
  assert.equal(record.priorLaneRun.passedCaseIds.length, 15); assert.deepEqual(record.notCovered, []);
  assert.deepEqual(record.decisionRun, { runId: RUN, attempt: '1', backend: 'cuda' }); assert.equal(record.laneBackend, 'mlx');
  assert.match(record.meaning, /priorLaneRun is this lane's own earlier run/);
  await verifyMatrixAdmission(plan);
  // An explicit subset is the same three (the dispatch kit's CASE_IDS).
  assert.deepEqual((await f.continueLane({ caseIds: MLX_REMAINING })).record.subsetCaseIds, MLX_REMAINING);
});
test('a lane continuation refuses re-running its own passes and a lane run that is not this lane\'s run on this decision', async t => {
  const f = await laneFixture(t);
  await assert.rejects(f.continueLane({ caseIds: [...MLX_REMAINING, 'mlx.reduced.real-drop-audio'] }), /mlx\.reduced\.real-drop-audio already PASSED in the mlx lane's run 41331644188/);
  // Without naming its own run the lane cannot see those passes (the cuda decision run has none for it).
  const blind = structuredClone(f.w.plan);
  assert.equal((await adopt({ ...f.w, plan: blind }, { backend: 'mlx', caseIds: null, inferenceAncestry: descends })).subsetCaseIds.length, 18);
  await assert.rejects(f.continueLane({ laneRun: { ...f.laneRun, plan: { ...f.lanePlan, inference: PIN } } }), /prior lane run 41331644188 ran at inference 354b41cc\w+, not this lane's a43724b8\w+/);
  await assert.rejects(f.continueLane(withLaneRecord(f, { decisionRun: { runId: '1', attempt: '1', backend: 'cuda' } })), /adopted another decision run/);
  await assert.rejects(f.continueLane(withLaneRecord(f, { laneBackend: 'cuda' })), /is a cuda lane, not mlx/);
  await assert.rejects(f.continueLane(withLaneRecord(f, { priorLaneRun: { runId: '7' } })), /lane runs are not chained/);
  await assert.rejects(f.continueLane(withLaneRecord(f, undefined)), /did not run on a declared prior matrix decision/);
  // A same-backend decision run already carries the lane's passes: no separate lane run is named.
  await assert.rejects(adopt(await workspace(t), { laneRun: f.laneRun }), /is this cuda lane's own run/);
  // Each counted pass is bound to the plan it ran under.
  const tampered = await laneFixture(t); await tampered.write('mlx.reduced.real-drop-lora', 'passed', { ...tampered.lanePlan, cases: [] });
  const file = path.join(tampered.laneRoot, 'mlx.reduced.real-drop-lora', 'execution.json'), row = JSON.parse(await readFile(file, 'utf8'));
  await writeFile(file, JSON.stringify({ ...row, planSha256: executionPlanDigest(tampered.lanePlan) }));
  await assert.rejects(tampered.continueLane(), /real-drop-lora: prior lane run's retained adapter plan is not the plan the case ran under/);
});
async function laneSplit(t, { toLane = null } = {}) {
  const f = await laneFixture(t), { plan } = await f.continueLane();
  const segments = [{ label: 'mlx-41331644188', caseRoot: f.laneRoot, runId: LANE_RUN, attempt: '1' }, { label: 'mlx-subset', caseRoot: path.join(f.w.root, 'seg-subset'), runId: '99', attempt: '1' }];
  for (const id of MLX_REMAINING) {
    const dir = path.join(segments[1].caseRoot, id); await mkdir(dir, { recursive: true });
    await writeFile(path.join(dir, 'adapter-plan.json'), JSON.stringify(plan));
    await writeFile(path.join(dir, 'execution.json'), JSON.stringify({ caseId: id, status: 'passed', runId: '99', runAttempt: '1', sceneworks: plan.sceneworks, inference: plan.inference, planSha256: executionPlanDigest(plan), verified: { files: [] } }));
  }
  for (const b of ['cuda', 'mlx']) await cp(path.join(f.w.root, `anchors-${b}`), path.join(segments[1].caseRoot, `anchors-${b}`), { recursive: true });
  if (toLane) await f.write(toLane, 'passed');
  return { f, segments };
}
test('verify-split-backend combines the MLX lane\'s two runs: 41331644188\'s 15 passes and the continuation\'s 3, each at its own SHA', async t => {
  const { segments } = await laneSplit(t), seen = [], controls = async () => [];
  const r = await verifySplitBackendEvidence({ backend: 'mlx', segments }, { verifyCase: async (plan, c) => { seen.push([c.id, plan.sceneworks]); }, controls });
  assert.equal(r.status, 'split-backend-content-reverified'); assert.deepEqual(r.missing, []); assert.equal(r.singleShaBackendReadback, false);
  assert.deepEqual(r.segments.map(s => [s.label, s.role, s.runId, s.sceneworks, s.inference, s.cases.length]), [['mlx-41331644188', 'prior-lane-run', LANE_RUN, [LANE_SW], [NEW_PIN], 15], ['mlx-subset', 'prior-decision-subset', '99', [NEW_SW], [NEW_PIN], 3]]);
  assert.equal(seen.length, 18); assert.equal(r.priorMatrixDecision.priorLaneRun.runId, LANE_RUN); assert.equal(r.priorMatrixDecision.decisionRun.runId, RUN);
  assert.deepEqual(r.inferenceChange, { prior: PIN, own: NEW_PIN, descends: true, firstParentLog: FIRST_PARENT_LOG }, 'the decision\'s pin change is still reported; both MLX runs ran at the own pin');
  assert.deepEqual(await verifySplitBackendEvidence({ backend: 'mlx', segments: [...segments].reverse() }, { verifyCase: async () => {}, controls }), r, 'roles come from the records, not manifest order');
  // The lane-run segment must be the run the continuation names, with only the passes it recorded.
  const other = await laneSplit(t); other.segments[0].runId = '41331644189';
  await assert.rejects(verifySplitBackendEvidence({ backend: 'mlx', segments: other.segments }, { verifyCase: async () => {}, controls }), /executed in run 41331644188, not segment mlx-41331644188/);
  const late = await laneSplit(t, { toLane: 'mlx.reduced.numeric-pruned-artifact' });
  await assert.rejects(verifySplitBackendEvidence({ backend: 'mlx', segments: late.segments }, { verifyCase: async () => {}, controls }), /passed in more than one segment/);
  // Without a recorded lane run, two prior-decision segments are not combined.
  const { segments: plain } = await laneSplit(t);
  for (const id of MLX_REMAINING) {
    const file = path.join(plain[1].caseRoot, id, 'adapter-plan.json'), p = JSON.parse(await readFile(file, 'utf8')); delete p.priorMatrixDecision.priorLaneRun; await writeFile(file, JSON.stringify(p));
    const ex = path.join(plain[1].caseRoot, id, 'execution.json'), row = JSON.parse(await readFile(ex, 'utf8')); await writeFile(ex, JSON.stringify({ ...row, planSha256: executionPlanDigest(p) }));
  }
  await assert.rejects(verifySplitBackendEvidence({ backend: 'mlx', segments: plain }, { verifyCase: async () => {}, controls }), /exactly one must continue the other as its recorded priorLaneRun/);
});
test('the lane loader fetches the named run\'s own-backend evidence and the plans its cases ran under', async t => {
  const f = await laneFixture(t), work = path.join(f.w.root, 'fetched-lane'), downloads = [];
  const gh = args => {
    const route = args[1];
    if (route === `repos/SceneWorks/SceneWorks/actions/runs/${LANE_RUN}`) return JSON.stringify({ status: 'completed', repository: { full_name: 'SceneWorks/SceneWorks' } });
    if (route.startsWith(`repos/SceneWorks/SceneWorks/actions/runs/${LANE_RUN}/artifacts`)) return JSON.stringify({ artifacts: [{ name: `h3-terminal-mlx-reduced-${LANE_RUN}-1`, expired: false }, { name: `h3-terminal-stage-mlx-${LANE_SW}-${LANE_RUN}-1` }] });
    throw new Error(`unexpected gh ${args.join(' ')}`);
  };
  const download = async ({ runId, name, directory }) => { downloads.push([runId, name]); await cp(f.laneRoot, directory, { recursive: true }); };
  const loaded = await loadPriorLaneEvidence({ gh, download, repository: 'SceneWorks/SceneWorks', runId: LANE_RUN, attempt: '1', backend: 'mlx', workDir: work });
  assert.deepEqual(downloads, [[LANE_RUN, `h3-terminal-mlx-reduced-${LANE_RUN}-1`]]); assert.equal(loaded.evidenceArtifact, `h3-terminal-mlx-reduced-${LANE_RUN}-1`);
  assert.deepEqual(loaded.laneRun.plan.priorMatrixDecision, f.lanePlan.priorMatrixDecision);
  const plan = structuredClone(f.w.plan);
  assert.deepEqual((await adopt({ ...f.w, plan }, { backend: 'mlx', caseIds: null, inferenceAncestry: descends, laneRun: loaded.laneRun })).subsetCaseIds, MLX_REMAINING);
  await assert.rejects(loadPriorLaneEvidence({ gh, download, repository: 'SceneWorks/SceneWorks', runId: LANE_RUN, attempt: '1', backend: 'cuda', workDir: path.join(f.w.root, 'x') }), /retained mlx case evidence, not this cuda lane's/);
  const mixed = path.join(f.w.root, 'mixed'); await cp(f.laneRoot, mixed, { recursive: true });
  await writeFile(path.join(mixed, 'mlx.reduced.real-drop-lora', 'adapter-plan.json'), JSON.stringify({ ...f.lanePlan, priorMatrixDecision: undefined }));
  await assert.rejects(retainedLanePlan(mixed, 'mlx'), /did not run on a declared prior matrix decision/);
});
test('the dispatch kit stamps prior_lane_run only for one lane on a declared prior decision', async () => {
  const { spawnSync } = await import('node:child_process');
  const script = fileURLToPath(new URL('../docs/testing/sc-26691-campaign/dispatch.sh', import.meta.url));
  const run = env => spawnSync('bash', [script, '--ack', 'a'.repeat(40)], { encoding: 'utf8', env: { ...process.env, DRY_RUN: '1', CASE_IDS: '', ...env } });
  let r = run({ PRIOR_LANE_RUN: LANE_RUN, PRIOR_MATRIX_RUN: '', BACKENDS: 'mlx' }); assert.equal(r.status, 2, r.stderr); assert.match(r.stderr, /PRIOR_LANE_RUN continues a lane on a declared prior decision; it needs PRIOR_MATRIX_RUN/);
  r = run({ PRIOR_LANE_RUN: LANE_RUN, PRIOR_MATRIX_RUN: `${RUN}/1`, BACKENDS: 'mlx cuda' }); assert.equal(r.status, 2, r.stderr); assert.match(r.stderr, /set BACKENDS to that one backend/);
  r = run({ PRIOR_LANE_RUN: 'x', PRIOR_MATRIX_RUN: `${RUN}/1`, BACKENDS: 'mlx' }); assert.equal(r.status, 2, r.stderr); assert.match(r.stderr, /PRIOR_LANE_RUN must be <run_id>\[\/<attempt>\]/);
  const kit = await readFile(script, 'utf8');
  assert.match(kit, /if \(laneRun\) p\.prior_lane_run = \{ run_id: Number\(laneRun\), attempt: Number\(laneAttempt\) \};/);
  assert.match(kit, /\(laneRun && !same\(back\.prior_lane_run, \{ run_id: Number\(laneRun\), attempt: Number\(laneAttempt\) \}\)\)/, 'the stamped lane run round-trips');
  const workflow = await readFile(new URL('../.github/workflows/h3-finetune-terminal.yml', import.meta.url), 'utf8');
  assert.match(workflow, /await loadPriorLaneEvidence\(\{gh, download, repository: env\.GITHUB_REPOSITORY, runId: String\(declaredLane\.run_id\), attempt: String\(declaredLane\.attempt\), backend: env\.H3_STAGE_BACKEND, workDir\}\)/);
  assert.match(workflow, /laneRun: lane\?\.laneRun \?\? null\}\);/);
  assert.match(workflow, /if\(request\.prior_lane_run!=null\)throw new Error\("prior_lane_run continues a lane on a declared prior decision; it needs prior_matrix_decision"\);/);
});
