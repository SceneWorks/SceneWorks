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
import { loadPriorDecisionEvidence, peerPacketArtifactId, reducedEvidenceArtifact, retainedDecisionPlan } from './lib/h3-prior-decision.mjs';

const FIXTURE = fileURLToPath(new URL('./fixtures/sc26691-prior-decision/', import.meta.url));
const PRIOR_SW = '4c3625898b20859f57ca38986ed80b1c427cdfdb', PIN = '354b41ccaccdc14afccff0d0fb1ae1b0be04836e', EVIDENCE = '0bada8f7287f414c30f77eb1d29ef4a544eb8ffc';
const NEW_SW = 'b'.repeat(40), RUN = '41213935677', PEER_RUN = '41213914294';
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
  assert.deepEqual(record.anchorPackets, { mlx: { runId: PEER_RUN, runAttempt: '1' }, cuda: { runId: RUN, runAttempt: '1' } });
  assert.equal(record.decision.selectedBackend, 'cuda'); assert.equal(record.decision.displacedBackend, 'mlx');
  assert.deepEqual(record.decision, w.decisionPlan.matrixDecision, 'recomputed from the retained anchors, equal to the retained decision');
  assert.equal(record.allocation.length, 25); assert.equal(record.priorPassedCaseIds.length, 18);
  assert.deepEqual(record.subsetCaseIds, REMAINING); assert.deepEqual(record.notCovered, []);
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
});
test('a prior-decision subset refuses cases that already passed, cases outside the allocation and drifted evidence', async t => {
  await assert.rejects(adopt(await workspace(t), { caseIds: ['cuda.reduced.real-drop-audio'] }), /already PASSED in run 41213935677/);
  await assert.rejects(adopt(await workspace(t), { backend: 'mlx', caseIds: ['mlx.reduced.real-drop-images'] }), /not in the mlx lane's allocation/);
  await assert.rejects(adopt(await workspace(t), { changedFiles: undefined }), /git difference/);
  const pin = await workspace(t); pin.plan.inference = 'c'.repeat(40);
  await assert.rejects(adopt(pin), /share this lane's inference pin/);
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
async function splitFixture(t, { duplicate = false, omit = null } = {}) {
  const w = await workspace(t), decided = { ...w.plan, sceneworks: PRIOR_SW, matrixDecision: w.decisionPlan.matrixDecision, matrixAnchors: w.decisionPlan.matrixAnchors };
  const subsetPlan = structuredClone(w.plan); await adopt({ ...w, plan: subsetPlan });
  const segments = [{ label: 'prior', caseRoot: path.join(w.root, 'seg-prior'), runId: RUN, attempt: '1' }, { label: 'subset', caseRoot: path.join(w.root, 'seg-subset'), runId: '99', attempt: '1' }];
  const write = async (segment, plan, id, runId) => {
    const dir = path.join(segment.caseRoot, id); await mkdir(dir, { recursive: true });
    await writeFile(path.join(dir, 'adapter-plan.json'), JSON.stringify(plan));
    await writeFile(path.join(dir, 'execution.json'), JSON.stringify({ caseId: id, status: 'passed', runId, runAttempt: '1', sceneworks: plan.sceneworks, inference: PIN, planSha256: executionPlanDigest(plan), verified: { files: [] } }));
  };
  for (const c of executionCases(decided).filter(c => c.backend === 'cuda' && !REMAINING.includes(c.id))) await write(segments[0], decided, c.id, RUN);
  for (const id of REMAINING) if (id !== omit) await write(segments[1], subsetPlan, id, '99');
  if (duplicate) await write(segments[1], subsetPlan, 'cuda.reduced.real-drop-audio', '99');
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
  assert.equal(verified.length, 25); assert.deepEqual(r.missing, []);
  assert.deepEqual(r.segments.map(s => [s.label, s.sceneworks, s.cases.length]), [['prior', [PRIOR_SW], 18], ['subset', [NEW_SW], 7]]);
  assert.deepEqual(r.crossCaseControls.map(v => v.crossSegment), [true, false]);
  assert.equal(r.priorMatrixDecision.decisionRun.runId, RUN);
  const gap = await verifySplitBackendEvidence({ backend: 'cuda', segments: (await splitFixture(t, { omit: REMAINING[0] })).segments }, { verifyCase: async () => {}, controls });
  assert.equal(gap.status, 'split-backend-incomplete'); assert.deepEqual(gap.missing, [REMAINING[0]]); assert.equal(gap.crossCaseControls, null);
  await assert.rejects(verifySplitBackendEvidence({ backend: 'cuda', segments: (await splitFixture(t, { duplicate: true })).segments }, { verifyCase: async () => {}, controls }), /passed in more than one segment/);
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
});
