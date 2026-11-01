// sc-26691 (2026-11-01): locate and download the retained evidence of a DECLARED prior measured
// matrix decision, so a lane can run an explicit case subset at a new SceneWorks SHA without a new
// peer anchor exchange. Everything is named by the request (`prior_matrix_decision: {run_id,
// attempt}`) or derived from that run's own retained files; nothing is searched for by recency.
//
// From the named run: its reduced case evidence (`h3-terminal-<backend>-reduced-<run>-<attempt>`),
// whose adapter plans carry the decision and the anchors it was measured from; its OWN anchor packet
// (`h3-anchors-<backend>-<sw>-<pin>-<evidence>-attempt-<attempt>`); and the PEER packet that run
// actually received, identified by the artifact id recorded in its anchors' receipt paths
// (`peer-inline-anchors-<peer>-<artifact id>-...`). Validation of the content is
// adoptPriorMatrixDecision's job (scripts/h3-finetune-terminal.mjs); this module only fetches.
import assert from 'node:assert/strict';
import path from 'node:path';
import { readdir, readFile } from 'node:fs/promises';
import { anchorArtifactBase, anchorArtifactName } from './h3-peer-lane.mjs';

const PEER_PACKET_PATH = /[\\/]peer-inline-anchors-(mlx|cuda)-(\d+)-[^\\/]+[\\/][^\\/]+$/;

/** The run's one retained reduced-evidence artifact, and the backend it belongs to. */
export function reducedEvidenceArtifact(artifacts, runId, attempt) {
  const pattern = new RegExp(`^h3-terminal-(mlx|cuda)-reduced-${runId}-${attempt}$`);
  const matches = (artifacts ?? []).filter(artifact => pattern.test(artifact.name));
  assert.equal(matches.length, 1, `run ${runId} attempt ${attempt} must retain exactly one reduced case-evidence artifact (found ${matches.length})`);
  assert.notEqual(matches[0].expired, true, `${matches[0].name} has expired; the prior decision can no longer be re-validated`);
  return { artifact: matches[0], backend: pattern.exec(matches[0].name)[1] };
}

/** The decision-carrying adapter plan from the retained case evidence; every such plan must agree. */
export async function retainedDecisionPlan(evidenceRoot, backend) {
  const plans = [];
  for (const entry of await readdir(evidenceRoot, { withFileTypes: true })) {
    if (!entry.isDirectory() || !entry.name.startsWith(`${backend}.`)) continue;
    let plan;
    try { plan = JSON.parse(await readFile(path.join(evidenceRoot, entry.name, 'adapter-plan.json'), 'utf8')); } catch (error) { if (error.code === 'ENOENT') continue; throw error; }
    if (plan.matrixDecision) plans.push({ caseId: entry.name, plan });
  }
  assert.ok(plans.length > 0, `no retained ${backend} case plan carries a measured matrix decision`);
  for (const { caseId, plan } of plans) {
    assert.deepEqual(plan.matrixDecision, plans[0].plan.matrixDecision, `${caseId} ran under another matrix decision`);
    assert.deepEqual(plan.matrixAnchors, plans[0].plan.matrixAnchors, `${caseId} ran under other anchors`);
  }
  return plans[0].plan;
}

/**
 * Ancestry of an inference pin change, read from the lane's own inference checkout (full history).
 * `git(args)` runs git in that checkout and returns stdout, throwing an error carrying `.status` on
 * a non-zero exit. `descends` is true only when `prior` is an ancestor of `own`
 * (`git merge-base --is-ancestor prior own` exits 0); exit 1 is "not an ancestor", and any other
 * failure (an unknown commit) is reported as such, never as descent. `firstParentLog` is the
 * one-line `<sha> <subject>` list of `git log --first-parent prior..own`, recorded verbatim.
 */
export function inferencePinAncestry({ git, prior, own }) {
  for (const [name, sha] of [['prior', prior], ['own', own]]) assert.match(String(sha ?? ''), /^[0-9a-f]{40}$/, `${name} inference pin must be a 40-hex SHA`);
  assert.notEqual(prior, own, 'an inference pin change needs two different pins');
  try { git(['merge-base', '--is-ancestor', prior, own]); } catch (error) {
    if (error.status === 1) return { prior, own, descends: false, firstParentLog: [] };
    throw new Error(`cannot establish inference ancestry ${prior}..${own} in the lane's inference checkout: ${error.message}`);
  }
  const firstParentLog = String(git(['log', '--first-parent', '--format=%H %s', `${prior}..${own}`])).split('\n').map(line => line.trimEnd()).filter(Boolean);
  assert.ok(firstParentLog.length > 0, `inference ${own} descends from ${prior} but the first-parent log between them is empty`);
  return { prior, own, descends: true, firstParentLog };
}

/** The artifact id of the peer packet the decision run received, from its anchors' receipt paths. */
export function peerPacketArtifactId(decisionPlan, peer) {
  const ids = new Set();
  for (const anchor of decisionPlan.matrixAnchors ?? []) {
    if (anchor.backend !== peer) continue;
    const match = PEER_PACKET_PATH.exec(String(anchor.receiptInput?.path ?? ''));
    assert.ok(match && match[1] === peer, `the ${peer} anchor ${anchor.caseId} does not name the peer packet it was received in`);
    ids.add(match[2]);
  }
  assert.equal(ids.size, 1, `the decision must name exactly one received ${peer} anchor packet (found ${ids.size})`);
  return [...ids][0];
}

/**
 * Fetch the declared prior decision's evidence into `workDir` (which must be under the run's own
 * output root). `gh(args)` returns stdout; `download({runId, name, directory})` fetches one artifact.
 */
export async function loadPriorDecisionEvidence({ gh, download, repository, runId, attempt, workDir }) {
  assert.match(String(runId ?? ''), /^\d+$/, 'prior_matrix_decision.run_id must be a numeric run id');
  assert.match(String(attempt ?? ''), /^\d+$/, 'prior_matrix_decision.attempt must be a numeric attempt');
  const run = JSON.parse(gh(['api', `repos/${repository}/actions/runs/${runId}`]));
  assert.equal(run.repository?.full_name, repository, `run ${runId} is not a run of ${repository}`);
  assert.equal(run.status, 'completed', `run ${runId} is ${run.status}; only a completed run's decision is adopted`);
  const artifacts = JSON.parse(gh(['api', `repos/${repository}/actions/runs/${runId}/artifacts?per_page=100`])).artifacts;
  const { artifact: reduced, backend } = reducedEvidenceArtifact(artifacts, runId, attempt);
  const priorEvidenceRoot = path.join(workDir, 'evidence');
  await download({ runId: String(runId), name: reduced.name, directory: priorEvidenceRoot });
  const decisionPlan = await retainedDecisionPlan(priorEvidenceRoot, backend);
  const identity = { sceneworks: decisionPlan.sceneworks, inference: decisionPlan.inference, evidence: decisionPlan.nativeEvidence?.sha };
  const ownName = anchorArtifactName(anchorArtifactBase({ backend, ...identity }), attempt);
  const own = artifacts.find(artifact => artifact.name === ownName);
  assert.ok(own, `run ${runId} retains no ${ownName}`); assert.notEqual(own.expired, true, `${ownName} has expired`);
  const peer = backend === 'mlx' ? 'cuda' : 'mlx', peerId = peerPacketArtifactId(decisionPlan, peer);
  const peerArtifact = JSON.parse(gh(['api', `repos/${repository}/actions/artifacts/${peerId}`]));
  assert.notEqual(peerArtifact.expired, true, `the received ${peer} anchor artifact ${peerId} has expired`);
  const peerAttempt = /-attempt-(\d+)$/.exec(peerArtifact.name ?? '')?.[1];
  assert.ok(peerAttempt, `artifact ${peerId} is not a per-attempt anchor packet`);
  assert.equal(peerArtifact.name, anchorArtifactName(anchorArtifactBase({ backend: peer, ...identity }), peerAttempt), `artifact ${peerId} is not the ${peer} packet for the decision's source`);
  const packets = {
    [backend]: { directory: path.join(workDir, `anchors-${backend}`), runId: String(runId), runAttempt: String(attempt), artifact: ownName },
    [peer]: { directory: path.join(workDir, `anchors-${peer}`), runId: String(peerArtifact.workflow_run?.id), runAttempt: peerAttempt, artifact: peerArtifact.name },
  };
  assert.match(packets[peer].runId, /^\d+$/, `artifact ${peerId} names no workflow run`);
  await download({ runId: packets[backend].runId, name: ownName, directory: packets[backend].directory });
  await download({ runId: packets[peer].runId, name: peerArtifact.name, directory: packets[peer].directory });
  return { decisionRun: { runId: String(runId), attempt: String(attempt), backend }, decisionPlan, packets, priorEvidenceRoot, evidenceArtifact: reduced.name };
}

/**
 * This lane's own earlier run on the same declared decision (request `prior_lane_run: {run_id,
 * attempt}`), when the decision run belongs to the OTHER backend: e.g. the MLX lane's run 41331644188
 * on run 41213935677's cuda decision. Its retained reduced evidence carries the adapter plans its
 * cases ran under (each with the prior-decision record it adopted) and their execution rows; which of
 * its cases passed, and whether it adopted the same decision, is adoptPriorMatrixDecision's check.
 */
export async function retainedLanePlan(evidenceRoot, backend) {
  const plans = [];
  for (const entry of await readdir(evidenceRoot, { withFileTypes: true })) {
    if (!entry.isDirectory() || !entry.name.startsWith(`${backend}.`)) continue;
    let plan;
    try { plan = JSON.parse(await readFile(path.join(evidenceRoot, entry.name, 'adapter-plan.json'), 'utf8')); } catch (error) { if (error.code === 'ENOENT') continue; throw error; }
    plans.push({ caseId: entry.name, plan });
  }
  assert.ok(plans.length > 0, `no retained ${backend} case plan in the prior lane run`);
  for (const { caseId, plan } of plans) {
    assert.ok(plan.priorMatrixDecision, `${caseId} in the prior lane run did not run on a declared prior matrix decision`);
    assert.deepEqual(plan.priorMatrixDecision, plans[0].plan.priorMatrixDecision, `${caseId} ran under another prior-decision record`);
  }
  return plans[0].plan;
}
export async function loadPriorLaneEvidence({ gh, download, repository, runId, attempt, backend, workDir }) {
  assert.match(String(runId ?? ''), /^\d+$/, 'prior_lane_run.run_id must be a numeric run id');
  assert.match(String(attempt ?? ''), /^\d+$/, 'prior_lane_run.attempt must be a numeric attempt');
  const run = JSON.parse(gh(['api', `repos/${repository}/actions/runs/${runId}`]));
  assert.equal(run.repository?.full_name, repository, `run ${runId} is not a run of ${repository}`);
  assert.equal(run.status, 'completed', `prior lane run ${runId} is ${run.status}; only a completed run's passes are counted`);
  const artifacts = JSON.parse(gh(['api', `repos/${repository}/actions/runs/${runId}/artifacts?per_page=100`])).artifacts;
  const { artifact, backend: retained } = reducedEvidenceArtifact(artifacts, runId, attempt);
  assert.equal(retained, backend, `prior lane run ${runId} retained ${retained} case evidence, not this ${backend} lane's`);
  const evidenceRoot = path.join(workDir, 'lane-evidence');
  await download({ runId: String(runId), name: artifact.name, directory: evidenceRoot });
  return { laneRun: { runId: String(runId), attempt: String(attempt), evidenceRoot, plan: await retainedLanePlan(evidenceRoot, backend) }, evidenceArtifact: artifact.name };
}
