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
