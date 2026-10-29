// sc-26691: the adaptive matrix pairs this lane with the OTHER backend's terminal job, dispatched as
// a separate `server-candle-linux.yml` run at the same SceneWorks SHA. A lane whose peer was never
// dispatched, or has already ended without publishing its anchors, used to wait the full bounded
// six hours for metadata that could not arrive. This classifies the peer from the Actions API so
// such a misconfigured dispatch fails within minutes, not hours.
//
// Shapes are the real API rows of runs 40855384972 (CUDA) and 40855343040 (MLX): the reusable
// job is named `h3-terminal / terminal`, and its requested labels name the lane even while queued.

export const PEER_JOB_NAME = 'h3-terminal / terminal';
/** How long a lane may wait for its peer to be dispatched at all before failing. */
export const PEER_APPEARANCE_GRACE_MS = 10 * 60 * 1000;
const ACTIVE = new Set(['queued', 'in_progress', 'waiting', 'requested', 'pending']);

/** Whether an Actions job row is the given backend's H3 terminal job. */
export function isPeerTerminalJob(job, backend) {
  if (job?.name !== PEER_JOB_NAME) return false;
  const labels = (job.labels ?? []).map(label => String(label).toLowerCase());
  if (!labels.includes('real-weights')) return false;
  return backend === 'mlx' ? labels.includes('macos') : labels.includes('windows') && labels.includes('cuda');
}

/**
 * Classify the peer lane from the workflow's dispatch runs at this SHA and their jobs.
 * Returns `alive` (a peer terminal job is queued/running, or a run has not created its jobs yet),
 * `succeeded` (a peer ended successfully; its anchors should exist), `ended` (every peer ended
 * without success) or `absent` (no peer was dispatched).
 */
export function classifyPeerLane({ runs, jobsByRun, selfRunId, headSha, peerBackend }) {
  let pendingRun = null, succeeded = null;
  const ended = [];
  for (const run of runs ?? []) {
    if (String(run.id) === String(selfRunId) || run.head_sha !== headSha || run.event !== 'workflow_dispatch') continue;
    const jobs = jobsByRun.get(String(run.id)) ?? [];
    const peers = jobs.filter(job => isPeerTerminalJob(job, peerBackend));
    for (const job of peers) {
      if (ACTIVE.has(job.status)) return { state: 'alive', run, job };
      if (job.conclusion === 'success') succeeded ??= { run, job };
      else ended.push({ run, job });
    }
    // A dispatch whose jobs are not created yet cannot be told apart by backend; it counts as a
    // possible peer only while the run itself is still active.
    if (!jobs.length && ACTIVE.has(run.status)) pendingRun ??= run;
  }
  if (pendingRun) return { state: 'alive', run: pendingRun, job: null };
  if (succeeded) return { state: 'succeeded', ...succeeded };
  if (ended.length) return { state: 'ended', run: ended[0].run, job: ended[0].job, ended };
  return { state: 'absent' };
}

/** Throw the fast-fail refusal for a peer that cannot deliver anchors, or return the classification. */
// `anchorsPublished` is false only when the caller has just searched for the peer's anchors and found
// none; before that search (the early preflight) a succeeded peer may well have published them.
export function assertPeerCanDeliver(classification, { peerBackend, now, startedAt, graceMs = PEER_APPEARANCE_GRACE_MS, anchorsPublished }) {
  const { state, run, job } = classification;
  if (state === 'alive' || anchorsPublished === true) return classification;
  if (state === 'succeeded' && anchorsPublished !== false) return classification;
  // A finished peer at this SHA may be a stale earlier dispatch while the real one is seconds away.
  if (now - startedAt < graceMs) return classification;
  if (state === 'succeeded') throw new Error(`adaptive_matrix: the paired ${peerBackend} terminal run ${run.id} finished successfully without publishing inline anchors; it did not run the adaptive matrix, so this lane cannot pair with it. Dispatch both lanes with adaptive_matrix:true, or this lane with adaptive_matrix:false.`);
  if (state === 'ended') throw new Error(`adaptive_matrix: the paired ${peerBackend} terminal run ${run.id} (job ${job.id}) already ended as ${job.conclusion} without publishing inline anchors; refusing to wait for anchors that cannot arrive. Re-dispatch the pair, or this lane with adaptive_matrix:false.`);
  throw new Error(`adaptive_matrix: no paired ${peerBackend} terminal run was dispatched at this SceneWorks SHA within ${Math.round(graceMs / 60000)} minutes; refusing to wait for anchors that cannot arrive. Dispatch the ${peerBackend} lane, or this lane with adaptive_matrix:false.`);
}

/** The peer's published inline-anchor artifact (earliest, unexpired, another run at this SHA), or null. */
export function findPeerAnchors({ gh, repository, name, headSha, selfRunId }) {
  const pages = JSON.parse(gh(['api', '--paginate', '--slurp', `repos/${repository}/actions/artifacts?name=${encodeURIComponent(name)}&per_page=100`]));
  return pages.flatMap(page => page.artifacts ?? []).filter(a => a.name === name && !a.expired && a.workflow_run?.head_sha === headSha && String(a.workflow_run.id) !== String(selfRunId)).sort((a, b) => Date.parse(a.created_at) - Date.parse(b.created_at))[0] ?? null;
}

/**
 * Wait (bounded) until the peer either published its anchors or is alive; throw as soon as it is
 * known it never can. `sleep` and `now` are injected so the policy is testable without waiting.
 */
export async function awaitPeerLane({ gh, repository, headSha, selfRunId, peerBackend, anchorName, startedAt, now = () => Date.now(), sleep = ms => new Promise(resolve => setTimeout(resolve, ms)), pollMs = 20000, graceMs = PEER_APPEARANCE_GRACE_MS }) {
  for (;;) {
    const anchors = findPeerAnchors({ gh, repository, name: anchorName, headSha, selfRunId });
    if (anchors) return { state: 'anchors', anchors };
    const classification = assertPeerCanDeliver(observePeerLane({ gh, repository, headSha, selfRunId, peerBackend }), { peerBackend, now: now(), startedAt, graceMs, anchorsPublished: false });
    if (classification.state === 'alive') return classification;
    await sleep(pollMs);
  }
}

/** Read the dispatch runs at `headSha` and their jobs through `gh api` (an injected executor). */
export function observePeerLane({ gh, repository, workflowPath = '.github/workflows/server-candle-linux.yml', headSha, selfRunId, peerBackend }) {
  const workflow = encodeURIComponent(workflowPath.split('/').pop());
  const runs = JSON.parse(gh(['api', '--paginate', '--slurp', `repos/${repository}/actions/workflows/${workflow}/runs?event=workflow_dispatch&head_sha=${headSha}&per_page=100`]))
    .flatMap(page => page.workflow_runs ?? []);
  const jobsByRun = new Map();
  for (const run of runs) {
    if (String(run.id) === String(selfRunId) || run.head_sha !== headSha) continue;
    jobsByRun.set(String(run.id), JSON.parse(gh(['api', '--paginate', '--slurp', `repos/${repository}/actions/runs/${run.id}/attempts/${run.run_attempt}/jobs?per_page=100`])).flatMap(page => page.jobs ?? []));
  }
  return classifyPeerLane({ runs, jobsByRun, selfRunId, headSha, peerBackend });
}
