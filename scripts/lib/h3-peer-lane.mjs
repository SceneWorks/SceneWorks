// sc-26691: the adaptive matrix pairs this lane with the OTHER backend's terminal job, dispatched as
// a separate `server-candle-linux.yml` run at the same SceneWorks SHA. A lane whose peer was never
// dispatched, or has already ended without publishing its anchors, used to wait out a fixed bound
// for metadata that could not arrive. This classifies the peer from the Actions API so
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

// ---------------------------------------------------------------------------
// Inline-anchor artifacts. One per run ATTEMPT: a fixed per-SHA name made a rerun attempt's upload
// collide (409) with its earlier attempt's artifact in the same run, and the earliest-created pick
// paired with whichever upload happened first. The consumer takes the newest one that validates.
export const DISPATCH_WORKFLOW_PATH = '.github/workflows/server-candle-linux.yml';
/** How often the anchor wait polls; each poll lists the dispatch runs, their artifacts and jobs. */
export const ANCHOR_POLL_MS = 60 * 1000;

/** The pin-bound part of a lane's inline-anchor artifact name. */
export function anchorArtifactBase({ backend, sceneworks, inference, evidence }) {
  for (const value of [backend, sceneworks, inference, evidence]) if (typeof value !== 'string' || !value) throw new Error('anchor artifact identity requires backend and all three SHAs');
  return `h3-anchors-${backend}-${sceneworks}-${inference}-${evidence}`;
}
/** The artifact name one run attempt publishes (mirrors the workflow's upload name). */
export function anchorArtifactName(base, runAttempt) {
  const attempt = Number(runAttempt);
  if (!Number.isSafeInteger(attempt) || attempt < 1) throw new Error(`run attempt must be a positive integer, not ${runAttempt}`);
  return `${base}-attempt-${attempt}`;
}
/** The run attempt an artifact name carries for `base`, or null when it is not one of `base`'s. */
export function anchorArtifactAttempt(name, base) {
  const prefix = `${base}-attempt-`;
  if (typeof name !== 'string' || !name.startsWith(prefix)) return null;
  const text = name.slice(prefix.length);
  return /^[1-9]\d*$/.test(text) ? Number(text) : null;
}

/** The workflow_dispatch runs of the dispatch workflow at `headSha`. */
export function listDispatchRuns({ gh, repository, workflowPath = DISPATCH_WORKFLOW_PATH, headSha }) {
  const workflow = encodeURIComponent(workflowPath.split('/').pop());
  return JSON.parse(gh(['api', '--paginate', '--slurp', `repos/${repository}/actions/workflows/${workflow}/runs?event=workflow_dispatch&head_sha=${headSha}&per_page=100`]))
    .flatMap(page => page.workflow_runs ?? []);
}

/**
 * Every unexpired inline-anchor artifact for `name` (the base) published by another dispatch run at
 * this SHA, newest first (creation time, then attempt), each annotated with its `runAttempt`.
 */
export function peerAnchorCandidates({ gh, repository, name, headSha, selfRunId, runs }) {
  const candidates = [];
  for (const run of runs ?? listDispatchRuns({ gh, repository, headSha })) {
    if (String(run.id) === String(selfRunId) || run.head_sha !== headSha) continue;
    const artifacts = JSON.parse(gh(['api', '--paginate', '--slurp', `repos/${repository}/actions/runs/${run.id}/artifacts?per_page=100`])).flatMap(page => page.artifacts ?? []);
    for (const artifact of artifacts) {
      const runAttempt = anchorArtifactAttempt(artifact.name, name);
      if (runAttempt === null || artifact.expired || artifact.workflow_run?.head_sha !== headSha || String(artifact.workflow_run?.id) !== String(run.id)) continue;
      candidates.push({ ...artifact, runAttempt });
    }
  }
  return candidates.sort((a, b) => Date.parse(b.created_at) - Date.parse(a.created_at) || b.runAttempt - a.runAttempt);
}

/** The newest published peer inline-anchor artifact at this SHA, or null. */
export function findPeerAnchors(options) {
  return peerAnchorCandidates(options)[0] ?? null;
}

/**
 * Wait (bounded) until the peer either published its anchors or is alive; throw as soon as it is
 * known it never can. `sleep` and `now` are injected so the policy is testable without waiting.
 */
export async function awaitPeerLane({ gh, repository, headSha, selfRunId, peerBackend, anchorName, startedAt, now = () => Date.now(), sleep = ms => new Promise(resolve => setTimeout(resolve, ms)), pollMs = 20000, graceMs = PEER_APPEARANCE_GRACE_MS }) {
  for (;;) {
    const runs = listDispatchRuns({ gh, repository, headSha });
    const anchors = findPeerAnchors({ gh, repository, name: anchorName, headSha, selfRunId, runs });
    if (anchors) return { state: 'anchors', anchors };
    const classification = assertPeerCanDeliver(observePeerLane({ gh, repository, headSha, selfRunId, peerBackend, runs }), { peerBackend, now: now(), startedAt, graceMs, anchorsPublished: false });
    if (classification.state === 'alive') return classification;
    await sleep(pollMs);
  }
}

/**
 * The adaptive matrix's anchor wait. It lasts exactly as long as the peer can still deliver: it
 * returns the newest published anchors that `validate` accepts, throws as soon as the peer lane is
 * observed ended, failed or absent without valid anchors, and is otherwise bounded only by this
 * lane's own budget `deadline`. (It used to stop at a fixed six hours that was never sized against
 * a measured peer duration, discarding a live peer's work.) A candidate `validate` rejects is
 * never retried; the next-newest one is tried instead.
 */
export async function waitForPeerAnchors({ gh, repository, name, headSha, selfRunId, peerBackend, startedAt, deadline, validate, now = () => Date.now(), sleep = ms => new Promise(resolve => setTimeout(resolve, ms)), pollMs = ANCHOR_POLL_MS, graceMs = PEER_APPEARANCE_GRACE_MS, log = () => {} }) {
  if (!Number.isFinite(deadline)) throw new Error('the anchor wait requires this lane\'s budget deadline');
  const rejected = new Map();
  const rejections = () => rejected.size ? ` Rejected anchors: ${[...rejected].map(([id, reason]) => `${id}: ${reason}`).join('; ')}.` : '';
  for (let polls = 0; ; polls++) {
    if (now() >= deadline) throw new Error(`adaptive_matrix: this lane reached its own budget deadline ${new Date(deadline).toISOString()} while the paired ${peerBackend} lane was still able to deliver but had published no valid inline anchors; no backend or timing is guessed.${rejections()}`);
    const runs = listDispatchRuns({ gh, repository, headSha });
    for (const artifact of peerAnchorCandidates({ gh, repository, name, headSha, selfRunId, runs })) {
      if (rejected.has(artifact.id)) continue;
      try { return { state: 'anchors', artifact, result: await validate(artifact) }; }
      catch (error) { rejected.set(artifact.id, error.message); log(`H3_PEER_ANCHORS_REJECTED: artifact ${artifact.id} (${artifact.name}, run ${artifact.workflow_run?.id}): ${error.message}`); }
    }
    let classification;
    try { classification = assertPeerCanDeliver(observePeerLane({ gh, repository, headSha, selfRunId, peerBackend, runs }), { peerBackend, now: now(), startedAt, graceMs, anchorsPublished: false }); }
    catch (error) { error.message += rejections(); throw error; }
    if (polls % 10 === 0) log(`H3_PEER_ANCHOR_WAIT: paired ${peerBackend} lane is ${classification.state}${classification.run ? ` (run ${classification.run.id})` : ''}; waiting until ${new Date(deadline).toISOString()} at the latest`);
    await sleep(Math.max(0, Math.min(pollMs, deadline - now())));
  }
}

/** Read the dispatch runs at `headSha` (or the given `runs`) and their jobs through `gh api` (an injected executor). */
export function observePeerLane({ gh, repository, workflowPath = DISPATCH_WORKFLOW_PATH, headSha, selfRunId, peerBackend, runs = listDispatchRuns({ gh, repository, workflowPath, headSha }) }) {
  const jobsByRun = new Map();
  for (const run of runs) {
    if (String(run.id) === String(selfRunId) || run.head_sha !== headSha) continue;
    jobsByRun.set(String(run.id), JSON.parse(gh(['api', '--paginate', '--slurp', `repos/${repository}/actions/runs/${run.id}/attempts/${run.run_attempt}/jobs?per_page=100`])).flatMap(page => page.jobs ?? []));
  }
  return classifyPeerLane({ runs, jobsByRun, selfRunId, headSha, peerBackend });
}
