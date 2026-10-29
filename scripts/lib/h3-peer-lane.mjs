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
// paired with whichever upload happened first.
//
// Pairing rule: anchors are taken ONLY from the paired peer run, which is the newest dispatch run
// at this SHA that carries the peer backend's terminal job. While a newer dispatch at this SHA has
// not created its jobs yet (it may be the real peer), no anchors are taken. A finished earlier pair
// at the same SHA is never mixed with the live one. Within the paired run, the newest attempt that
// validates wins.
export const DISPATCH_WORKFLOW_PATH = '.github/workflows/server-candle-linux.yml';
/** The anchor wait's base poll interval; it backs off to ANCHOR_POLL_MAX_MS while nothing changes. */
export const ANCHOR_POLL_MS = 60 * 1000;
export const ANCHOR_POLL_MAX_MS = 5 * 60 * 1000;
/** gh api retries (5xx, rate limit, network) with exponential backoff from GH_BACKOFF_MS. */
export const GH_RETRIES = 4;
export const GH_BACKOFF_MS = 5000;
/** How long the Actions API may stay unreachable (every retry failing) before the wait gives up. */
export const API_OUTAGE_LIMIT_MS = 30 * 60 * 1000;
/** Download attempts per candidate per poll (a fresh directory each); a failed download is retried next poll, never rejected. */
export const DOWNLOAD_ATTEMPTS = 3;

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

const defaultSleep = ms => new Promise(resolve => setTimeout(resolve, ms));
/** `gh api` through the injected executor, retrying any failure with exponential backoff. */
export async function ghJson(gh, args, { sleep = defaultSleep, retries = GH_RETRIES, backoffMs = GH_BACKOFF_MS, log = () => {} } = {}) {
  for (let attempt = 1; ; attempt++) {
    try { return JSON.parse(gh(args)); }
    catch (error) {
      if (attempt > retries) throw new Error(`gh ${args.join(' ')} failed ${attempt} times: ${error.message}`);
      const wait = backoffMs * 3 ** (attempt - 1);
      log(`H3_GH_RETRY: ${args.find(a => a.startsWith('repos/')) ?? args.join(' ')} failed (${String(error.message).split('\n')[0]}); retry ${attempt}/${retries} in ${wait / 1000} s`);
      await sleep(wait);
    }
  }
}

/** The paired peer run (see the pairing rule above), or null while it cannot be told yet. */
export function pairedPeerRun({ runs, jobsByRun, selfRunId, headSha, peerBackend }) {
  const candidates = (runs ?? []).filter(run => String(run.id) !== String(selfRunId) && run.head_sha === headSha && run.event === 'workflow_dispatch');
  const peers = candidates.filter(run => (jobsByRun.get(String(run.id)) ?? []).some(job => isPeerTerminalJob(job, peerBackend))).sort((a, b) => Number(b.id) - Number(a.id));
  const newest = peers[0] ?? null;
  const newerUnknown = candidates.some(run => !(jobsByRun.get(String(run.id)) ?? []).length && ACTIVE.has(run.status) && (!newest || Number(run.id) > Number(newest.id)));
  return newerUnknown ? null : newest;
}

/**
 * Reads the peer lane through `gh api` with retries, caching what cannot change any more: the
 * jobs and artifacts of a completed run attempt. Artifacts are listed only for the paired peer run.
 */
export class PeerLaneObserver {
  constructor({ gh, repository, headSha, selfRunId, peerBackend, workflowPath = DISPATCH_WORKFLOW_PATH, sleep = defaultSleep, log = () => {} }) {
    Object.assign(this, { gh, repository, headSha, selfRunId, peerBackend, workflowPath, sleep, log });
    this.jobsCache = new Map(); this.artifactsCache = new Map();
  }
  api(route) { return ghJson(this.gh, ['api', '--paginate', '--slurp', route], { sleep: this.sleep, log: this.log }); }
  async runs() {
    const workflow = encodeURIComponent(this.workflowPath.split('/').pop());
    return (await this.api(`repos/${this.repository}/actions/workflows/${workflow}/runs?event=workflow_dispatch&head_sha=${this.headSha}&per_page=100`)).flatMap(page => page.workflow_runs ?? []);
  }
  async cached(cache, run, route, field) {
    const key = `${run.id}/${run.run_attempt}`;
    if (cache.has(key)) return cache.get(key);
    const rows = (await this.api(route)).flatMap(page => page[field] ?? []);
    if (run.status === 'completed') cache.set(key, rows);
    return rows;
  }
  /** One observation: the dispatch runs, their jobs, the classification, the paired peer run and its anchor candidates (newest first). */
  async observe({ name } = {}) {
    const runs = await this.runs(), jobsByRun = new Map();
    for (const run of runs) {
      if (String(run.id) === String(this.selfRunId) || run.head_sha !== this.headSha) continue;
      jobsByRun.set(String(run.id), await this.cached(this.jobsCache, run, `repos/${this.repository}/actions/runs/${run.id}/attempts/${run.run_attempt}/jobs?per_page=100`, 'jobs'));
    }
    const context = { runs, jobsByRun, selfRunId: this.selfRunId, headSha: this.headSha, peerBackend: this.peerBackend };
    const classification = classifyPeerLane(context), paired = pairedPeerRun(context), candidates = [];
    if (paired && name !== undefined) {
      for (const artifact of await this.cached(this.artifactsCache, paired, `repos/${this.repository}/actions/runs/${paired.id}/artifacts?per_page=100`, 'artifacts')) {
        const runAttempt = anchorArtifactAttempt(artifact.name, name);
        if (runAttempt === null || artifact.expired || artifact.workflow_run?.head_sha !== this.headSha || String(artifact.workflow_run?.id) !== String(paired.id)) continue;
        candidates.push({ ...artifact, runAttempt });
      }
      candidates.sort((a, b) => b.runAttempt - a.runAttempt || Date.parse(b.created_at) - Date.parse(a.created_at));
    }
    const fingerprint = JSON.stringify([runs.map(run => [run.id, run.status, run.run_attempt]), [...jobsByRun].map(([id, jobs]) => [id, jobs.map(job => [job.id, job.status])]), candidates.map(a => a.id)]);
    return { runs, jobsByRun, classification, paired, candidates, fingerprint };
  }
}

/** The paired peer run's inline-anchor artifacts for `name` (the base), newest attempt first. */
export async function peerAnchorCandidates({ gh, repository, name, headSha, selfRunId, peerBackend, sleep, log }) {
  return (await new PeerLaneObserver({ gh, repository, headSha, selfRunId, peerBackend, sleep, log }).observe({ name })).candidates;
}
/** The newest published inline-anchor artifact of the paired peer run, or null. */
export async function findPeerAnchors(options) {
  return (await peerAnchorCandidates(options))[0] ?? null;
}

/**
 * Wait (bounded) until the peer either published its anchors or is alive; throw as soon as it is
 * known it never can. `sleep` and `now` are injected so the policy is testable without waiting.
 */
export async function awaitPeerLane({ gh, repository, headSha, selfRunId, peerBackend, anchorName, startedAt, now = () => Date.now(), sleep = defaultSleep, pollMs = 20000, graceMs = PEER_APPEARANCE_GRACE_MS, log }) {
  const observer = new PeerLaneObserver({ gh, repository, headSha, selfRunId, peerBackend, sleep, log });
  for (;;) {
    const { classification, candidates } = await observer.observe({ name: anchorName });
    if (candidates.length) return { state: 'anchors', anchors: candidates[0] };
    assertPeerCanDeliver(classification, { peerBackend, now: now(), startedAt, graceMs, anchorsPublished: false });
    if (classification.state === 'alive') return classification;
    await sleep(pollMs);
  }
}

/**
 * The adaptive matrix's anchor wait. It lasts exactly as long as the paired peer can still deliver:
 * it returns the newest paired-run anchors that `validate` accepts, throws as soon as the peer lane
 * is observed ended, failed or absent without valid anchors, and is otherwise bounded only by
 * `deadline` (this lane's work deadline minus what it still needs afterwards; it used to stop at a
 * fixed six hours never sized against a measured peer duration). `download(artifact)` is IO: a
 * failure is retried with backoff (the caller uses a fresh directory each try) and the candidate is
 * kept for the next poll. Only `validate(artifact, directory)` rejects a candidate permanently. A
 * transient Actions API failure is retried; only a continuous outage past API_OUTAGE_LIMIT_MS ends
 * the wait. The poll interval backs off while nothing observable changes.
 */
export async function waitForPeerAnchors({ gh, repository, name, headSha, selfRunId, peerBackend, startedAt, deadline, download, validate, now = () => Date.now(), sleep = defaultSleep, pollMs = ANCHOR_POLL_MS, maxPollMs = ANCHOR_POLL_MAX_MS, graceMs = PEER_APPEARANCE_GRACE_MS, outageLimitMs = API_OUTAGE_LIMIT_MS, downloadAttempts = DOWNLOAD_ATTEMPTS, log = () => {} }) {
  if (!Number.isFinite(deadline)) throw new Error('the anchor wait requires this lane\'s work deadline');
  if (typeof download !== 'function' || typeof validate !== 'function') throw new Error('the anchor wait requires separate download (retryable) and validate (permanent) steps');
  const observer = new PeerLaneObserver({ gh, repository, headSha, selfRunId, peerBackend, sleep, log });
  const rejected = new Map(), lastDownloadError = new Map();
  const notes = () => `${rejected.size ? ` Rejected anchors: ${[...rejected].map(([id, reason]) => `${id}: ${reason}`).join('; ')}.` : ''}${lastDownloadError.size ? ` Last download errors: ${[...lastDownloadError].map(([id, reason]) => `${id}: ${reason}`).join('; ')}.` : ''}`;
  const pause = ms => sleep(Math.max(0, Math.min(ms, deadline - now())));
  let interval = pollMs, fingerprint, outageSince = null, polls = 0;
  for (;;) {
    if (now() >= deadline) throw new Error(`adaptive_matrix: this lane reached its peer-anchor wait deadline ${new Date(deadline).toISOString()} (its work deadline minus the time its remaining cases need) while the paired ${peerBackend} lane was still able to deliver but had published no valid inline anchors; no backend or timing is guessed.${notes()}`);
    let observed;
    try { observed = await observer.observe({ name }); outageSince = null; }
    catch (error) {
      outageSince ??= now();
      if (now() - outageSince >= outageLimitMs) throw new Error(`adaptive_matrix: the Actions API stayed unreachable for ${Math.round((now() - outageSince) / 60000)} minutes while waiting for the paired ${peerBackend} anchors: ${error.message}${notes()}`);
      log(`H3_PEER_ANCHOR_WAIT: Actions API unavailable (${error.message.split('\n')[0]}); polling again`);
      await pause(interval); continue;
    }
    for (const artifact of observed.candidates) {
      if (rejected.has(artifact.id)) continue;
      let directory;
      for (let attempt = 1; attempt <= downloadAttempts && directory === undefined; attempt++) {
        try { directory = await download(artifact); lastDownloadError.delete(artifact.id); }
        catch (error) { lastDownloadError.set(artifact.id, error.message); log(`H3_PEER_ANCHORS_DOWNLOAD_RETRY: artifact ${artifact.id} (${artifact.name}) attempt ${attempt}/${downloadAttempts}: ${error.message}`); if (attempt < downloadAttempts) await pause(GH_BACKOFF_MS * 3 ** (attempt - 1)); }
      }
      if (directory === undefined) continue; // Kept: tried again on the next poll.
      try { return { state: 'anchors', artifact, directory, result: await validate(artifact, directory) }; }
      catch (error) { rejected.set(artifact.id, error.message); log(`H3_PEER_ANCHORS_REJECTED: artifact ${artifact.id} (${artifact.name}, run ${artifact.workflow_run?.id}): ${error.message}`); }
    }
    const pendingDownloads = observed.candidates.some(artifact => !rejected.has(artifact.id));
    let classification;
    try { classification = assertPeerCanDeliver(observed.classification, { peerBackend, now: now(), startedAt, graceMs, anchorsPublished: pendingDownloads }); }
    catch (error) { error.message += notes(); throw error; }
    interval = observed.fingerprint === fingerprint && !pendingDownloads ? Math.min(interval * 2, maxPollMs) : pollMs;
    fingerprint = observed.fingerprint;
    if (polls++ % 10 === 0) log(`H3_PEER_ANCHOR_WAIT: paired ${peerBackend} lane is ${classification.state}${observed.paired ? ` (paired run ${observed.paired.id})` : ''}; waiting until ${new Date(deadline).toISOString()} at the latest`);
    await pause(interval);
  }
}

/** Read the dispatch runs at `headSha` and their jobs through `gh api` (an injected executor), with retries. */
export async function observePeerLane({ gh, repository, workflowPath = DISPATCH_WORKFLOW_PATH, headSha, selfRunId, peerBackend, sleep, log }) {
  return (await new PeerLaneObserver({ gh, repository, workflowPath, headSha, selfRunId, peerBackend, sleep, log }).observe()).classification;
}
