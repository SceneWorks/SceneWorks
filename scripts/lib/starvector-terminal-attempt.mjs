import { lstat, mkdir, open, readFile } from "node:fs/promises";
import path from "node:path";

const ID = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;
const SHA = /^[a-f0-9]{40}$/;
const fail = (message) => { throw new Error(`starvector terminal attempt: ${message}`); };

async function appendIdentity(file, value) {
  const bytes = JSON.stringify(value, null, 2) + "\n";
  let handle;
  try {
    handle = await open(file, "wx", 0o600);
    await handle.writeFile(bytes);
    await handle.sync();
  } catch (error) {
    if (error.code !== "EEXIST") throw error;
    const info = await lstat(file);
    if (!info.isFile() || info.isSymbolicLink()) fail("attempt claim is not a regular file");
    if (await readFile(file, "utf8") !== bytes) fail(`attempt identity already claimed: ${file}`);
  } finally { await handle?.close(); }
}

// A source revision identifies code, not a single execution. Retain all old
// pin-keyed markers, then append one successor per failed attempt. Further
// recovery names that successor as predecessor rather than rewriting this link.
export async function claimTerminalAttempt(leaseRoot, permanentPin, campaignRunId, {
  workflowRunId, workflowRunAttempt, predecessor, platform = process.platform,
} = {}) {
  if (!SHA.test(permanentPin ?? "") || !ID.test(campaignRunId ?? "")) fail("exact pin and portable campaign identity required");
  if (!/^[1-9][0-9]*$/.test(String(workflowRunId ?? "")) || !Number.isSafeInteger(workflowRunAttempt) || workflowRunAttempt < 1) fail("current workflow run and attempt required");
  if (!predecessor || !ID.test(predecessor.campaign_id ?? "") || predecessor.campaign_id === campaignRunId || !["failure", "cancelled", "timed_out"].includes(predecessor.workflow?.conclusion)) fail("a distinct verified failed predecessor is required");
  if (String(predecessor.workflow.run_id) === String(workflowRunId) && predecessor.workflow.run_attempt === workflowRunAttempt) fail("successor cannot reuse the failed workflow attempt");
  const root = path.join(leaseRoot, "starvector-attempts");
  await mkdir(root, { recursive: true });
  if ((await lstat(root)).isSymbolicLink()) fail("attempt directory is a symlink");
  if (predecessor.stage === "upstream-reference") {
    const readClaim = async (file) => {
      const info = await lstat(file).catch(error => error.code === "ENOENT" ? null : Promise.reject(error));
      if (!info) return null;
      if (!info.isFile() || info.isSymbolicLink()) fail("predecessor claim is not a regular file");
      return readFile(file, "utf8");
    };
    const prior = predecessor.predecessor_campaign_id;
    if (!ID.test(prior ?? "") || prior === predecessor.campaign_id) fail("failed execution lacks its original predecessor");
    const claimed = await readClaim(path.join(root, `${predecessor.campaign_id}.json`));
    const priorLink = await readClaim(path.join(root, `successor-of-${prior}.json`));
    const marker = await readClaim(path.join(leaseRoot, `starvector-terminal-${predecessor.inference_revision}-${predecessor.campaign_id}-upstream-reference.tuple.json`));
    if (claimed === null && priorLink === null && marker === null) {
      // The authenticated job census proved native tuples never ran. The Mac
      // therefore has no claim to fabricate; Windows must retain the real one.
      if (platform !== "darwin") fail("failed upstream execution claim is missing on its host");
    } else {
      const expected = { schema_version: 1, permanent_pin: predecessor.inference_revision, campaign_run_id: predecessor.campaign_id, workflow_run_id: String(predecessor.workflow.run_id), workflow_run_attempt: predecessor.workflow.run_attempt, predecessor_campaign_id: prior };
      if (claimed !== `${JSON.stringify(expected, null, 2)}\n` || priorLink !== claimed || marker === null) fail("failed upstream execution claim differs from authenticated predecessor");
      const value = JSON.parse(marker);
      if (Object.keys(value).sort().join(",") !== "campaign_run_id,permanent_pin,started_at,tuple" || value.permanent_pin !== predecessor.inference_revision || value.campaign_run_id !== predecessor.campaign_id || value.tuple !== "upstream-reference" || !Number.isFinite(Date.parse(value.started_at))) fail("failed upstream tuple marker differs");
    }
  }
  const identity = { schema_version: 1, permanent_pin: permanentPin, campaign_run_id: campaignRunId, workflow_run_id: String(workflowRunId), workflow_run_attempt: workflowRunAttempt, predecessor_campaign_id: predecessor.campaign_id };
  await appendIdentity(path.join(root, `successor-of-${predecessor.campaign_id}.json`), identity);
  await appendIdentity(path.join(root, `${campaignRunId}.json`), identity);
  return identity;
}
