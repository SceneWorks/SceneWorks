import { mkdir, open, readFile } from "node:fs/promises";
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
    if (await readFile(file, "utf8") !== bytes) fail(`attempt identity already claimed: ${file}`);
  } finally { await handle?.close(); }
}

// A source revision identifies code, not a single execution. Retain all old
// pin-keyed markers, then append one successor per failed attempt. Further
// recovery names that successor as predecessor rather than rewriting this link.
export async function claimTerminalAttempt(leaseRoot, permanentPin, campaignRunId, {
  workflowRunId, workflowRunAttempt, predecessor,
} = {}) {
  if (!SHA.test(permanentPin ?? "") || !ID.test(campaignRunId ?? "")) fail("exact pin and portable campaign identity required");
  if (!/^[1-9][0-9]*$/.test(String(workflowRunId ?? "")) || !Number.isSafeInteger(workflowRunAttempt) || workflowRunAttempt < 1) fail("current workflow run and attempt required");
  if (!predecessor || !ID.test(predecessor.campaign_id ?? "") || predecessor.campaign_id === campaignRunId || !["failure", "cancelled", "timed_out"].includes(predecessor.workflow?.conclusion)) fail("a distinct verified failed predecessor is required");
  if (String(predecessor.workflow.run_id) === String(workflowRunId) && predecessor.workflow.run_attempt === workflowRunAttempt) fail("successor cannot reuse the failed workflow attempt");
  const root = path.join(leaseRoot, "starvector-attempts");
  await mkdir(root, { recursive: true });
  const identity = { schema_version: 1, permanent_pin: permanentPin, campaign_run_id: campaignRunId, workflow_run_id: String(workflowRunId), workflow_run_attempt: workflowRunAttempt, predecessor_campaign_id: predecessor.campaign_id };
  await appendIdentity(path.join(root, `successor-of-${predecessor.campaign_id}.json`), identity);
  await appendIdentity(path.join(root, `${campaignRunId}.json`), identity);
  return identity;
}
