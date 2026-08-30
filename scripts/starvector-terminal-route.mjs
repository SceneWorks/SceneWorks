#!/usr/bin/env node
// Source-owned production entrypoint. It only uses the typed Vector Studio route;
// the pre-provisioned metric command converts completed route artifacts into the
// canonical raw-results.json consumed by the pinned inference validator.
import { createHash } from "node:crypto";
import { execFile as execFileCallback } from "node:child_process";
import { appendFile, readFile, stat } from "node:fs/promises";
import { promisify } from "node:util";
import path from "node:path";

const execFile = promisify(execFileCallback);
const die = (message) => { throw new Error(`starvector terminal route: ${message}`); };
const terminal = new Set(["completed", "failed", "cancelled", "canceled"]);
const sha = (value) => createHash("sha256").update(value).digest("hex");
const json = async (file) => JSON.parse(await readFile(file, "utf8"));

export function vectorRequest(record) {
  if (!record?.projectId || !record.sourceAssetId || !record.model) die("each case requires projectId, sourceAssetId, and StarVector model");
  return { projectId: record.projectId, projectName: record.projectName, mode: "image_to_svg", model: record.model, sourceAssetId: record.sourceAssetId, prompt: record.prompt ?? "", sampling: record.sampling, detailBudget: record.detailBudget };
}

async function request(url, init) {
  const response = await fetch(url, init); const body = await response.json();
  if (!response.ok) die(`typed vector route ${response.status}: ${JSON.stringify(body)}`); return body;
}
export async function submitAndPoll(baseUrl, record, transcript, fetchOptions = {}) {
  const created = await request(new URL("/api/v1/image/vectorize/jobs", baseUrl), { method: "POST", headers: { "content-type": "application/json", ...fetchOptions.headers }, body: JSON.stringify(vectorRequest(record)) });
  if (created.type !== "vector_generate" || !created.id) die("typed vector route did not create vector_generate job");
  await appendFile(transcript, JSON.stringify({ phase: "created", case_id: record.case_id, job: created }) + "\n");
  for (let attempt = 0; attempt < 7200; attempt += 1) {
    const job = await request(new URL(`/api/v1/jobs/${created.id}`, baseUrl), { headers: fetchOptions.headers });
    await appendFile(transcript, JSON.stringify({ phase: "polled", case_id: record.case_id, job }) + "\n");
    if (terminal.has(job.status)) return job;
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
  die(`vector_generate job ${created.id} did not finish within two hours`);
}

async function main() {
  const output = process.env.STARVECTOR_TERMINAL_OUTPUT, tuple = process.env.STARVECTOR_TERMINAL_TUPLE, bundlePath = process.env.STARVECTOR_TERMINAL_CASE_BUNDLE, baseUrl = process.env.STARVECTOR_TERMINAL_API_URL, metricCommand = process.env.STARVECTOR_TERMINAL_METRIC_COMMAND;
  if (process.env.STARVECTOR_TERMINAL_NO_JOB_DOWNLOADS !== "1") die("no-job-downloads guard is required");
  if (!output || !tuple || !bundlePath || !baseUrl || !metricCommand) die("output, tuple, case bundle, API URL, and pre-provisioned metric command required");
  const bundle = await json(bundlePath); const cases = bundle?.tuples?.[tuple];
  if (!Array.isArray(cases) || cases.length !== 120) die(`case bundle must carry exactly 120 ${tuple} route cases`);
  const transcript = path.join(output, "vector-generate-route.ndjson");
  for (const record of cases) await submitAndPoll(baseUrl, record, transcript, bundle.fetch ?? {});
  const transcriptSha = sha(await readFile(transcript));
  await execFile(metricCommand, [bundlePath, output, tuple, transcriptSha], { env: { ...process.env, STARVECTOR_TERMINAL_ROUTE_TRANSCRIPT_SHA256: transcriptSha } });
  await stat(path.join(output, "raw-results.json"));
}
if (import.meta.url === `file://${process.argv[1]}`) main().catch((error) => { console.error(error.message); process.exitCode = 1; });
