import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { claimTerminalAttempt } from "./starvector-terminal-attempt.mjs";

test("failed attempts can advance on the same pin without rewriting historical markers", async (t) => {
  const root = await mkdtemp(path.join(tmpdir(), "starvector-attempt-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const pin = "a".repeat(40), old = path.join(root, `starvector-terminal-${pin}.campaign.json`), bytes = "original immutable marker\n";
  await writeFile(old, bytes);
  const predecessor = { campaign_id: "failed", workflow: { run_id: "10", run_attempt: 1, conclusion: "failure" } };
  const options = { workflowRunId: "11", workflowRunAttempt: 1, predecessor };
  await claimTerminalAttempt(root, pin, "corrected", options);
  await claimTerminalAttempt(root, pin, "corrected", options); // next tuple in same workflow
  assert.equal(await readFile(old, "utf8"), bytes);
  await assert.rejects(claimTerminalAttempt(root, pin, "competing", options), /already claimed/);
  await assert.rejects(claimTerminalAttempt(root, pin, "corrected", { ...options, workflowRunAttempt: 2 }), /already claimed/);
  await claimTerminalAttempt(root, pin, "second-fix", { workflowRunId: "12", workflowRunAttempt: 1, predecessor: { campaign_id: "corrected", workflow: { run_id: "11", run_attempt: 1, conclusion: "cancelled" } } });
  assert.equal(await readFile(old, "utf8"), bytes);
});

test("successor requires distinct failed workflow and safe portable identities", async () => {
  const base = { workflowRunId: "10", workflowRunAttempt: 1, predecessor: { campaign_id: "old", workflow: { run_id: "10", run_attempt: 1, conclusion: "failure" } } };
  await assert.rejects(claimTerminalAttempt("unused", "a".repeat(40), "new", base), /reuse the failed/);
  await assert.rejects(claimTerminalAttempt("unused", "a".repeat(40), "../new", base), /portable/);
  await assert.rejects(claimTerminalAttempt("unused", "a".repeat(40), "new", { ...base, predecessor: { ...base.predecessor, workflow: { ...base.predecessor.workflow, conclusion: "success" } } }), /failed predecessor/);
});
