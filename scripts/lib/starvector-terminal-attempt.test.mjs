import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, symlink, writeFile } from "node:fs/promises";
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

test("upstream successor binds real prior claim and marker, preserves history, and is idempotent per workflow", async (t) => {
  const root = await mkdtemp(path.join(tmpdir(), "upstream-attempt-")); t.after(() => rm(root, { recursive: true, force: true }));
  const pin = "a".repeat(40), native = { campaign_id: "retired-native", workflow: { run_id: "10", run_attempt: 1, conclusion: "failure" } };
  await claimTerminalAttempt(root, pin, "upstream-failed", { workflowRunId: "11", workflowRunAttempt: 1, predecessor: native });
  const marker = path.join(root, `starvector-terminal-${pin}-upstream-failed-upstream-reference.tuple.json`);
  await writeFile(marker, JSON.stringify({ permanent_pin: pin, campaign_run_id: "upstream-failed", tuple: "upstream-reference", started_at: "2026-09-11T11:00:00Z" }));
  const files = [marker, path.join(root, "starvector-attempts/upstream-failed.json"), path.join(root, "starvector-attempts/successor-of-retired-native.json")];
  const original = await Promise.all(files.map(file => readFile(file, "utf8")));
  const predecessor = { stage: "upstream-reference", campaign_id: "upstream-failed", inference_revision: pin, predecessor_campaign_id: "retired-native", workflow: { run_id: "11", run_attempt: 1, conclusion: "failure" } };
  const options = { workflowRunId: "12", workflowRunAttempt: 1, predecessor, platform: "win32" };
  await claimTerminalAttempt(root, pin, "next", options);
  await claimTerminalAttempt(root, pin, "next", options);
  assert.deepEqual(await Promise.all(files.map(file => readFile(file, "utf8"))), original);
  await assert.rejects(() => claimTerminalAttempt(root, pin, "competitor", options), /already claimed/);
  await assert.rejects(() => claimTerminalAttempt(root, pin, "wrong-run", { ...options, predecessor: { ...predecessor, workflow: { ...predecessor.workflow, run_id: "9" } } }), /differs from authenticated/);
  await rm(files[1]); await symlink(files[2], files[1]);
  await assert.rejects(() => claimTerminalAttempt(root, pin, "linked-history", options), /not a regular file/);
  await rm(files[1]); await writeFile(files[1], original[1]);
  await writeFile(marker, JSON.stringify({ permanent_pin: pin, campaign_run_id: "upstream-failed", tuple: "mlx:1b", started_at: "2026-09-11T11:00:00Z" }));
  await assert.rejects(() => claimTerminalAttempt(root, pin, "wrong-marker", options), /tuple marker differs/);
});

test("upstream-only failure permits an unclaimed Mac, but never missing or partial Windows history", async (t) => {
  const root = await mkdtemp(path.join(tmpdir(), "unclaimed-attempt-")); t.after(() => rm(root, { recursive: true, force: true }));
  const predecessor = { stage: "upstream-reference", campaign_id: "failed", inference_revision: "a".repeat(40), predecessor_campaign_id: "older", workflow: { run_id: "10", run_attempt: 1, conclusion: "failure" } };
  const options = { workflowRunId: "11", workflowRunAttempt: 1, predecessor };
  await assert.rejects(() => claimTerminalAttempt(root, "a".repeat(40), "next", { ...options, platform: "win32" }), /claim is missing/);
  await claimTerminalAttempt(root, "a".repeat(40), "next", { ...options, platform: "darwin" });
  await writeFile(path.join(root, "starvector-attempts/successor-of-older.json"), "partial history");
  await assert.rejects(() => claimTerminalAttempt(root, "a".repeat(40), "next", { ...options, platform: "darwin" }), /differs from authenticated/);
});
