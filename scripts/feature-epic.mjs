#!/usr/bin/env node

import { existsSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

import {
  FeatureEpicError,
  adoptPullRequest,
  auditState,
  bootstrap,
  createPlan,
  enforceCiPolicy,
  parseReceipt,
  readJson,
  recordEvidence,
  refreshStoryInventory,
  recover,
} from "./lib/feature-epic.mjs";

const USAGE = `Usage:
  node scripts/feature-epic.mjs plan \\
    --epic sc-N --slug kebab-slug --stories sc-N,sc-N \\
    --workspace /absolute/empty/parent --state /absolute/state.json \\
    [--adopt-existing] [--deployment-required] [--privileged-workflow repo:file.yml ...] \\
    [--recover-stale-lock-after-seconds N]

  node scripts/feature-epic.mjs bootstrap --state /absolute/state.json --apply \
    [--recover-stale-lock-after-seconds N]
  node scripts/feature-epic.mjs recover --state /absolute/state.json --complete \
    [--recover-stale-lock-after-seconds N]
  node scripts/feature-epic.mjs refresh-stories --state /absolute/state.json --apply \
    [--recover-stale-lock-after-seconds N]
  node scripts/feature-epic.mjs record-pr --state /absolute/state.json \\
    --repo sceneworks|inference --number N \\
    [--run repo:run-id ...] [--recover-stale-lock-after-seconds N]
  node scripts/feature-epic.mjs record-pr --state /absolute/state.json \\
    --repo sceneworks|inference --number N \\
    [--deployment repo:deployment-id ...] [--recover-stale-lock-after-seconds N]
  node scripts/feature-epic.mjs adopt-pr --state /absolute/state.json \\
    --repo sceneworks|inference --number N \\
    --disposition precanonical-story|historical-train|governance-proof \\
    [--recover-stale-lock-after-seconds N]
  node scripts/feature-epic.mjs ci-policy \\
    [--event /absolute/event.json] [--event-name pull_request|merge_group|push|workflow_dispatch] \\
    [--repo-root /absolute/checkout]
  node scripts/feature-epic.mjs audit \\
    --state /absolute/state.json --report /absolute/report.json \
    [--recover-stale-lock-after-seconds N]

Safety properties:
  * plan reads the complete live Shortcut epic and is otherwise read-only except for atomically creating the explicit state file.
  * refresh-stories adopts additive live scope only before any final-integration PR intent.
  * state/report commands hold an exclusive owner-recorded lock; stale recovery is explicit unless a same-host PID is dead.
  * create-mode bootstrap stages both state-owned queues, then creates, activates, and verifies each repository.
  * adopt-existing validates immutable heads, checks, pins, and active/effective policies without remote mutation.
  * adopt-pr records exact merged history; no waiver or open/unmerged PR can satisfy adoption.
  * recover may disable only a state-created exact id around its missing create-only ref.
  * recover reconciles and reactivates every state-owned queue before exposing a clone.
  * no command updates, force-pushes, or deletes a remote ref.
  * no command updates an unknown or pre-existing ruleset.
  * audit is report-only; cleanup remains an explicit, separately authorized operation.`;

function fail(message, code = 1) {
  process.stderr.write(`feature-epic: ${message}\n`);
  process.exitCode = code;
}

function parseArgs(argv) {
  const command = argv[0];
  if (!command || command.startsWith("-")) throw new FeatureEpicError(USAGE, "USAGE");
  const values = new Map();
  const flags = new Set();
  const repeatable = new Set(["--run", "--deployment", "--privileged-workflow"]);
  const booleans = new Set(["--apply", "--complete", "--deployment-required", "--adopt-existing"]);
  for (let index = 1; index < argv.length; index += 1) {
    const token = argv[index];
    if (!token.startsWith("--")) throw new FeatureEpicError(`unexpected positional argument: ${token}`, "USAGE");
    if (booleans.has(token)) {
      if (flags.has(token)) throw new FeatureEpicError(`duplicate flag: ${token}`, "USAGE");
      flags.add(token);
      continue;
    }
    const next = argv[index + 1];
    if (next === undefined || next.startsWith("--")) {
      throw new FeatureEpicError(`${token} requires a value`, "USAGE");
    }
    index += 1;
    if (repeatable.has(token)) {
      const current = values.get(token) ?? [];
      current.push(next);
      values.set(token, current);
    } else {
      if (values.has(token)) throw new FeatureEpicError(`duplicate option: ${token}`, "USAGE");
      values.set(token, next);
    }
  }
  return { command, values, flags };
}

function required(values, key) {
  const value = values.get(key);
  if (typeof value !== "string" || value.length === 0) {
    throw new FeatureEpicError(`${key} is required`, "USAGE");
  }
  return value;
}

function positiveInteger(value, label) {
  if (!/^[1-9][0-9]*$/.test(value ?? "")) {
    throw new FeatureEpicError(`${label} must be a positive integer`, "USAGE");
  }
  return Number(value);
}

function lockDependencies(values) {
  const raw = values.get("--recover-stale-lock-after-seconds");
  if (raw === undefined) return {};
  const seconds = positiveInteger(raw, "--recover-stale-lock-after-seconds");
  if (!Number.isSafeInteger(seconds) || seconds < 300) {
    throw new FeatureEpicError(
      "--recover-stale-lock-after-seconds must be a safe integer of at least 300",
      "USAGE",
    );
  }
  return { lockStaleAfterMs: seconds * 1000 };
}

function assertOnly(parsed, allowedOptions, allowedFlags = []) {
  const allowedValues = new Set(allowedOptions);
  const allowedBoolean = new Set(allowedFlags);
  for (const key of parsed.values.keys()) {
    if (!allowedValues.has(key)) throw new FeatureEpicError(`unknown option for ${parsed.command}: ${key}`, "USAGE");
  }
  for (const key of parsed.flags) {
    if (!allowedBoolean.has(key)) throw new FeatureEpicError(`unknown flag for ${parsed.command}: ${key}`, "USAGE");
  }
}

function privilegedWorkflow(value) {
  const match = /^(sceneworks|inference):([A-Za-z0-9][A-Za-z0-9._-]*\.ya?ml)$/.exec(value ?? "");
  if (!match) throw new FeatureEpicError("--privileged-workflow must be repo:file.yml", "USAGE");
  return { repo: match[1], workflow: match[2] };
}

function emit(value) {
  process.stdout.write(`${JSON.stringify(value, null, 2)}\n`);
}

function run(parsed) {
  const { command, values, flags } = parsed;
  if (command === "plan") {
    assertOnly(
      parsed,
      [
        "--epic",
        "--slug",
        "--stories",
        "--workspace",
        "--state",
        "--privileged-workflow",
        "--recover-stale-lock-after-seconds",
      ],
      ["--deployment-required", "--adopt-existing"],
    );
    const workflows = values.get("--privileged-workflow")?.map(privilegedWorkflow);
    const state = createPlan({
      epic: required(values, "--epic"),
      slug: required(values, "--slug"),
      stories: required(values, "--stories").split(",").filter(Boolean),
      workspaceParent: required(values, "--workspace"),
      statePath: required(values, "--state"),
      deploymentRequired: flags.has("--deployment-required"),
      privilegedWorkflows: workflows,
      adoptExisting: flags.has("--adopt-existing"),
    }, lockDependencies(values));
    emit({ ok: true, phase: state.phase, state: resolve(required(values, "--state")), featureBranch: state.featureBranch });
    return;
  }
  if (command === "bootstrap") {
    assertOnly(parsed, ["--state", "--recover-stale-lock-after-seconds"], ["--apply"]);
    const state = bootstrap(
      required(values, "--state"),
      { apply: flags.has("--apply") },
      lockDependencies(values),
    );
    emit({
      ok: true,
      phase: state.phase,
      featureBranch: state.featureBranch,
      policies: state.policies,
      repositories: state.repositories,
    });
    return;
  }
  if (command === "recover") {
    assertOnly(parsed, ["--state", "--recover-stale-lock-after-seconds"], ["--complete"]);
    const state = recover(
      required(values, "--state"),
      { complete: flags.has("--complete") },
      lockDependencies(values),
    );
    emit({
      ok: true,
      phase: state.phase,
      featureBranch: state.featureBranch,
      policies: state.policies,
      repositories: state.repositories,
    });
    return;
  }
  if (command === "refresh-stories") {
    assertOnly(parsed, ["--state", "--recover-stale-lock-after-seconds"], ["--apply"]);
    const result = refreshStoryInventory(
      required(values, "--state"),
      { apply: flags.has("--apply") },
      lockDependencies(values),
    );
    emit({
      ok: true,
      featureBranch: result.state.featureBranch,
      expectedStories: result.state.expectedStories,
      refresh: result.refresh,
    });
    return;
  }
  if (command === "record-pr") {
    assertOnly(
      parsed,
      ["--state", "--repo", "--number", "--run", "--deployment", "--recover-stale-lock-after-seconds"],
    );
    const receipt = recordEvidence(required(values, "--state"), {
      repo: required(values, "--repo"),
      number: positiveInteger(required(values, "--number"), "--number"),
      runReceipts: (values.get("--run") ?? []).map((value) => parseReceipt(value, "--run")),
      deploymentIds: (values.get("--deployment") ?? []).map((value) => parseReceipt(value, "--deployment")),
    }, lockDependencies(values));
    emit({ ok: true, recorded: receipt });
    return;
  }
  if (command === "adopt-pr") {
    assertOnly(
      parsed,
      ["--state", "--repo", "--number", "--disposition", "--recover-stale-lock-after-seconds"],
    );
    const receipt = adoptPullRequest(required(values, "--state"), {
      repo: required(values, "--repo"),
      number: positiveInteger(required(values, "--number"), "--number"),
      disposition: required(values, "--disposition"),
    }, lockDependencies(values));
    emit({ ok: true, adopted: receipt });
    return;
  }
  if (command === "ci-policy") {
    assertOnly(parsed, ["--event", "--event-name", "--repo-root"]);
    const eventPath = values.get("--event") ?? process.env.GITHUB_EVENT_PATH;
    const eventName = values.get("--event-name") ?? process.env.GITHUB_EVENT_NAME;
    const repoRoot = values.get("--repo-root") ?? process.env.GITHUB_WORKSPACE ?? resolve(fileURLToPath(new URL("..", import.meta.url)));
    if (!eventPath || !existsSync(eventPath)) {
      throw new FeatureEpicError("ci-policy requires an existing --event or GITHUB_EVENT_PATH", "USAGE");
    }
    if (!eventName) throw new FeatureEpicError("ci-policy requires --event-name or GITHUB_EVENT_NAME", "USAGE");
    emit(enforceCiPolicy({ eventName, event: readJson(eventPath, "GitHub event"), repoRoot }));
    return;
  }
  if (command === "audit") {
    assertOnly(parsed, ["--state", "--report", "--recover-stale-lock-after-seconds"]);
    const report = auditState(
      required(values, "--state"),
      required(values, "--report"),
      lockDependencies(values),
    );
    emit(report);
    if (!report.ok) process.exitCode = 1;
    return;
  }
  throw new FeatureEpicError(`unknown command: ${command}\n\n${USAGE}`, "USAGE");
}

try {
  run(parseArgs(process.argv.slice(2)));
} catch (error) {
  if (error instanceof FeatureEpicError) {
    fail(`${error.message}${error.code === "USAGE" && !error.message.includes("Usage:") ? `\n\n${USAGE}` : ""}`);
  } else {
    fail(error?.stack ?? String(error));
  }
}
