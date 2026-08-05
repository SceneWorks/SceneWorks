#!/usr/bin/env node
// Decide whether a merge-queue speculative merge touches anything a path-filtered lane cares about.
//
// ## Why this exists
//
// GitHub's `merge_group` event supports NO `paths:` / `paths-ignore:` filter — only `branches:`
// (community discussion 45899; still open). Every other trigger on the backend lanes is path-gated,
// and those filters are not decoration: `macos-mlx.yml` runs on a hand-registered pool of two Macs,
// one of them a developer's daily driver, and `windows-candle.yml` runs on the fleet's single `cuda`
// box. A bare `merge_group:` would wake them for a README-only queue entry.
//
// So the filter moves from the trigger into a hosted gate job, and this script is that gate.
//
// ## The paths are READ FROM THE WORKFLOW, not copied
//
// The lane's own `paths: &anchor` block is the single source of truth. Copying the list here would
// rot on the first entry added to one copy and not the other — and it would rot SILENTLY, in the
// safe-looking direction: a new watched path that this script does not know about makes the gate say
// "not relevant" and skips the lane, which reads as a green merge group rather than as a hole.
// `scripts/platform-review-contracts.test.mjs` already forces new watched paths into that anchor, so
// parsing the anchor inherits that guard for free.
//
// ## Conservative on doubt
//
// An unparseable anchor, an empty changed-file list, or a missing base is `run=true`. A gate that
// cannot tell must run the lane: the cost of a wrong `true` is one job, and the cost of a wrong
// `false` is an unvalidated merge into `main`.

import { appendFile, readFile } from "node:fs/promises";
import process from "node:process";
import { pathToFileURL } from "node:url";

/**
 * GitHub filter-pattern syntax: `*` matches any run of characters except `/`, `**` matches any run
 * including `/`, and `**​/` is "zero or more directories". Deliberately the same semantics as the
 * `matches()` helper in scripts/platform-review-contracts.test.mjs — that one audits the anchors,
 * this one evaluates them, and they must agree about what an anchor entry means.
 */
export function matchesGlob(glob, target) {
  let pattern = "";
  for (let i = 0; i < glob.length; i += 1) {
    const char = glob[i];
    if (char === "*") {
      if (glob[i + 1] === "*") {
        if (glob[i + 2] === "/") {
          pattern += "(?:.*/)?";
          i += 2;
        } else {
          pattern += ".*";
          i += 1;
        }
      } else {
        pattern += "[^/]*";
      }
    } else {
      pattern += "\\^$+?.()|[]{}".includes(char) ? `\\${char}` : char;
    }
  }
  return new RegExp(`^${pattern}$`).test(target);
}

/**
 * Pull the entries out of a workflow's `paths: &<anchor>` block.
 *
 * Every entry is required to be a bare double-quoted string — the same shape
 * platform-review-contracts.test.mjs already enforces on these anchors, which is what makes a
 * line-oriented read of the block sound. A `- entry` without quotes is a parse failure rather than a
 * silently dropped path.
 */
export function parseWorkflowPaths(source, anchor) {
  const marker = `paths: &${anchor}`;
  const at = source.indexOf(marker);
  if (at < 0) throw new Error(`workflow declares no \`${marker}\` anchor`);

  const entries = [];
  for (const line of source.slice(at + marker.length).split("\n").slice(1)) {
    if (line.trim() === "" || line.trim().startsWith("#")) continue;
    const item = /^\s+- (.*)$/.exec(line);
    if (!item) break; // dedented to the next `on:` key — the block is over.
    const quoted = /^"([^"]*)"$/.exec(item[1].trim());
    if (!quoted) {
      throw new Error(
        `anchor \`${anchor}\` entry is not a bare double-quoted string: ${item[1].trim()}`,
      );
    }
    entries.push(quoted[1]);
  }
  if (entries.length === 0) throw new Error(`anchor \`${anchor}\` is empty`);
  return entries;
}

export function relevanceDecision(globs, changedFiles) {
  if (changedFiles.length === 0) {
    return {
      run: true,
      reason: "No changed files could be determined; running the lane conservatively.",
    };
  }
  for (const file of changedFiles) {
    const glob = globs.find((candidate) => matchesGlob(candidate, file));
    if (glob) return { run: true, reason: `${file} matches watched path "${glob}".` };
  }
  return {
    run: false,
    reason:
      `None of the ${changedFiles.length} changed file(s) match the lane's ${globs.length} ` +
      "watched paths.",
  };
}

function flag(name) {
  const at = process.argv.indexOf(name);
  return at < 0 ? undefined : process.argv[at + 1];
}

async function main() {
  const workflow = flag("--workflow");
  const anchor = flag("--anchor");
  if (!workflow || !anchor) {
    throw new Error(
      "usage: merge-group-relevance.mjs --workflow <path> --anchor <name> < changed-files",
    );
  }

  let input = "";
  process.stdin.setEncoding("utf8");
  for await (const chunk of process.stdin) input += chunk;
  const changedFiles = input.split(/\r?\n/).filter(Boolean);

  // A malformed anchor is a repo bug, not a reason to skip a backend lane on the way into `main`.
  let decision;
  try {
    decision = relevanceDecision(parseWorkflowPaths(await readFile(workflow, "utf8"), anchor), changedFiles);
  } catch (error) {
    decision = { run: true, reason: `Could not evaluate watched paths (${error.message}); running the lane conservatively.` };
  }

  console.log(decision.reason);
  if (process.env.GITHUB_OUTPUT) {
    await appendFile(process.env.GITHUB_OUTPUT, `run=${decision.run}\n`, "utf8");
  } else {
    console.log(`run=${decision.run}`);
  }
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
}
