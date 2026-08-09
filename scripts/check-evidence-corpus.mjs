#!/usr/bin/env node
/**
 * Evidence-corpus shrink gate (sc-18224).
 *
 * After sc-18100 retired the one-shot census assertions, every remaining consistency check over
 * the calibration corpus verifies that artifacts agree WITH EACH OTHER — bundle vs manifest vs
 * matrix. None of them verifies that the corpus has not SHRUNK: a bad merge, an overzealous
 * regeneration, or an agent deleting an offending record and regenerating the matrix to match all
 * leave a perfectly self-consistent tree and pass every gate. This check closes that hole.
 *
 * What it compares: the set of record ids in `docs/generated/memory-calibration-evidence.json` at
 * HEAD (working tree, so uncommitted deletions are caught locally too) against the same file at
 * the merge-base of HEAD and origin/main. Any id present at the base but absent at HEAD fails
 * unless it is declared in `config/memory-evidence-tombstones.json` — an explicit, reviewable act.
 * Growth never fails. On main itself the merge-base IS HEAD, so the gate is inert post-merge.
 *
 * The tombstone file is kept honest in both directions: a deletion without a tombstone fails, and
 * a tombstone for a record that still exists at HEAD fails. Tombstones for records absent from
 * BOTH sides are history (the deletion already merged) and are deliberately allowed to remain.
 *
 * CI reality: `.github/workflows/check.yml` and `publish-runpod.yml` check out at the
 * actions/checkout default fetch-depth of 1, so neither origin/main nor HEAD's own parents exist
 * locally. `resolveBaseCommit` therefore fetches `origin main` and, if the histories are still
 * disconnected, deepens BOTH sides — `git fetch --deepen origin` widens the checkout's own
 * configured refspec (the PR merge ref / merge-queue ref / tag), and a second fetch widens
 * origin/main — escalating to `--unshallow` before failing CLOSED. It never silently passes when
 * the base cannot be determined, because "cannot see the base" is exactly the condition a
 * deletion would want.
 */

import { execFileSync } from "node:child_process";
import { readFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const SCRIPT_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

export const EVIDENCE_BUNDLE_PATH = "docs/generated/memory-calibration-evidence.json";
export const TOMBSTONES_PATH = "config/memory-evidence-tombstones.json";

// Mirrors `packages/schemas/memory-calibration.schema.json` records[].id.
const RECORD_ID_PATTERN = /^imc-[0-9a-f]{20}$/;
const STORY_REF_PATTERN = /^sc-\d+$/;

function git(root, args) {
  return execFileSync("git", args, {
    cwd: root,
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
  });
}

function tryGit(root, args) {
  try {
    return git(root, args).trim();
  } catch {
    return null;
  }
}

/** Parse the bundle text into its set of record ids, failing on anything malformed. */
export function recordIdSet(text, at) {
  let bundle;
  try {
    bundle = JSON.parse(text);
  } catch (error) {
    throw new Error(`${at}: evidence bundle is not valid JSON (${error.message})`);
  }
  if (!Array.isArray(bundle?.records)) {
    throw new Error(`${at}: evidence bundle has no records array`);
  }
  const ids = new Set();
  for (const [index, record] of bundle.records.entries()) {
    const id = record?.id;
    if (typeof id !== "string" || !RECORD_ID_PATTERN.test(id)) {
      throw new Error(`${at}: records[${index}] id ${JSON.stringify(id ?? null)} does not match ${RECORD_ID_PATTERN}`);
    }
    if (ids.has(id)) throw new Error(`${at}: duplicate record id ${id}`);
    ids.add(id);
  }
  return ids;
}

/** Parse and validate the tombstone file; returns a Map of record id -> entry. */
export function parseTombstones(text, at) {
  let parsed;
  try {
    parsed = JSON.parse(text);
  } catch (error) {
    throw new Error(`${at}: not valid JSON (${error.message})`);
  }
  if (!Array.isArray(parsed?.tombstones)) {
    throw new Error(`${at}: missing tombstones array`);
  }
  const entries = new Map();
  for (const [index, entry] of parsed.tombstones.entries()) {
    const where = `${at}: tombstones[${index}]`;
    if (typeof entry?.id !== "string" || !RECORD_ID_PATTERN.test(entry.id)) {
      throw new Error(`${where} id must match ${RECORD_ID_PATTERN}`);
    }
    if (typeof entry.reason !== "string" || entry.reason.trim() === "") {
      throw new Error(`${where} (${entry.id}) needs a non-empty reason`);
    }
    if (typeof entry.story !== "string" || !STORY_REF_PATTERN.test(entry.story)) {
      throw new Error(`${where} (${entry.id}) story must match ${STORY_REF_PATTERN}`);
    }
    if (entries.has(entry.id)) throw new Error(`${where} duplicates tombstone for ${entry.id}`);
    entries.set(entry.id, entry);
  }
  return entries;
}

/** The pure comparison: base ids vs head ids under the tombstone allowlist. */
export function corpusErrors({ baseIds, headIds, tombstones }) {
  const errors = [];
  for (const id of baseIds) {
    if (!headIds.has(id) && !tombstones.has(id)) {
      errors.push(
        `evidence record ${id} exists at the merge-base but not at HEAD and has no tombstone. ` +
          `Deleting calibration evidence must be an explicit, reviewable act: add an entry to ` +
          `${TOMBSTONES_PATH} with the reason and the authorizing story, or restore the record.`,
      );
    }
  }
  for (const id of tombstones.keys()) {
    if (headIds.has(id)) {
      errors.push(
        `${TOMBSTONES_PATH} declares ${id} deleted, but the record still exists in ` +
          `${EVIDENCE_BUNDLE_PATH}. Remove the premature tombstone or delete the record in the same change.`,
      );
    }
  }
  return errors;
}

/**
 * Find the commit to diff against: the merge-base of HEAD and origin/main.
 *
 * Fast path (full local clones, fetch-depth: 0 checkouts): plain `git merge-base`. Shallow path
 * (both CI lanes that run `npm run check`): fetch origin/main, then deepen the checkout's own
 * ref and origin/main in escalating steps until the histories connect, ending at --unshallow.
 * Fails closed if no base can be determined — a silent pass here would re-open the exact
 * fail-open this gate exists to close.
 */
export function resolveBaseCommit({ root = SCRIPT_ROOT, env = process.env, log = () => {} } = {}) {
  const override = env.SCENEWORKS_EVIDENCE_BASE;
  if (override) {
    const commit = tryGit(root, ["rev-parse", "--verify", `${override}^{commit}`]);
    if (!commit) throw new Error(`SCENEWORKS_EVIDENCE_BASE=${override} does not name a commit`);
    return commit;
  }
  const mainRefspec = "+refs/heads/main:refs/remotes/origin/main";
  const mergeBase = () => tryGit(root, ["merge-base", "HEAD", "origin/main"]);

  let base = mergeBase();
  if (base) return base;

  log(`evidence corpus: origin/main not resolvable locally, fetching it`);
  if (tryGit(root, ["fetch", "-q", "--no-tags", "origin", mainRefspec]) === null) {
    throw new Error(
      "cannot fetch origin/main to determine the evidence-corpus base. " +
        "Fetch it manually or set SCENEWORKS_EVIDENCE_BASE to a base commit.",
    );
  }
  base = mergeBase();
  if (base) return base;

  for (const deepen of [64, 1024]) {
    log(`evidence corpus: histories still disconnected, deepening by ${deepen}`);
    // No refspec: widens whatever remote.origin.fetch is configured to — in a CI checkout that is
    // exactly the ref HEAD was created from (PR merge ref, merge-queue ref, or tag).
    tryGit(root, ["fetch", "-q", "--no-tags", `--deepen=${deepen}`, "origin"]);
    tryGit(root, ["fetch", "-q", "--no-tags", `--deepen=${deepen}`, "origin", mainRefspec]);
    base = mergeBase();
    if (base) return base;
  }

  if (tryGit(root, ["rev-parse", "--is-shallow-repository"]) === "true") {
    log("evidence corpus: unshallowing to find the merge-base");
    tryGit(root, ["fetch", "-q", "--no-tags", "--unshallow", "origin"]);
    if (tryGit(root, ["rev-parse", "--is-shallow-repository"]) === "true") {
      tryGit(root, ["fetch", "-q", "--no-tags", "--unshallow", "origin", mainRefspec]);
    }
    base = mergeBase();
    if (base) return base;
  }

  throw new Error(
    "cannot determine a merge-base between HEAD and origin/main even after unshallowing. " +
      "The evidence-corpus gate fails closed rather than skipping the shrink check; " +
      "set SCENEWORKS_EVIDENCE_BASE to an explicit base commit to override.",
  );
}

/** Full check against a repo checkout. Returns the outcome instead of exiting, for tests. */
export async function runCheck({ root = SCRIPT_ROOT, baseRef = null, env = process.env, log = () => {} } = {}) {
  const base = baseRef
    ? tryGit(root, ["rev-parse", "--verify", `${baseRef}^{commit}`])
    : resolveBaseCommit({ root, env, log });
  if (!base) throw new Error(`--base ${baseRef} does not name a commit`);

  // "Bundle absent at the base" must be CONFIRMED from the base commit's tree, never inferred
  // from a failed `git show` — a read failure (corrupt object, partial clone whose promisor fetch
  // failed, ...) would otherwise degrade to an empty base set and vacuously pass a real deletion.
  // `ls-tree` needs only the tree objects: empty output is a confirmed-absent path, any other
  // failure fails closed.
  const baseEntry = tryGit(root, ["ls-tree", base, "--", EVIDENCE_BUNDLE_PATH]);
  if (baseEntry === null) {
    throw new Error(`cannot read the tree of base commit ${base}; failing closed rather than skipping the shrink check`);
  }
  let baseIds;
  if (baseEntry === "") {
    // Confirmed: the base commit predates the bundle. Nothing to shrink from; growth-only.
    baseIds = new Set();
  } else {
    let baseText;
    try {
      baseText = git(root, ["show", `${base}:${EVIDENCE_BUNDLE_PATH}`]);
    } catch (error) {
      throw new Error(
        `${EVIDENCE_BUNDLE_PATH} exists at base commit ${base} but its blob cannot be read ` +
          `(${error.message.trim().split("\n")[0]}); failing closed rather than treating it as an empty corpus`,
      );
    }
    baseIds = recordIdSet(baseText, `${base.slice(0, 12)}:${EVIDENCE_BUNDLE_PATH}`);
  }
  const headIds = recordIdSet(await readFile(path.join(root, EVIDENCE_BUNDLE_PATH), "utf8"), EVIDENCE_BUNDLE_PATH);
  // The tombstone file itself must exist at HEAD — deleting it would delete the allowlist's audit
  // trail along with the gate's escape hatch, so its absence fails rather than defaulting to empty.
  const tombstones = parseTombstones(await readFile(path.join(root, TOMBSTONES_PATH), "utf8"), TOMBSTONES_PATH);

  return { base, baseIds, headIds, tombstones, errors: corpusErrors({ baseIds, headIds, tombstones }) };
}

/**
 * Prove the gate can fire without a deliberately-broken checkin, using the REAL bundle: delete a
 * real record in memory and require the comparison to red, tombstone it and require green, then
 * point a tombstone at a live record and require red again.
 */
export async function selfTest() {
  const headIds = recordIdSet(await readFile(path.join(SCRIPT_ROOT, EVIDENCE_BUNDLE_PATH), "utf8"), EVIDENCE_BUNDLE_PATH);
  if (headIds.size === 0) throw new Error("self-test: the evidence bundle has no records to exercise the gate with");
  const [victim] = headIds;
  const shrunk = new Set(headIds);
  shrunk.delete(victim);

  const none = new Map();
  const deletion = corpusErrors({ baseIds: headIds, headIds: shrunk, tombstones: none });
  if (deletion.length !== 1 || !deletion[0].includes(victim)) {
    throw new Error(`self-test: deleting ${victim} must produce exactly one error naming it, got ${JSON.stringify(deletion)}`);
  }

  const tombstoned = new Map([[victim, { id: victim, reason: "self-test", story: "sc-18224" }]]);
  const allowed = corpusErrors({ baseIds: headIds, headIds: shrunk, tombstones: tombstoned });
  if (allowed.length !== 0) {
    throw new Error(`self-test: a tombstoned deletion must pass, got ${JSON.stringify(allowed)}`);
  }

  const premature = corpusErrors({ baseIds: headIds, headIds, tombstones: tombstoned });
  if (premature.length !== 1 || !premature[0].includes(victim)) {
    throw new Error(`self-test: a tombstone for a live record must produce exactly one error, got ${JSON.stringify(premature)}`);
  }

  const unchanged = corpusErrors({ baseIds: headIds, headIds, tombstones: none });
  if (unchanged.length !== 0) {
    throw new Error(`self-test: an unchanged corpus must pass, got ${JSON.stringify(unchanged)}`);
  }
}

async function main() {
  const args = process.argv.slice(2);
  if (args.includes("--self-test")) {
    await selfTest();
    console.log("check-evidence-corpus self-test passed");
    return;
  }
  const baseFlag = args.indexOf("--base");
  const baseRef = baseFlag === -1 ? null : args[baseFlag + 1];
  if (baseFlag !== -1 && !baseRef) {
    console.error("--base requires a ref");
    process.exit(2);
  }

  const { base, baseIds, headIds, tombstones, errors } = await runCheck({ baseRef, log: (line) => console.error(line) });
  if (errors.length > 0) {
    for (const error of errors) console.error(`error: ${error}`);
    process.exit(1);
  }
  console.log(
    `evidence corpus ok: ${headIds.size} records at HEAD, ${baseIds.size} at base ${base.slice(0, 12)}, ` +
      `${tombstones.size} tombstone${tombstones.size === 1 ? "" : "s"}`,
  );
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((error) => {
    console.error(`check-evidence-corpus: ${error.message}`);
    process.exit(1);
  });
}
