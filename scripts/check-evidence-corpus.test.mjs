import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  EVIDENCE_BUNDLE_PATH,
  TOMBSTONES_PATH,
  corpusErrors,
  parseTombstones,
  recordIdSet,
  runCheck,
  selfTest,
} from "./check-evidence-corpus.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const id = (seed) => `imc-${seed.repeat(20).slice(0, 20)}`;
const A = id("a");
const B = id("b");
const C = id("c");
const D = id("d");

const bundle = (...ids) => JSON.stringify({ records: ids.map((recordId) => ({ id: recordId })) });
const tombstoneFile = (...entries) => JSON.stringify({ tombstones: entries });
const tombstone = (recordId, overrides = {}) => ({
  id: recordId,
  reason: "captured against a retracted harness fixture",
  story: "sc-18224",
  ...overrides,
});
const tombstoneMap = (...entries) => new Map(entries.map((entry) => [entry.id, entry]));

test("recordIdSet parses ids and rejects malformed bundles", () => {
  assert.deepEqual([...recordIdSet(bundle(A, B), "t")], [A, B]);
  assert.throws(() => recordIdSet("not json", "t"), /not valid JSON/);
  assert.throws(() => recordIdSet("{}", "t"), /no records array/);
  assert.throws(() => recordIdSet(bundle(A, A), "t"), /duplicate record id/);
  assert.throws(() => recordIdSet(JSON.stringify({ records: [{ id: "imc-nothex" }] }), "t"), /does not match/);
  assert.throws(() => recordIdSet(JSON.stringify({ records: [{}] }), "t"), /does not match/);
});

test("parseTombstones enforces id, reason, story, and uniqueness", () => {
  const parsed = parseTombstones(tombstoneFile(tombstone(A)), "t");
  assert.equal(parsed.get(A).story, "sc-18224");
  assert.throws(() => parseTombstones("[]", "t"), /missing tombstones array/);
  assert.throws(() => parseTombstones(tombstoneFile({ ...tombstone(A), id: "bad" }), "t"), /id must match/);
  assert.throws(() => parseTombstones(tombstoneFile(tombstone(A, { reason: "  " })), "t"), /non-empty reason/);
  assert.throws(() => parseTombstones(tombstoneFile(tombstone(A, { story: "18224" })), "t"), /story must match/);
  assert.throws(() => parseTombstones(tombstoneFile(tombstone(A), tombstone(A)), "t"), /duplicates tombstone/);
});

test("corpusErrors: shrink fails, tombstoned shrink passes, growth passes", () => {
  const baseIds = new Set([A, B, C]);
  const shrunk = new Set([A, C]);

  const untombstoned = corpusErrors({ baseIds, headIds: shrunk, tombstones: new Map() });
  assert.equal(untombstoned.length, 1);
  assert.match(untombstoned[0], new RegExp(`${B}.*no tombstone`));

  assert.deepEqual(corpusErrors({ baseIds, headIds: shrunk, tombstones: tombstoneMap(tombstone(B)) }), []);
  assert.deepEqual(corpusErrors({ baseIds, headIds: new Set([A, B, C, D]), tombstones: new Map() }), []);
});

test("corpusErrors: a tombstone for a live record fails; a historical tombstone does not", () => {
  const headIds = new Set([A, B]);
  const live = corpusErrors({ baseIds: headIds, headIds, tombstones: tombstoneMap(tombstone(B)) });
  assert.equal(live.length, 1);
  assert.match(live[0], new RegExp(`${B}.*still exists`));

  // D was deleted long ago: absent from base AND head. Its tombstone is history, not an error.
  assert.deepEqual(corpusErrors({ baseIds: headIds, headIds, tombstones: tombstoneMap(tombstone(D)) }), []);
});

test("self-test exercises the gate against the real checked-in bundle", async () => {
  await selfTest();
});

// ---------------------------------------------------------------------------
// End-to-end: real git repos, including the shallow-clone shape CI runs with.
// ---------------------------------------------------------------------------

function run(cwd, command, args) {
  return execFileSync(command, args, { cwd, encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] });
}

function gitIn(cwd, ...args) {
  return run(cwd, "git", ["-c", "user.name=test", "-c", "user.email=test@example.invalid", ...args]);
}

async function writeCorpus(repo, { ids, tombstones = [] }) {
  await writeFile(path.join(repo, EVIDENCE_BUNDLE_PATH), bundle(...ids));
  await writeFile(path.join(repo, TOMBSTONES_PATH), tombstoneFile(...tombstones));
}

/**
 * Upstream history: main carries records A,B,C; branch `deletes-b` deletes B with no
 * tombstone; branch `tombstones-b` deletes B WITH a tombstone.
 */
async function scaffoldUpstream(t) {
  const dir = await mkdtemp(path.join(os.tmpdir(), "evidence-corpus-e2e-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  const upstream = path.join(dir, "upstream");
  gitIn(dir, "init", "-q", "-b", "main", upstream);
  await mkdir(path.join(upstream, path.dirname(EVIDENCE_BUNDLE_PATH)), { recursive: true });
  await mkdir(path.join(upstream, path.dirname(TOMBSTONES_PATH)), { recursive: true });
  await writeCorpus(upstream, { ids: [A, B, C] });
  gitIn(upstream, "add", "-A");
  gitIn(upstream, "commit", "-q", "-m", "seed corpus");

  gitIn(upstream, "checkout", "-q", "-b", "deletes-b");
  await writeCorpus(upstream, { ids: [A, C] });
  gitIn(upstream, "commit", "-qam", "delete B without a tombstone");

  gitIn(upstream, "checkout", "-q", "main");
  gitIn(upstream, "checkout", "-q", "-b", "tombstones-b");
  await writeCorpus(upstream, { ids: [A, C], tombstones: [tombstone(B)] });
  gitIn(upstream, "commit", "-qam", "delete B with a tombstone");
  gitIn(upstream, "checkout", "-q", "main");
  return { dir, upstream };
}

test("e2e: full clone — untombstoned deletion reds, tombstoned deletion greens", async (t) => {
  const { dir, upstream } = await scaffoldUpstream(t);

  const red = path.join(dir, "red");
  gitIn(dir, "clone", "-q", upstream, red);
  gitIn(red, "checkout", "-q", "deletes-b");
  const failing = await runCheck({ root: red });
  assert.equal(failing.errors.length, 1);
  assert.match(failing.errors[0], new RegExp(`${B}.*no tombstone`));
  assert.deepEqual([...failing.baseIds], [A, B, C]);
  assert.deepEqual([...failing.headIds], [A, C]);

  const green = path.join(dir, "green");
  gitIn(dir, "clone", "-q", upstream, green);
  gitIn(green, "checkout", "-q", "tombstones-b");
  assert.deepEqual((await runCheck({ root: green })).errors, []);
});

test("e2e: an uncommitted working-tree deletion reds locally", async (t) => {
  const { dir, upstream } = await scaffoldUpstream(t);
  const work = path.join(dir, "work");
  gitIn(dir, "clone", "-q", upstream, work);
  gitIn(work, "checkout", "-q", "-b", "wip");
  await writeCorpus(work, { ids: [A, C] });
  const { errors } = await runCheck({ root: work });
  assert.equal(errors.length, 1);
  assert.match(errors[0], new RegExp(`${B}.*no tombstone`));
});

test("e2e: on main itself the gate is inert after a tombstoned deletion merges", async (t) => {
  const { dir, upstream } = await scaffoldUpstream(t);
  gitIn(upstream, "merge", "-q", "--no-ff", "-m", "merge tombstoned deletion", "tombstones-b");
  const work = path.join(dir, "post-merge");
  gitIn(dir, "clone", "-q", upstream, work);
  const outcome = await runCheck({ root: work });
  assert.deepEqual(outcome.errors, []);
  // merge-base of main with itself is HEAD: base and head sets are identical.
  assert.deepEqual([...outcome.baseIds], [...outcome.headIds]);
});

test("e2e: shallow single-branch clone (the CI checkout shape) fetches its own base and reds", async (t) => {
  const { dir, upstream } = await scaffoldUpstream(t);
  const shallow = path.join(dir, "shallow");
  // Mirrors actions/checkout fetch-depth: 1 — HEAD has no parents locally, origin/main is not
  // fetched at all, and remote.origin.fetch covers only the checked-out ref.
  gitIn(dir, "clone", "-q", "--depth=1", "--single-branch", "--branch", "deletes-b", `file://${upstream}`, shallow);
  assert.equal(gitIn(shallow, "rev-parse", "--is-shallow-repository").trim(), "true");

  const fetches = [];
  const { errors } = await runCheck({ root: shallow, log: (line) => fetches.push(line) });
  assert.equal(errors.length, 1);
  assert.match(errors[0], new RegExp(`${B}.*no tombstone`));
  assert.ok(fetches.length > 0, "the shallow path must have had to fetch origin/main");
});

test("e2e: SCENEWORKS_EVIDENCE_BASE overrides base resolution and fails closed on garbage", async (t) => {
  const { dir, upstream } = await scaffoldUpstream(t);
  const work = path.join(dir, "override");
  gitIn(dir, "clone", "-q", upstream, work);
  gitIn(work, "checkout", "-q", "deletes-b");

  // Overriding the base to the deleting commit itself makes base == head: no shrink visible.
  const pinned = await runCheck({ root: work, env: { SCENEWORKS_EVIDENCE_BASE: "HEAD" } });
  assert.deepEqual(pinned.errors, []);

  await assert.rejects(
    runCheck({ root: work, env: { SCENEWORKS_EVIDENCE_BASE: "not-a-ref" } }),
    /does not name a commit/,
  );
});

test("e2e: an unreadable base blob fails closed instead of passing vacuously", async (t) => {
  const { dir, upstream } = await scaffoldUpstream(t);
  const work = path.join(dir, "unreadable");
  // fetch.unpackLimit forces the transferred pack to explode into loose objects so a single
  // blob can be removed — the shape of a partial clone whose promisor fetch fails at read time.
  gitIn(dir, "init", "-q", "-b", "deletes-b", work);
  gitIn(work, "remote", "add", "origin", upstream);
  gitIn(work, "-c", "fetch.unpackLimit=1000000", "fetch", "-q", "origin");
  gitIn(work, "checkout", "-q", "-b", "deletes-b", "origin/deletes-b");

  const blob = gitIn(work, "rev-parse", "origin/main:" + EVIDENCE_BUNDLE_PATH).trim();
  const object = path.join(work, ".git", "objects", blob.slice(0, 2), blob.slice(2));
  await rm(object);

  // The deletion of B is real and untombstoned; with the base blob unreadable the gate must
  // throw, never degrade to an empty base set and report zero errors.
  await assert.rejects(runCheck({ root: work }), /cannot be read.*failing closed/s);
});

test("e2e: a base commit that predates the bundle is confirmed-absent and growth-only", async (t) => {
  const dir = await mkdtemp(path.join(os.tmpdir(), "evidence-corpus-e2e-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  const upstream = path.join(dir, "upstream");
  gitIn(dir, "init", "-q", "-b", "main", upstream);
  await writeFile(path.join(upstream, "README.md"), "pre-bundle era\n");
  gitIn(upstream, "add", "-A");
  gitIn(upstream, "commit", "-q", "-m", "no bundle yet");
  gitIn(upstream, "checkout", "-q", "-b", "introduces-bundle");
  await mkdir(path.join(upstream, path.dirname(EVIDENCE_BUNDLE_PATH)), { recursive: true });
  await mkdir(path.join(upstream, path.dirname(TOMBSTONES_PATH)), { recursive: true });
  await writeCorpus(upstream, { ids: [A, B] });
  gitIn(upstream, "add", "-A");
  gitIn(upstream, "commit", "-q", "-m", "introduce bundle");
  gitIn(upstream, "checkout", "-q", "main");

  const work = path.join(dir, "work");
  gitIn(dir, "clone", "-q", upstream, work);
  gitIn(work, "checkout", "-q", "introduces-bundle");
  const outcome = await runCheck({ root: work });
  assert.deepEqual(outcome.errors, []);
  assert.equal(outcome.baseIds.size, 0);
  assert.deepEqual([...outcome.headIds], [A, B]);
});

test("this suite is wired into npm run check", async () => {
  const { scripts } = JSON.parse(await readFile(path.join(ROOT, "package.json"), "utf8"));
  // sc-19758 unwired the GATE. `npm run check` used to run `check-evidence-corpus.mjs --self-test`
  // and then the live corpus check; it is now the unit tests alone, so the assertion that pinned
  // that self-test-then-live sequence is gone with it. The script is still on disk and still works
  // when run deliberately — `node scripts/check-evidence-corpus.mjs`.
  //
  // This suite stays wired, and that half is still worth pinning: the tests are cheap, they grade
  // the corpus logic rather than the pin, and dropping them from the chain would be an accident
  // rather than a decision.
  assert.match(scripts.check, /scripts\/check-evidence-corpus\.test\.mjs/);
});
