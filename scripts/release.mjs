#!/usr/bin/env node
// One command to cut a desktop release. Wraps the two steps that are easy to
// forget (see .github/workflows/release.yml + scripts/sync-version.mjs):
//
//   1. `npm version <bump>` at the repo root — bumps root package.json, the
//      `version` lifecycle hook runs sync-version.mjs (tauri.conf.json +
//      apps/desktop + apps/web move in lockstep), and npm makes ONE commit +
//      an annotated tag `v<version>`.
//   2. `git push` the branch AND the tag — the `v*` tag push triggers
//      release.yml, which builds/signs/notarizes/staples and creates the
//      DRAFT GitHub Release (macOS DMG + Windows installers + updater
//      latest.json). Nothing is published until you review the draft and hit
//      "Publish" in the GitHub UI.
//
// Usage:
//   npm run release -- patch            # 0.5.1 -> 0.5.2
//   npm run release -- minor            # 0.5.1 -> 0.6.0
//   npm run release -- major            # 0.5.1 -> 1.0.0
//   npm run release -- 0.6.0-rc.1       # explicit version; `-` => CI marks it prerelease
//   npm run release -- prerelease       # 0.5.1 -> 0.5.2-0
//
// Flags:
//   --yes         skip the confirmation prompt before pushing
//   --no-push     bump + commit + tag locally, then STOP (push yourself later)
//   --any-branch  allow releasing from a branch other than main
//   --watch       after pushing, stream the release workflow run via `gh`
//
// Pushing the tag kicks off a real signed/notarized build and drafts a public
// release, so the script pauses for confirmation before the push unless --yes.
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { createInterface } from "node:readline/promises";

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");

// ---- args -----------------------------------------------------------------
const argv = process.argv.slice(2);
const flags = new Set(argv.filter((a) => a.startsWith("--")));
const positional = argv.filter((a) => !a.startsWith("--"));
const bump = positional[0];

const KNOWN_FLAGS = new Set([
  "--yes",
  "--no-push",
  "--any-branch",
  "--watch",
]);
for (const f of flags) {
  if (!KNOWN_FLAGS.has(f)) fail(`unknown flag: ${f}`);
}

const VALID_KEYWORDS = new Set([
  "major",
  "minor",
  "patch",
  "premajor",
  "preminor",
  "prepatch",
  "prerelease",
]);
if (!bump) {
  fail(
    "missing version bump.\n" +
      "  npm run release -- patch|minor|major|prerelease   (or an explicit x.y.z)",
  );
}
// Accept an npm keyword or an explicit semver (with optional -prerelease / +build).
const isKeyword = VALID_KEYWORDS.has(bump);
const isExplicit = /^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$/.test(
  bump,
);
if (!isKeyword && !isExplicit) {
  fail(
    `"${bump}" is not a valid bump.\n` +
      `  keywords: ${[...VALID_KEYWORDS].join(", ")}\n` +
      "  or an explicit semver like 0.6.0 / 0.6.0-rc.1",
  );
}

// ---- preconditions --------------------------------------------------------
const branch = git("rev-parse", "--abbrev-ref", "HEAD");
if (branch === "HEAD") {
  fail("detached HEAD — check out a branch before releasing.");
}
if (branch !== "main" && !flags.has("--any-branch")) {
  fail(
    `on branch "${branch}", not main. Releases are normally cut from main.\n` +
      "  Re-run with --any-branch to override.",
  );
}

// npm version refuses on a dirty tree, but fail early with a clearer message.
const dirty = git("status", "--porcelain");
if (dirty) {
  fail(
    "working tree is not clean — commit or stash first:\n" +
      dirty
        .split("\n")
        .map((l) => "  " + l)
        .join("\n"),
  );
}

// Make sure we're not tagging a stale HEAD behind the remote.
try {
  git("fetch", "--quiet", "origin", branch);
  const behind = git("rev-list", "--count", `HEAD..origin/${branch}`);
  if (behind !== "0") {
    fail(
      `local ${branch} is ${behind} commit(s) behind origin/${branch}. ` +
        "Pull first so the tag points at the latest commit.",
    );
  }
} catch {
  // No upstream / offline — sync-check is best-effort, keep going.
  console.warn(
    `release: couldn't compare against origin/${branch} (offline or no upstream?) — skipping behind-check.`,
  );
}

const currentVersion = readRootVersion();

// ---- bump: npm version does the commit + tag ------------------------------
console.log(`release: current version ${currentVersion}, bumping "${bump}"…`);
// `npm version` prints the new tag (e.g. "v0.5.2") and creates the commit + tag.
run("npm", ["version", bump], repoRoot);

const newVersion = readRootVersion();
const tag = `v${newVersion}`;
if (newVersion === currentVersion) {
  fail(
    `version did not change (still ${currentVersion}); nothing tagged. Aborting before push.`,
  );
}
console.log(`release: committed + tagged ${tag}.`);

// ---- push (the part that actually drafts the release) ---------------------
if (flags.has("--no-push")) {
  console.log(
    "\nrelease: --no-push set. When ready, run:\n" +
      `  git push origin ${branch} && git push origin ${tag}\n` +
      "The tag push triggers .github/workflows/release.yml (creates the DRAFT release).",
  );
  process.exit(0);
}

if (!flags.has("--yes")) {
  const rl = createInterface({ input: process.stdin, output: process.stdout });
  const answer = (
    await rl.question(
      `\nPush ${branch} + ${tag} to origin? This starts a signed build and drafts a public release. [y/N] `,
    )
  )
    .trim()
    .toLowerCase();
  rl.close();
  if (answer !== "y" && answer !== "yes") {
    console.log(
      "release: not pushed. The bump commit + tag exist locally. To undo:\n" +
        `  git tag -d ${tag} && git reset --hard HEAD~1\n` +
        "Or push later:\n" +
        `  git push origin ${branch} && git push origin ${tag}`,
    );
    process.exit(0);
  }
}

console.log(`release: pushing ${branch}…`);
run("git", ["push", "origin", branch], repoRoot);
console.log(`release: pushing ${tag}…`);
run("git", ["push", "origin", tag], repoRoot);

const [owner, repo] = originSlug();
const actionsUrl = `https://github.com/${owner}/${repo}/actions/workflows/release.yml`;
console.log(
  `\nrelease: ${tag} pushed. CI is building the DRAFT release now:\n  ${actionsUrl}\n` +
    "When it finishes, review the draft and hit Publish:\n" +
    `  https://github.com/${owner}/${repo}/releases`,
);

if (flags.has("--watch")) {
  try {
    // Give the tag-push event a moment to register the run, then stream it.
    run("gh", ["run", "watch", "--exit-status", "--workflow", "release.yml"], repoRoot);
  } catch {
    console.warn(
      "release: `gh run watch` unavailable or failed — watch it in the browser instead.",
    );
  }
}

// ---- helpers --------------------------------------------------------------
function git(...args) {
  return execFileSync("git", args, { cwd: repoRoot, encoding: "utf8" }).trim();
}

function run(cmd, args, cwd) {
  execFileSync(cmd, args, { cwd, stdio: "inherit" });
}

function readRootVersion() {
  const v = JSON.parse(
    readFileSync(join(repoRoot, "package.json"), "utf8"),
  ).version;
  if (!v) fail("no version field in root package.json");
  return v;
}

function originSlug() {
  const url = git("remote", "get-url", "origin");
  const m = url.match(/github\.com[:/]([^/]+)\/(.+?)(?:\.git)?$/);
  return m ? [m[1], m[2]] : ["SceneWorks", "SceneWorks"];
}

function fail(msg) {
  console.error(`release: ${msg}`);
  process.exit(1);
}
