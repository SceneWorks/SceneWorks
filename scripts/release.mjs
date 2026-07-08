#!/usr/bin/env node
// Cut a desktop release in the two steps this repo actually uses. `main` is
// branch-protected (direct pushes are rejected — see the GH013 rule), so the
// version bump can't be pushed straight to main; it has to land through a PR
// first, then the tag is applied to the merged commit. That's how every prior
// release tag landed (v0.4.0 = "chore(release): 0.4.0 (#984)" on main, then
// tagged). This script wraps both phases:
//
//   Phase 1 — bump (open a release PR):
//     npm run release -- minor            # 0.5.1 -> 0.6.0 on a release/ branch + PR
//     npm run release -- patch|major|prerelease | <x.y.z>
//   ...review + merge that PR to main...
//
//   Phase 2 — tag (draft the release):
//     git switch main && git pull         # get the merged bump commit
//     npm run release -- tag              # tag main -> pushes v<x> -> release.yml
//
// Phase 1 runs `npm version <bump> --no-git-tag-version` (bumps root
// package.json + the `version` hook syncs tauri.conf.json + apps/desktop +
// apps/web), moves the change onto a `release/v<x>` branch, commits, pushes,
// and opens the PR. NO tag yet — the tag must point at the *merged* commit,
// whose SHA doesn't exist until the PR lands.
//
// Phase 2 reads the version from the (now merged) package.json, tags main
// `v<x>`, and pushes just the tag. The tag push triggers release.yml, which
// signs/notarizes and creates the DRAFT GitHub Release. Nothing is published
// until you review the draft and hit Publish.
//
// Flags:
//   --yes         skip the confirmation prompt (tag phase / push)
//   --any-branch  bump from / tag a branch other than main
//   --no-pr       phase 1: create the branch + commit + push, but don't open the PR
//   --watch       phase 2: stream the release workflow run via `gh`
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
const subject = positional[0];

const KNOWN_FLAGS = new Set(["--yes", "--any-branch", "--no-pr", "--watch"]);
for (const f of flags) if (!KNOWN_FLAGS.has(f)) fail(`unknown flag: ${f}`);

const VALID_KEYWORDS = new Set([
  "major",
  "minor",
  "patch",
  "premajor",
  "preminor",
  "prepatch",
  "prerelease",
]);
const SEMVER = /^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$/;

if (!subject) {
  fail(
    "usage:\n" +
      "  npm run release -- minor|patch|major|prerelease | <x.y.z>   (phase 1: open release PR)\n" +
      "  npm run release -- tag                                       (phase 2: tag main -> draft release)",
  );
}

if (subject === "tag") {
  await tagPhase();
} else if (VALID_KEYWORDS.has(subject) || SEMVER.test(subject)) {
  await bumpPhase(subject);
} else {
  fail(
    `"${subject}" is not a valid bump or the "tag" subcommand.\n` +
      `  bump: ${[...VALID_KEYWORDS].join(", ")} or an explicit semver (0.6.0 / 0.6.0-rc.1)\n` +
      "  tag:  run `npm run release -- tag` after the release PR merges",
  );
}

// ---- phase 1: bump on a release branch + open a PR ------------------------
async function bumpPhase(bump) {
  const branch = requireCleanBranch();
  ensureUpToDate(branch);

  const currentVersion = readRootVersion();
  console.log(`release: current version ${currentVersion}, bumping "${bump}"…`);

  // Bump + sync every version-bearing file, but do NOT commit or tag: the tag
  // has to wait for the merged commit (phase 2).
  run("npm", ["version", bump, "--no-git-tag-version"], repoRoot);
  const newVersion = readRootVersion();
  if (newVersion === currentVersion) {
    // Restore and bail rather than leave the tree dirty.
    run("git", ["checkout", "--", "."], repoRoot);
    run("git", ["reset", "--quiet"], repoRoot);
    fail(`version did not change (still ${currentVersion}); nothing to do.`);
  }

  const relBranch = `release/v${newVersion}`;
  const tag = `v${newVersion}`;
  if (tagExists(tag)) {
    run("git", ["checkout", "--", "."], repoRoot);
    run("git", ["reset", "--quiet"], repoRoot);
    fail(`tag ${tag} already exists — is ${newVersion} already released?`);
  }

  // Carry the staged bump onto a fresh release branch and commit it there,
  // leaving the base branch clean.
  run("git", ["switch", "-c", relBranch], repoRoot);
  run("git", ["commit", "-a", "-m", `chore(release): ${newVersion}`], repoRoot);
  console.log(`release: committed bump on ${relBranch}.`);

  if (flags.has("--no-pr")) {
    console.log(
      "\nrelease: --no-pr set. Push + open the PR yourself:\n" +
        `  git push -u origin ${relBranch}\n` +
        `  gh pr create --base main --head ${relBranch} --title "chore(release): ${newVersion}"\n` +
        `Then after it merges: git switch main && git pull && npm run release -- tag`,
    );
    return;
  }

  run("git", ["push", "-u", "origin", relBranch], repoRoot);
  const body =
    `Version bump to \`${newVersion}\` (root + tauri.conf.json + apps/desktop + apps/web via sync-version.mjs).\n\n` +
    "After this merges, cut the release:\n" +
    "```\ngit switch main && git pull\nnpm run release -- tag\n```\n" +
    "The tag push triggers `.github/workflows/release.yml`, which drafts the GitHub Release.";
  run(
    "gh",
    [
      "pr",
      "create",
      "--base",
      "main",
      "--head",
      relBranch,
      "--title",
      `chore(release): ${newVersion}`,
      "--body",
      body,
    ],
    repoRoot,
  );

  console.log(
    `\nrelease: opened the ${newVersion} bump PR. Review + merge it, then:\n` +
      "  git switch main && git pull\n" +
      "  npm run release -- tag",
  );
}

// ---- phase 2: tag the merged commit -> draft the release ------------------
async function tagPhase() {
  const branch = requireCleanBranch();
  ensureUpToDate(branch);

  const version = readRootVersion();
  const tag = `v${version}`;
  if (tagExists(tag)) {
    fail(
      `tag ${tag} already exists. If ${version} is already released, bump again first; ` +
        "otherwise delete the stale tag before retrying.",
    );
  }

  // Sanity: the version-bump commit should already be on main. If HEAD's
  // package.json version equals the latest release tag's version, the bump PR
  // probably hasn't merged yet.
  const latest = latestReleaseVersion();
  if (latest && latest === version) {
    fail(
      `package.json is still ${version}, which matches the latest tag v${latest}. ` +
        "Has the release bump PR merged into main yet?",
    );
  }

  console.log(`release: about to tag ${branch} @ ${shortHead()} as ${tag}.`);
  if (!(await confirm(`Tag + push ${tag}? This drafts a public release. [y/N] `))) {
    console.log("release: not tagged.");
    return;
  }

  run("git", ["tag", "-a", tag, "-m", tag], repoRoot);
  run("git", ["push", "origin", tag], repoRoot);

  const [owner, repo] = originSlug();
  console.log(
    `\nrelease: ${tag} pushed. CI is drafting the release now:\n` +
      `  https://github.com/${owner}/${repo}/actions/workflows/release.yml\n` +
      "When it finishes, review the draft and hit Publish:\n" +
      `  https://github.com/${owner}/${repo}/releases`,
  );

  if (flags.has("--watch")) {
    try {
      run(
        "gh",
        ["run", "watch", "--exit-status", "--workflow", "release.yml"],
        repoRoot,
      );
    } catch {
      console.warn("release: `gh run watch` failed — watch it in the browser.");
    }
  }
}

// ---- shared preconditions -------------------------------------------------
function requireCleanBranch() {
  const branch = git("rev-parse", "--abbrev-ref", "HEAD");
  if (branch === "HEAD") fail("detached HEAD — check out a branch first.");
  if (branch !== "main" && !flags.has("--any-branch")) {
    fail(
      `on branch "${branch}", not main. Releases are cut from main. ` +
        "Re-run with --any-branch to override.",
    );
  }
  const dirty = git("status", "--porcelain");
  if (dirty) {
    fail(
      "working tree is not clean — commit or stash first:\n" +
        dirty.split("\n").map((l) => "  " + l).join("\n"),
    );
  }
  return branch;
}

function ensureUpToDate(branch) {
  try {
    git("fetch", "--quiet", "origin", branch);
    const behind = git("rev-list", "--count", `HEAD..origin/${branch}`);
    if (behind !== "0") {
      fail(
        `local ${branch} is ${behind} commit(s) behind origin/${branch}. ` +
          "Pull first so the release points at the latest commit.",
      );
    }
  } catch {
    console.warn(
      `release: couldn't compare against origin/${branch} (offline or no upstream?) — skipping behind-check.`,
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

function tagExists(tag) {
  return git("tag", "--list", tag) === tag;
}

function shortHead() {
  return git("rev-parse", "--short", "HEAD");
}

function latestReleaseVersion() {
  // Newest v* tag by creation date, minus the leading "v". Empty if none.
  const out = git("tag", "--list", "v*", "--sort=-creatordate");
  const first = out.split("\n")[0]?.trim();
  return first ? first.replace(/^v/, "") : "";
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

async function confirm(prompt) {
  if (flags.has("--yes")) return true;
  const rl = createInterface({ input: process.stdin, output: process.stdout });
  const answer = (await rl.question(`\n${prompt}`)).trim().toLowerCase();
  rl.close();
  return answer === "y" || answer === "yes";
}

function fail(msg) {
  console.error(`release: ${msg}`);
  process.exit(1);
}
