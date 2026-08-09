import assert from "node:assert/strict";
import test from "node:test";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

// sc-18227: the `node --test` list in `npm run check` is hand-maintained, and this class of bug has
// recurred — a new `*.test.mjs` lands on disk, is never added to the list, and runs nowhere
// (`check-docker-api-runtime.test.mjs` sat orphaned until this story). This meta-test diffs the
// on-disk `*.test.mjs` files against the lists that actually execute them, so the next orphan reds
// `npm run check` instead of silently never running.

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

// Deliberate exclusions from every runner, path -> reason. Every entry must name a real file and
// carry a non-empty reason — no silent exclusions. Currently empty on purpose: everything runs.
const EXCLUDED = new Map([]);

// Directory names never descended into. `node_modules`/`.git`/`target`/`dist` are dependency or
// build output; `.claude` holds agent worktrees (full repo checkouts) whose files would double-count.
const SKIPPED_DIRS = new Set(["node_modules", ".git", "target", "dist", ".claude"]);

function onDiskTestFiles() {
  const found = [];
  const walk = (dir) => {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      // Dirent reports symlinks as symlinks, not directories, so a symlinked
      // node_modules (worktree setups) is skipped twice over.
      if (entry.isDirectory()) {
        if (!SKIPPED_DIRS.has(entry.name)) walk(path.join(dir, entry.name));
      } else if (entry.isFile() && entry.name.endsWith(".test.mjs")) {
        found.push(path.relative(repoRoot, path.join(dir, entry.name)));
      }
    }
  };
  walk(repoRoot);
  return found.sort();
}

function npmScript(pkgRelativeDir, name) {
  const pkg = JSON.parse(
    readFileSync(path.join(repoRoot, pkgRelativeDir, "package.json"), "utf8"),
  );
  const script = pkg.scripts?.[name];
  assert.ok(script, `${pkgRelativeDir || "."}/package.json has a "${name}" script`);
  return script;
}

function testFileTokens(shellCommand, pkgRelativeDir) {
  const segments = shellCommand
    .split("&&")
    .map((segment) => segment.trim())
    .filter((segment) => segment.startsWith("node --test"));
  assert.ok(
    segments.length > 0,
    `expected a "node --test" segment in: ${shellCommand}`,
  );
  return segments.flatMap((segment) =>
    segment
      .split(/\s+/)
      .filter((token) => token.endsWith(".test.mjs"))
      .map((token) => path.join(pkgRelativeDir, token)),
  );
}

function executedTestFiles() {
  return [
    // The hand-maintained list this story is about.
    ...testFileTokens(npmScript(".", "check"), "."),
    // Desktop scripts run through their own suite (desktop-linux/-windows workflows and the
    // desktop-*-check lanes' `npm --prefix apps/desktop test` / `desktop:check`).
    ...testFileTokens(npmScript("apps/desktop", "test"), "apps/desktop"),
  ];
}

test("every on-disk *.test.mjs is executed by npm run check or the desktop suite", () => {
  const onDisk = onDiskTestFiles();
  // A broken walk (wrong root, extension-filter drift, an over-broad SKIPPED_DIRS entry) would
  // return nothing and make the orphan diff below pass vacuously. This file is on disk by
  // definition, so the walk must find it.
  assert.ok(
    onDisk.includes(path.join("scripts", "test-list-coverage.test.mjs")),
    "the discovery walk failed to find this very file — the walk is broken, not the repo clean",
  );
  const executed = new Set(executedTestFiles());
  const orphans = onDisk.filter(
    (file) => !executed.has(file) && !EXCLUDED.has(file),
  );
  assert.deepEqual(
    orphans,
    [],
    "orphaned test files run nowhere — add them to the `node --test` list in package.json " +
      '`scripts.check` (or apps/desktop `scripts.test`), or to EXCLUDED in this file with a reason',
  );
});

test("every listed test file exists on disk", () => {
  const stale = executedTestFiles().filter(
    (file) => !existsSync(path.join(repoRoot, file)),
  );
  assert.deepEqual(stale, [], "package.json lists test files that no longer exist");
});

test("every exclusion names a real file, carries a reason, and is not also listed", () => {
  const executed = new Set(executedTestFiles());
  for (const [file, reason] of EXCLUDED) {
    assert.ok(
      typeof reason === "string" && reason.trim().length > 0,
      `EXCLUDED entry ${file} needs a non-empty reason`,
    );
    assert.ok(
      existsSync(path.join(repoRoot, file)),
      `EXCLUDED entry ${file} does not exist on disk — remove the stale entry`,
    );
    assert.ok(
      !executed.has(file),
      `EXCLUDED entry ${file} is also in a runner list — it is not excluded; remove the entry`,
    );
  }
});

test("this meta-test is itself in the npm run check list", () => {
  assert.ok(
    testFileTokens(npmScript(".", "check"), ".").includes(
      path.join(".", "scripts/test-list-coverage.test.mjs"),
    ),
    "scripts/test-list-coverage.test.mjs must run in `npm run check` for this guard to guard anything",
  );
});
