import assert from "node:assert/strict";
import { execFileSync, spawnSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, extname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

import {
  SOURCE_BASENAMES,
  SOURCE_EXTENSIONS,
  findUnexpectedControlBytes,
  isTrackedSourcePath,
} from "./check-source-control-bytes.mjs";

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(SCRIPT_DIR, "..");
const CHECKER = join(SCRIPT_DIR, "check-source-control-bytes.mjs");

const REQUIRED_TEXT_EXTENSIONS = [
  ".mjs",
  ".cjs",
  ".mts",
  ".ts",
  ".tsx",
  ".js",
  ".jsx",
  ".rs",
  ".json",
  ".jsonc",
  ".md",
  ".toml",
  ".yml",
  ".yaml",
];

const DECLARED_BINARY_EXTENSIONS = [
  ".bin",
  ".ckpt",
  ".gguf",
  ".gif",
  ".gz",
  ".icns",
  ".ico",
  ".jpeg",
  ".jpg",
  ".log",
  ".mov",
  ".mp4",
  ".onnx",
  ".otf",
  ".pdf",
  ".png",
  ".pt",
  ".rgb",
  ".safetensors",
  ".ttf",
  ".webm",
  ".webp",
  ".woff",
  ".woff2",
  ".zip",
];

function git(cwd, ...args) {
  return execFileSync("git", args, { cwd, encoding: "utf8" });
}

/**
 * `git init` with git's background work disabled.
 *
 * Committing can fork `gc --auto` (and, on newer git, `maintenance`), which writes into `.git`
 * asynchronously. If that lands while the teardown below is walking the tree, `rmSync` fails with
 * `ENOTEMPTY: rmdir '<tmp>/.git'` — a flake seen on `main`'s Check run at `8e4a5683e`. This is a
 * REQUIRED check, so when it fires it blocks the merge queue for everyone.
 *
 * Disabling the trigger is the actual fix; `removeRepo` retries as a second line for anything else
 * touching the directory (indexer, antivirus, a slow unlink). Both fixtures go through these two
 * helpers so a third cannot quietly reintroduce the race by calling `git(repo, "init")` directly.
 */
function initRepo(repo) {
  git(repo, "init", "--quiet");
  git(repo, "config", "gc.auto", "0");
  git(repo, "config", "maintenance.auto", "false");
  return repo;
}

/** Teardown that tolerates a straggler writing into the tree mid-walk. */
function removeRepo(repo) {
  rmSync(repo, { force: true, recursive: true, maxRetries: 10, retryDelay: 50 });
}

test("allows TAB, LF, and CR but reports every other C0 byte by offset", () => {
  const contents = Buffer.from([0x09, 0x0a, 0x0d, 0x00, 0x01, 0x1f, 0x20]);
  assert.deepEqual(findUnexpectedControlBytes(contents), [
    { byte: 0x00, offset: 3 },
    { byte: 0x01, offset: 4 },
    { byte: 0x1f, offset: 5 },
  ]);
});

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

test("explicit text attributes cover every path selected by the scanner", () => {
  const attributes = readFileSync(join(REPO_ROOT, ".gitattributes"), "utf8");
  const assertTextDiffPattern = (pattern) => {
    assert.match(
      attributes,
      new RegExp(`^${escapeRegExp(pattern)}\\s+text\\s+diff$`, "m"),
      `${pattern} must have explicit text and diff attributes`,
    );
  };

  for (const extension of SOURCE_EXTENSIONS) {
    assertTextDiffPattern(`*${extension}`);
  }
  for (const name of SOURCE_BASENAMES) {
    assertTextDiffPattern(name);
  }
  for (const extension of REQUIRED_TEXT_EXTENSIONS) {
    assert.equal(SOURCE_EXTENSIONS.has(extension), true, `${extension} must be scanned`);
  }
});

test("every currently tracked path is selected source or a declared binary format", () => {
  const binaryExtensions = new Set(DECLARED_BINARY_EXTENSIONS);
  const tracked = git(REPO_ROOT, "ls-files", "-z").split("\0").filter(Boolean);

  for (const file of tracked) {
    assert.equal(
      isTrackedSourcePath(file) || binaryExtensions.has(extname(file)),
      true,
      `${file} must be classified as source or declared binary`,
    );
  }
});

test("CLI ignores declared binary assets and rejects a tracked source NUL", (t) => {
  const repo = mkdtempSync(join(tmpdir(), "sceneworks-control-bytes-"));
  t.after(() => removeRepo(repo));

  initRepo(repo);
  writeFileSync(join(repo, "clean.mjs"), 'const separator = "\\t";\n');
  for (const extension of DECLARED_BINARY_EXTENSIONS) {
    writeFileSync(join(repo, `asset${extension}`), Buffer.from([0x00, 0x01, 0x02]));
  }
  git(repo, "add", ".");

  const clean = spawnSync(process.execPath, [CHECKER], { cwd: repo, encoding: "utf8" });
  assert.equal(clean.status, 0, clean.stderr);

  writeFileSync(join(repo, "clean.mjs"), Buffer.from('const separator = "\0";\n'));
  const mutated = spawnSync(process.execPath, [CHECKER], {
    cwd: repo,
    encoding: "utf8",
  });
  assert.equal(mutated.status, 1);
  assert.match(mutated.stderr, /clean\.mjs: byte offset 19: 0x00/);
  assert.match(mutated.stderr, /TAB \(0x09\), LF \(0x0a\), and CR \(0x0d\) only/);
});

test("explicit text attributes keep a NUL-bearing source diff textual", (t) => {
  const repo = mkdtempSync(join(tmpdir(), "sceneworks-text-attribute-"));
  t.after(() => removeRepo(repo));

  writeFileSync(join(repo, ".gitattributes"), readFileSync(join(REPO_ROOT, ".gitattributes")));
  writeFileSync(join(repo, "probe.mjs"), 'export const value = "clean";\n');
  initRepo(repo);
  git(repo, "config", "user.email", "codex@example.invalid");
  git(repo, "config", "user.name", "Codex");
  git(repo, "add", ".");
  git(repo, "commit", "--quiet", "-m", "baseline");

  writeFileSync(join(repo, "probe.mjs"), Buffer.from('export const value = "dirty\0value";\n'));
  const numstat = git(repo, "diff", "--numstat", "--", "probe.mjs");
  const patch = git(repo, "diff", "--", "probe.mjs");

  assert.doesNotMatch(numstat, /^-\t-\t/m);
  assert.doesNotMatch(patch, /Binary files/);
  assert.match(patch, /^@@/m);
});
