// The inference checkout a pin bump derives from (scripts/bump-inference.mjs).
//
// Without `--repo` the bump shallow-fetches inference into a temp dir, and it used to leave that dir
// behind on every run: 19 leaked `bump-inference-*` dirs held 63 GB of the dev Mac's disk on
// 2026-09-16. These drive the real helper with a fake fetch, so they need no network and no
// inference clone.
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { directInferenceCrates, parseCliArgs, withInferenceCheckout } from "./bump-inference.mjs";

const SCRIPT = resolve(dirname(fileURLToPath(import.meta.url)), "bump-inference.mjs");
const SHA = "a".repeat(40);

// Stands in for the shallow fetch: records where it was asked to clone and leaves clone-shaped
// content there, including a read-only pack file like the ones git writes. `fail` models a fetch
// that dies mid-transfer, after the partial clone is already on disk.
function fakeFetch(calls, { fail = false } = {}) {
  return (sha, dir) => {
    calls.push({ sha, dir });
    mkdirSync(join(dir, ".git/objects/pack"), { recursive: true });
    writeFileSync(join(dir, ".git/objects/pack/pack-0.pack"), "pack", { mode: 0o444 });
    if (fail) throw new Error("fetch died mid-transfer");
    writeFileSync(join(dir, "Cargo.toml"), "[workspace]\n");
  };
}

// A regression would leak into the real temp dir; remove whatever the fake fetch was handed.
const removeFetched = (t, calls) =>
  t.after(() => {
    for (const { dir } of calls) rmSync(dir, { recursive: true, force: true });
  });

test("a fetched checkout is removed once the steps that read it return", (t) => {
  const calls = [];
  removeFetched(t, calls);
  let seen = null;
  const result = withInferenceCheckout(
    SHA,
    null,
    (repo) => {
      seen = repo;
      assert.ok(existsSync(join(repo, "Cargo.toml")), "the steps run against the fetched clone");
      return "derived";
    },
    { fetch: fakeFetch(calls) },
  );
  assert.equal(result, "derived");
  assert.equal(calls.length, 1);
  assert.equal(calls[0].sha, SHA);
  assert.equal(seen, calls[0].dir);
  assert.equal(existsSync(calls[0].dir), false, `the temp clone survived: ${calls[0].dir}`);
});

test("a fetched checkout is removed when a later step throws", (t) => {
  const calls = [];
  removeFetched(t, calls);
  assert.throws(
    () =>
      withInferenceCheckout(
        SHA,
        null,
        () => {
          throw new Error("cargo update failed");
        },
        { fetch: fakeFetch(calls) },
      ),
    /cargo update failed/,
  );
  assert.equal(calls.length, 1);
  assert.equal(existsSync(calls[0].dir), false, `the temp clone survived: ${calls[0].dir}`);
});

test("a partial clone is removed when the fetch itself fails", (t) => {
  const calls = [];
  removeFetched(t, calls);
  let used = false;
  assert.throws(
    () =>
      withInferenceCheckout(
        SHA,
        null,
        () => {
          used = true;
        },
        { fetch: fakeFetch(calls, { fail: true }) },
      ),
    /fetch died mid-transfer/,
  );
  assert.equal(used, false, "no step runs against a failed fetch");
  assert.equal(calls.length, 1);
  assert.equal(existsSync(calls[0].dir), false, `the partial clone survived: ${calls[0].dir}`);
});

test("an explicit --repo is used as-is and never deleted, even when a step throws", (t) => {
  const repo = mkdtempSync(join(tmpdir(), "caller-inference-repo-"));
  t.after(() => rmSync(repo, { recursive: true, force: true }));
  writeFileSync(join(repo, "Cargo.toml"), "[workspace]\n");
  const calls = [];
  removeFetched(t, calls);

  let seen = null;
  const result = withInferenceCheckout(
    SHA,
    repo,
    (checkout) => {
      seen = checkout;
      return "derived";
    },
    { fetch: fakeFetch(calls) },
  );
  assert.equal(result, "derived");
  assert.equal(seen, repo);
  assert.ok(existsSync(join(repo, "Cargo.toml")), `the caller's --repo was deleted: ${repo}`);

  assert.throws(
    () =>
      withInferenceCheckout(
        SHA,
        repo,
        () => {
          throw new Error("verifyNoSkew failed");
        },
        { fetch: fakeFetch(calls) },
      ),
    /verifyNoSkew failed/,
  );
  assert.ok(existsSync(join(repo, "Cargo.toml")), `the caller's --repo was deleted: ${repo}`);
  assert.deepEqual(calls, [], "an explicit --repo is never replaced by a fetch");
});

// The import guard that lets this file load the module must still let a direct run through. A bad
// SHA exits before anything is read, written or fetched.
test("running the script directly still reaches its CLI", () => {
  const run = spawnSync(process.execPath, [SCRIPT, "--sha", "not-a-sha"], { encoding: "utf8" });
  assert.equal(run.status, 2, `${run.stdout}${run.stderr}`);
  assert.match(run.stderr, /not a 40-char commit SHA: not-a-sha/);
});

// The CLI guard. A `--help` that falls through does not merely print the wrong thing: with no
// `--sha` it resolves the latest inference SHA over the network, fetches a clone, rewrites three
// manifests and runs `cargo update`. Each case runs the real CLI with a TMPDIR of its own and
// asserts nothing was cloned into it, which is what proves the guard ran BEFORE the fetch.
function runCli(t, argv) {
  const box = mkdtempSync(join(tmpdir(), "bump-cli-tmpdir-"));
  t.after(() => rmSync(box, { recursive: true, force: true }));
  const result = spawnSync(process.execPath, [SCRIPT, ...argv], {
    encoding: "utf8",
    cwd: resolve(dirname(SCRIPT), ".."),
    env: { ...process.env, TMPDIR: box },
  });
  return { result, leaked: readdirSync(box) };
}

test("--help prints usage and bumps nothing", (t) => {
  const { result, leaked } = runCli(t, ["--help"]);
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /usage: node scripts\/bump-inference\.mjs/);
  assert.match(result.stdout, /--sha <sha40>/);
  assert.deepEqual(leaked, [], "a help invocation must not fetch a checkout");
});

test("-h is the same door", (t) => {
  const { result, leaked } = runCli(t, ["-h"]);
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /usage: node scripts\/bump-inference\.mjs/);
  assert.deepEqual(leaked, []);
});

test("a mistyped flag is refused instead of bumping to the latest SHA", (t) => {
  const { result, leaked } = runCli(t, ["--halp"]);
  assert.equal(result.status, 2);
  assert.match(result.stderr, /unexpected argument: --halp/);
  assert.match(result.stderr, /usage: node scripts\/bump-inference\.mjs/);
  assert.deepEqual(leaked, [], "a refused argument must not fetch a checkout");
});

test("a SHA passed positionally is refused, not silently replaced by the latest", (t) => {
  const { result, leaked } = runCli(t, ["b".repeat(40)]);
  assert.equal(result.status, 2);
  assert.match(result.stderr, /unexpected argument: b{40}/);
  assert.deepEqual(leaked, []);
});

test("the flags the bump really takes still parse", () => {
  for (const argv of [
    [],
    ["--dry-run"],
    ["--sha", "c".repeat(40)],
    ["--repo", "/tmp/inference"],
    ["--sha", "c".repeat(40), "--repo", "/tmp/inference", "--dry-run", "--no-deps"],
    ["--self-test"],
  ]) {
    assert.deepEqual(parseCliArgs(argv), { help: false }, argv.join(" "));
  }
  // A value that looks like a flag is still a value, never an unknown argument.
  assert.deepEqual(parseCliArgs(["--repo", "--dry-run"]), { help: false });
});

// sc-26691: inference sc-27252 split gen-core into sibling crates the workspace now depends on
// directly. The bump must refresh every one of them, so the update set is discovered from the
// manifests rather than read from a list that rots with the next split.
test("directInferenceCrates discovers every direct inference dependency, not the Candle patch", () => {
  const git = '{ git = "https://github.com/SceneWorks/inference", rev = "' + SHA + '" }';
  const worker = [
    "[dependencies]",
    "sceneworks-gen-core = " + git,
    "sceneworks-gen-core-iris = " + git,
    'serde = { version = "1.0", features = ["derive"] }',
    'other = { git = "https://github.com/someone/else", rev = "' + SHA + '" }',
    "",
    "[target.'cfg(target_os = \"macos\")'.dependencies]",
    'mlx-gen-ltx = { git = "https://github.com/SceneWorks/inference", rev = "' + SHA + '", optional = true }',
  ].join("\n");
  const root = ['[patch."https://github.com/huggingface/candle"]', "candle-core = " + git].join("\n");
  const core = ["[dependencies]", "sceneworks-gen-core-checkpoint = " + git, "sceneworks-gen-core-iris = " + git].join("\n");
  assert.deepEqual(directInferenceCrates([root, worker, core]), [
    "mlx-gen-ltx",
    "sceneworks-gen-core",
    "sceneworks-gen-core-checkpoint",
    "sceneworks-gen-core-iris",
  ]);
});
