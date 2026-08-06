// Unit tests for the per-provider closure digest (sc-17774).
//
// These run WITHOUT an inference clone — CI has none (check-license-coverage.mjs:481) — so every
// test drives the module through its injected `runGit` against a synthetic two-provider workspace.
//
// The headline property is the one the epic exists for and it is tested in both directions:
// a change to one provider's code path must move THAT provider's digest and no other's, and a
// change to a shared crate must move everyone who links it. A unit that absolved everything would
// be indistinguishable from a broken one, so the "must move" cases carry as much weight here as the
// "must hold" ones.
import assert from "node:assert/strict";
import test from "node:test";
import { createHash } from "node:crypto";

import {
  assertProviderIsDeclaredInCrate,
  closureText,
  digestsAtRevision,
  lockedClosure,
  manifestPathDependencies,
  parseCargoLock,
  providerClosureDigest,
  rootManifestSlices,
  treeBlobs,
  workspaceDependencyPaths,
} from "./inference-closure-digest.mjs";

const oid = (body) => createHash("sha1").update(body).digest("hex");

/**
 * A synthetic inference-shaped workspace.
 *
 * `crates/a` deliberately HOSTS `crates/a/a-sub`, mirroring `crates/media/mlx-gen` hosting
 * `mlx-gen-qwen-image`: a prefix match rather than a per-crate `src/` match would drag the sibling
 * in and silently restore the cross-model coupling this module removes.
 */
function workspace(overrides = {}) {
  const files = {
    "Cargo.toml": [
      "[workspace]",
      'members = ["crates/*"]',
      "",
      "[workspace.dependencies]",
      'shared = { path = "crates/shared" }',
      'unused-elsewhere = { path = "crates/unused" }',
      "",
      "[profile.release]",
      "lto = true",
      "",
      "[patch.crates-io]",
      'serde = { git = "https://example.invalid/serde" }',
    ].join("\n"),
    "Cargo.lock": [
      "version = 4",
      "",
      "[[package]]",
      'name = "a"',
      'version = "0.0.0"',
      "dependencies = [",
      ' "shared",',
      ' "only-a-uses-me",',
      "]",
      "",
      "[[package]]",
      'name = "b"',
      'version = "0.0.0"',
      "dependencies = [",
      ' "shared",',
      "]",
      "",
      "[[package]]",
      'name = "shared"',
      'version = "0.0.0"',
      "",
      "[[package]]",
      'name = "only-a-uses-me"',
      'version = "1.0.0"',
      'source = "registry+https://github.com/rust-lang/crates.io-index"',
      'checksum = "aaaa"',
      "",
      "[[package]]",
      'name = "nobody-uses-me"',
      'version = "9.9.9"',
      'source = "registry+https://github.com/rust-lang/crates.io-index"',
      'checksum = "zzzz"',
    ].join("\n"),
    "rust-toolchain.toml": '[toolchain]\nchannel = "1.96.0"\n',
    ".cargo/config.toml": "[build]\nrustflags = []\n",

    "crates/shared/Cargo.toml": '[package]\nname = "shared"\n',
    "crates/shared/src/lib.rs": "pub fn shared() {}\n",

    "crates/a/Cargo.toml": [
      "[package.metadata.legacy-workspace-dependencies]",
      'decoy = { path = "../decoy" }',
      "",
      "[package]",
      'name = "a"',
      "",
      "[dependencies]",
      "shared = { workspace = true }",
      'only-a-uses-me = "1"',
      "",
      "[dev-dependencies]",
      'devonly = { path = "../devonly" }',
    ].join("\n"),
    "crates/a/src/lib.rs": 'pub const ID: &str = "provider_a";\n',
    "crates/a/tests/it.rs": "// excluded from the closure\n",

    "crates/a/a-sub/Cargo.toml": '[package]\nname = "a-sub"\n',
    "crates/a/a-sub/src/lib.rs": "pub fn sub() {}\n",

    "crates/b/Cargo.toml": [
      "[package]",
      'name = "b"',
      "",
      "[dependencies]",
      "shared = { workspace = true }",
    ].join("\n"),
    "crates/b/src/lib.rs": 'pub const ID: &str = "provider_b";\n',

    "crates/decoy/Cargo.toml": '[package]\nname = "decoy"\n',
    "crates/decoy/src/lib.rs": "pub fn decoy() {}\n",
    "crates/devonly/Cargo.toml": '[package]\nname = "devonly"\n',
    "crates/devonly/src/lib.rs": "pub fn devonly() {}\n",
    "crates/unused/Cargo.toml": '[package]\nname = "unused-elsewhere"\n',
    "crates/unused/src/lib.rs": "pub fn unused() {}\n",
    ...overrides,
  };
  return files;
}

/** A `runGit` stub over one or more named revisions. */
function fakeGit(revisions) {
  return (_repo, args) => {
    const [command] = args;
    if (command === "rev-parse") {
      const rev = args[1].replace(/\^\{commit\}$/, "");
      if (!revisions[rev]) throw new Error(`unknown revision ${rev}`);
      return `${rev}\n`;
    }
    if (command === "ls-tree") {
      const rev = args[args.length - 1];
      const files = revisions[rev];
      if (!files) throw new Error(`unknown revision ${rev}`);
      return Object.entries(files)
        .map(([file, body]) => `100644 blob ${oid(body)}\t${file}`)
        .join("\n");
    }
    if (command === "show") {
      const [rev, file] = args[1].split(/:(.*)/s);
      const body = revisions[rev]?.[file];
      if (body === undefined) throw new Error(`no ${file} at ${rev}`);
      return body;
    }
    if (command === "grep") {
      const rev = args[args.indexOf("--") - 1];
      const needle = args[args.indexOf("-F") + 1];
      const prefix = args[args.indexOf("--") + 1];
      const hits = Object.entries(revisions[rev] ?? {})
        .filter(([file, body]) => file.startsWith(prefix) && body.includes(needle))
        .map(([file]) => `${rev}:${file}`);
      if (!hits.length) throw new Error("git grep: no match");
      return hits.join("\n");
    }
    throw new Error(`unexpected git ${command}`);
  };
}

const PROVIDERS = { provider_a: "crates/a", provider_b: "crates/b" };

/** `oid -> body` over every revision's fixture, standing in for `git cat-file --batch`. */
function fakeBodies(revisions) {
  const byOid = new Map();
  for (const files of Object.values(revisions)) {
    for (const body of Object.values(files)) byOid.set(oid(body), body);
  }
  return (oids) => new Map(oids.map((id) => [id, byOid.get(id)]));
}

function digestsFor(revisions, revision = "r1") {
  return digestsAtRevision({
    repo: "/fake",
    revision,
    providers: PROVIDERS,
    readBodies: fakeBodies(revisions),
    runGit: fakeGit(revisions),
  });
}

test("a provider's closure is its crate plus what it links, and nothing else", () => {
  const digests = digestsFor({ r1: workspace() });
  assert.deepEqual(digests.get("provider_a").crates, ["crates/a", "crates/shared"]);
  assert.deepEqual(digests.get("provider_b").crates, ["crates/b", "crates/shared"]);
});

test("a `workspace = true` dependency is resolved through the root manifest", () => {
  // The regression this pins: reading only inline `path = "…"` drops `sceneworks-gen-core`, which
  // owns the memory-strategy contract. Its absence would be a false green on the shared contract.
  const paths = workspaceDependencyPaths(workspace()["Cargo.toml"]);
  assert.equal(paths.get("shared"), "crates/shared");
  assert.ok(digestsFor({ r1: workspace() }).get("provider_a").crates.includes("crates/shared"));
});

test("`[package.metadata.legacy-workspace-dependencies]` is inert and never walked", () => {
  // It is shaped exactly like a real dependency table and is not one.
  assert.ok(!digestsFor({ r1: workspace() }).get("provider_a").crates.includes("crates/decoy"));
});

test("dev-dependencies are outside the closure, build- and target-dependencies are inside", () => {
  const manifest = [
    "[dependencies]",
    'plain = { path = "../plain" }',
    "[build-dependencies]",
    'built = { path = "../built" }',
    '[target."cfg(target_os = \\"macos\\")".dependencies]',
    'mac = { path = "../mac" }',
    "[dev-dependencies]",
    'dev = { path = "../dev" }',
    '[target."cfg(unix)".dev-dependencies]',
    'unixdev = { path = "../unixdev" }',
  ].join("\n");
  assert.deepEqual(manifestPathDependencies(manifest, "crates/x"), [
    "crates/built",
    "crates/mac",
    "crates/plain",
  ]);
});

test("a crate that HOSTS sibling crates does not absorb them", () => {
  // `crates/a/a-sub` sits under `crates/a`. A prefix match would pull it in.
  const digest = providerClosureDigest({
    repo: "/fake",
    revision: "r1",
    provider: "provider_a",
    crateDir: "crates/a",
    readBodies: fakeBodies({ r1: workspace() }),
    runGit: fakeGit({ r1: workspace() }),
  });
  assert.ok(!digest.text.includes("crates/a/a-sub"));
});

test("tests/ and benches/ are outside the digested source", () => {
  const digest = providerClosureDigest({
    repo: "/fake",
    revision: "r1",
    provider: "provider_a",
    crateDir: "crates/a",
    readBodies: fakeBodies({ r1: workspace() }),
    runGit: fakeGit({ r1: workspace() }),
  });
  assert.ok(digest.text.includes("crates/a/src/lib.rs"));
  assert.ok(!digest.text.includes("crates/a/tests/it.rs"));
});

test("the locked set is restricted to packages the closure reaches", () => {
  const lock = parseCargoLock(workspace()["Cargo.lock"]);
  const reached = lockedClosure(lock, ["a", "shared"]).map((entry) => entry.name);
  assert.deepEqual(reached.sort(), ["a", "only-a-uses-me", "shared"]);
  assert.ok(!reached.includes("nobody-uses-me"));
});

test("an unrelated dependency bump in Cargo.lock does not move a digest", () => {
  // The whole lock file moves on nearly every inference commit. Only the reached slice is hashed.
  const bumped = workspace({
    "Cargo.lock": workspace()["Cargo.lock"].replace('version = "9.9.9"', 'version = "9.9.10"'),
  });
  const before = digestsFor({ r1: workspace() });
  const after = digestsFor({ r1: bumped });
  for (const provider of Object.keys(PROVIDERS)) {
    assert.equal(before.get(provider).digest, after.get(provider).digest, provider);
  }
});

test("a dependency bump the closure DOES reach moves only that closure", () => {
  const bumped = workspace({
    "Cargo.lock": workspace()["Cargo.lock"].replace('version = "1.0.0"', 'version = "1.0.1"'),
  });
  const before = digestsFor({ r1: workspace() });
  const after = digestsFor({ r1: bumped });
  assert.notEqual(before.get("provider_a").digest, after.get("provider_a").digest);
  assert.equal(before.get("provider_b").digest, after.get("provider_b").digest);
});

test("THE HEADLINE: one provider's source change never moves another's digest", () => {
  const edited = workspace({ "crates/a/src/lib.rs": 'pub const ID: &str = "provider_a"; // edit\n' });
  const before = digestsFor({ r1: workspace() });
  const after = digestsFor({ r1: edited });
  assert.notEqual(
    before.get("provider_a").digest,
    after.get("provider_a").digest,
    "a provider's own source change MUST move its digest — a unit that absolves everything is broken",
  );
  assert.equal(
    before.get("provider_b").digest,
    after.get("provider_b").digest,
    "provider_b does not compile crates/a and must not be demoted by it",
  );
});

test("a shared-crate change moves every provider that links it", () => {
  const edited = workspace({ "crates/shared/src/lib.rs": "pub fn shared() { /* edit */ }\n" });
  const before = digestsFor({ r1: workspace() });
  const after = digestsFor({ r1: edited });
  for (const provider of Object.keys(PROVIDERS)) {
    assert.notEqual(before.get(provider).digest, after.get(provider).digest, provider);
  }
});

test("a crate NO provider links cannot move any digest", () => {
  const edited = workspace({ "crates/unused/src/lib.rs": "pub fn unused() { /* edit */ }\n" });
  const before = digestsFor({ r1: workspace() });
  const after = digestsFor({ r1: edited });
  for (const provider of Object.keys(PROVIDERS)) {
    assert.equal(before.get(provider).digest, after.get(provider).digest, provider);
  }
});

test("a closure digest is independent of the revision it was read at", () => {
  // The first draft hashed the revision into the digest text, which reconstructed pin identity
  // exactly: every provider moved on every bump while the source was byte-identical.
  const files = workspace();
  const digests = digestsFor({ r1: files, r2: files }, "r1");
  const later = digestsFor({ r1: files, r2: files }, "r2");
  for (const provider of Object.keys(PROVIDERS)) {
    assert.equal(digests.get(provider).digest, later.get(provider).digest, provider);
  }
});

test("workspace build inputs are in every closure", () => {
  // A SEMANTIC edit, not a comment — comment churn in these files is absolved (see the test below),
  // so appending `# edit` would assert the opposite of the intended policy.
  for (const [file, edit] of [
    ["rust-toolchain.toml", '[toolchain]\nchannel = "1.97.0"\n'],
    [".cargo/config.toml", '[build]\nrustflags = ["-Ctarget-cpu=native"]\n'],
  ]) {
    const edited = workspace({ [file]: edit });
    const before = digestsFor({ r1: workspace() });
    const after = digestsFor({ r1: edited });
    for (const provider of Object.keys(PROVIDERS)) {
      assert.notEqual(before.get(provider).digest, after.get(provider).digest, `${file} / ${provider}`);
    }
  }
});

test("a missing workspace build input is a hard failure, not a quietly smaller digest", () => {
  const files = workspace();
  delete files["rust-toolchain.toml"];
  assert.throws(() => digestsFor({ r1: files }), /workspace build input/);
});

test("the root manifest contributes [profile] and [patch], not [workspace.dependencies]", () => {
  const slices = rootManifestSlices(workspace()["Cargo.toml"]).join("\n");
  assert.match(slices, /\[profile\.release\]/);
  assert.match(slices, /\[patch\.crates-io\]/);
  assert.ok(!slices.includes("workspace.dependencies"));

  // Adding a workspace dependency no provider links must not move anything; the lock already
  // carries the versions that matter.
  const edited = workspace({
    "Cargo.toml": workspace()["Cargo.toml"].replace(
      "[profile.release]",
      'brand-new = { path = "crates/brand-new" }\n\n[profile.release]',
    ),
  });
  const before = digestsFor({ r1: workspace() });
  const after = digestsFor({ r1: edited });
  for (const provider of Object.keys(PROVIDERS)) {
    assert.equal(before.get(provider).digest, after.get(provider).digest, provider);
  }
});

test("a profile change moves every digest", () => {
  const edited = workspace({
    "Cargo.toml": workspace()["Cargo.toml"].replace("lto = true", "lto = false"),
  });
  const before = digestsFor({ r1: workspace() });
  const after = digestsFor({ r1: edited });
  for (const provider of Object.keys(PROVIDERS)) {
    assert.notEqual(before.get(provider).digest, after.get(provider).digest, provider);
  }
});

test("a declaration pointing at a crate that never names the provider is refused", () => {
  const runGit = fakeGit({ r1: workspace() });
  assert.doesNotThrow(() =>
    assertProviderIsDeclaredInCrate({
      repo: "/fake",
      revision: "r1",
      provider: "provider_a",
      crateDir: "crates/a",
      runGit,
    }),
  );
  assert.throws(
    () =>
      assertProviderIsDeclaredInCrate({
        repo: "/fake",
        revision: "r1",
        provider: "provider_a",
        crateDir: "crates/b",
        runGit,
      }),
    /never names it/,
  );
});

test("two providers in the SAME crate get distinct digests", () => {
  // `flux1_dev` and `flux1_schnell` share `candle-gen-flux`. The provider is part of the digest
  // identity so a digest can never be compared across providers by accident.
  const runGit = fakeGit({ r1: workspace() });
  const readBodies = fakeBodies({ r1: workspace() });
  const one = providerClosureDigest({ repo: "/fake", revision: "r1", provider: "p1", crateDir: "crates/a", readBodies, runGit });
  const two = providerClosureDigest({ repo: "/fake", revision: "r1", provider: "p2", crateDir: "crates/a", readBodies, runGit });
  assert.notEqual(one.digest, two.digest);
});

test("treeBlobs refuses an empty revision instead of digesting nothing", () => {
  assert.throws(() => treeBlobs("/fake", "empty", { runGit: () => "" }), /listed no blobs/);
});

test("closureText is stable under input ordering", () => {
  const base = {
    provider: "p",
    crateDir: "crates/a",
    crates: ["crates/a", "crates/shared"],
    files: [["crates/a/src/lib.rs", "aaa"], ["crates/shared/src/lib.rs", "bbb"]],
    locked: [{ name: "x", version: "1", checksum: "c" }],
    rootSlices: ["[profile.release]\nlto = true"],
    workspace: [["rust-toolchain.toml", "ddd"]],
  };
  assert.equal(closureText(base), closureText({ ...base }));
});

test("a comment-only edit to closure source does not move the digest", () => {
  // This is the capability the deleted flux2-only artifact audit existed to provide, now uniform
  // across every provider. sc-17775 §5.4 measured the vendored kernel tree moving three times in 60
  // days, all documentation.
  const edited = workspace({
    "crates/a/src/lib.rs": '// a new explanatory comment\npub const ID: &str = "provider_a";\n',
  });
  const before = digestsFor({ r1: workspace() });
  const after = digestsFor({ r1: edited });
  assert.equal(before.get("provider_a").digest, after.get("provider_a").digest);
});

test("the quote guard keeps a comment carrying a string literal hashed", () => {
  // A commented-out line with a literal could be a real change; the stripper errs noisy-but-safe.
  const edited = workspace({
    "crates/a/src/lib.rs": '// pub const ID: &str = "was_provider_a";\npub const ID: &str = "provider_a";\n',
  });
  const before = digestsFor({ r1: workspace() });
  const after = digestsFor({ r1: edited });
  assert.notEqual(before.get("provider_a").digest, after.get("provider_a").digest);
});

test("a code change that only LOOKS like a comment still moves the digest", () => {
  const edited = workspace({ "crates/a/src/lib.rs": 'pub const ID: &str = "provider_a2";\n' });
  const before = digestsFor({ r1: workspace() });
  const after = digestsFor({ r1: edited });
  assert.notEqual(before.get("provider_a").digest, after.get("provider_a").digest);
});

test("a language the stripper does not understand is hashed as raw bytes", () => {
  // `.metal` / `.cu` kernels reach the closure through `src/`. Stripping a language this module does
  // not model is how a real change becomes invisible, so those are hashed verbatim.
  const base = workspace({ "crates/a/src/kernel.metal": "// kernel\nkernel void go() {}\n" });
  const edited = workspace({ "crates/a/src/kernel.metal": "// kernel v2\nkernel void go() {}\n" });
  assert.notEqual(
    digestsFor({ r1: base }).get("provider_a").digest,
    digestsFor({ r1: edited }).get("provider_a").digest,
  );
});

test("comment-only churn in a workspace build input is absolved too", () => {
  const edited = workspace({
    "rust-toolchain.toml": '# why this channel\n[toolchain]\nchannel = "1.96.0"\n',
  });
  const before = digestsFor({ r1: workspace() });
  const after = digestsFor({ r1: edited });
  for (const provider of Object.keys(PROVIDERS)) {
    assert.equal(before.get(provider).digest, after.get(provider).digest, provider);
  }
});
