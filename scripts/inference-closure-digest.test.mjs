// Unit tests for the per-provider closure digest (sc-17774).
//
// These run WITHOUT an inference clone, so they stay fast and hermetic on every lane: each test
// drives the module through its injected `runGit`/`readBodies` against a synthetic two-provider
// workspace. The REAL derivation is graded separately, by `check.yml` re-deriving the shipped
// digests against a shallow fetch of the pinned inference revision.
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
  patchTargetPaths,
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
      // A UNIVERSAL crate, at its real repository path so the production `UNIVERSAL_CRATES` constant
      // is what these tests exercise — not a test-only injection that could drift from it.
      'gen-core = { path = "crates/contracts/gen-core" }',
      "",
      "[profile.release]",
      "lto = true",
      "",
      "[patch.crates-io]",
      'serde = { git = "https://example.invalid/serde" }',
      "",
      // The sc-17935 shape: inference reaches its vendored CUDA kernels ONLY through this table.
      // `patched` is reached by `a` (through `only-a-uses-me`) and by nobody else; `unreachable-patch`
      // hangs off `nobody-uses-me`, so it is a local patch target no provider compiles.
      '[patch."https://example.invalid/upstream"]',
      'patched = { path = "crates/vendor/patched" }',
      'unreachable-patch = { path = "crates/vendor/unreached" }',
    ].join("\n"),
    "Cargo.lock": [
      "version = 4",
      "",
      "[[package]]",
      'name = "a"',
      'version = "0.0.0"',
      "dependencies = [",
      ' "shared",',
      ' "gen-core",',
      ' "only-a-uses-me",',
      "]",
      "",
      "[[package]]",
      'name = "b"',
      'version = "0.0.0"',
      "dependencies = [",
      ' "shared",',
      ' "gen-core",',
      "]",
      "",
      "[[package]]",
      'name = "shared"',
      'version = "0.0.0"',
      "",
      "[[package]]",
      'name = "gen-core"',
      'version = "0.0.0"',
      "",
      "[[package]]",
      'name = "only-a-uses-me"',
      'version = "1.0.0"',
      'source = "registry+https://github.com/rust-lang/crates.io-index"',
      'checksum = "aaaa"',
      "dependencies = [",
      ' "patched",',
      "]",
      "",
      // A patched package's lock entry carries no source and no checksum — its identity is the local
      // tree, which is exactly what nothing hashed before sc-17935.
      "[[package]]",
      'name = "patched"',
      'version = "0.10.2"',
      "",
      "[[package]]",
      'name = "nobody-uses-me"',
      'version = "9.9.9"',
      'source = "registry+https://github.com/rust-lang/crates.io-index"',
      'checksum = "zzzz"',
      "dependencies = [",
      ' "unreachable-patch",',
      "]",
      "",
      "[[package]]",
      'name = "unreachable-patch"',
      'version = "0.0.1"',
    ].join("\n"),
    "rust-toolchain.toml": '[toolchain]\nchannel = "1.96.0"\n',
    ".cargo/config.toml": "[build]\nrustflags = []\n",

    "crates/contracts/gen-core/Cargo.toml": '[package]\nname = "gen-core"\n',
    "crates/contracts/gen-core/src/lib.rs": "pub fn contract() {}\n",

    "crates/shared/Cargo.toml": '[package]\nname = "shared"\n',
    "crates/shared/src/lib.rs": "pub fn shared() {}\n",
    // An ORDINARY closure crate's build script — sc-17935 item 2 is "never hashed, for ANY crate",
    // so it has to be pinned somewhere other than the patch target.
    "crates/shared/build.rs": "fn main() { println!(\"cargo:rustc-cfg=shared\"); }\n",

    "crates/a/Cargo.toml": [
      "[package.metadata.legacy-workspace-dependencies]",
      'decoy = { path = "../decoy" }',
      "",
      "[package]",
      'name = "a"',
      "",
      "[dependencies]",
      "shared = { workspace = true }",
      "gen-core = { workspace = true }",
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
      "gen-core = { workspace = true }",
    ].join("\n"),
    "crates/b/src/lib.rs": 'pub const ID: &str = "provider_b";\n',

    // The vendored-kernel shape: a build script at the crate ROOT that compiles a non-Rust tree
    // under `src/`, plus a README that is not a compile input.
    "crates/vendor/patched/Cargo.toml": '[package]\nname = "patched"\n',
    "crates/vendor/patched/build.rs": "fn main() { compile_kernels(); }\n",
    "crates/vendor/patched/src/lib.rs": "pub fn patched() {}\n",
    "crates/vendor/patched/src/kernel.cu": "__global__ void k() { /* v1 */ }\n",
    "crates/vendor/patched/README.md": "# vendored\n",

    "crates/vendor/unreached/Cargo.toml": '[package]\nname = "unreachable-patch"\n',
    "crates/vendor/unreached/build.rs": "fn main() {}\n",
    "crates/vendor/unreached/src/lib.rs": "pub fn unreached() {}\n",

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
  // `crates/vendor/patched` is linked, just not through a dependency edge — it replaces a package
  // the lock says `a` compiles. Nothing declares it as a `path =` dependency (sc-17935 item 1).
  // The universal crate IS walked and IS listed — v4 removes its CONTENT from the currency digest,
  // not the crate from the closure. See `UNIVERSAL_CRATES`.
  assert.deepEqual(digests.get("provider_a").crates, [
    "crates/a", "crates/contracts/gen-core", "crates/shared", "crates/vendor/patched",
  ]);
  assert.deepEqual(digests.get("provider_b").crates, [
    "crates/b", "crates/contracts/gen-core", "crates/shared",
  ]);
  assert.deepEqual(digests.get("provider_a").universalCrates, ["crates/contracts/gen-core"]);
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
  assert.deepEqual(reached.sort(), ["a", "gen-core", "only-a-uses-me", "patched", "shared"]);
  assert.ok(!reached.includes("nobody-uses-me"));
  assert.ok(!reached.includes("unreachable-patch"));
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

test("THE v4 HEADLINE: a universal crate's source change moves NO provider's currency digest", () => {
  // The defect this exists to prevent, and it is not hypothetical — it has come back eight times.
  //
  // `core-llm` and `gen-core` are in ALL 17 real closures, so under v3 any edit to either moved
  // every provider at once, which is repository-wide pin identity wearing a per-provider digest's
  // clothes. Measured on the real f32fce06 -> 857f2454 bump: 17 of 17 providers demoted, every one
  // attributable solely to `core-llm`, whose entire diff was a new `starvector.rs` plus
  // `pub mod starvector;` and its re-exports. Adding a NEW model invalidated the measured memory of
  // every EXISTING model. Under v4 that same bump demotes 3, and each of those links a genuinely
  // changed model crate.
  const edited = workspace({
    "crates/contracts/gen-core/src/lib.rs": "pub fn contract() { /* semantic edit */ }\n",
  });
  const before = digestsFor({ r1: workspace() });
  const after = digestsFor({ r1: edited });
  for (const provider of Object.keys(PROVIDERS)) {
    assert.equal(
      before.get(provider).digest,
      after.get(provider).digest,
      `${provider}: a universal crate cannot carry per-provider currency`,
    );
    // The signal is not lost, only separated: it is recorded where a reviewer sees it in the diff.
    assert.notEqual(
      before.get(provider).sharedContractDigest,
      after.get(provider).sharedContractDigest,
      `${provider}: the shared-contract change must still be visible`,
    );
  }
});

test("a universal crate's files never enter the digested source", () => {
  // Mutation guard for the test above: if the exclusion were implemented by something that happened
  // to hold on this fixture rather than by scoping `[source]`, this fails.
  const text = providerClosureDigest({
    repo: "/fake", revision: "r1", provider: "provider_a", crateDir: "crates/a",
    readBodies: fakeBodies({ r1: workspace() }),
    runGit: fakeGit({ r1: workspace() }),
  }).text;
  const source = text.split("[source]")[1].split("[workspace]")[0];
  assert.ok(!source.includes("crates/contracts/gen-core"), source);
  assert.ok(source.includes("crates/a/src/lib.rs"), source);
  // Listed by name, so ADDING or DROPPING a universal crate is still a structural digest change —
  // only its file contents stop counting.
  assert.match(text, /\[universal-crates\]\ncrates\/contracts\/gen-core\n/);
});

test("dropping a universal crate from the closure still moves the digest", () => {
  // The exclusion must not become a hole: a provider that STOPS linking the shared contract has
  // genuinely changed shape, and that is a structural fact `[crates]`/`[universal-crates]` carry.
  const edited = workspace({
    "crates/b/Cargo.toml": '[package]\nname = "b"\n\n[dependencies]\nshared = { workspace = true }\n',
  });
  assert.notEqual(
    digestsFor({ r1: workspace() }).get("provider_b").digest,
    digestsFor({ r1: edited }).get("provider_b").digest,
  );
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

// ---------------------------------------------------------------------------------------------
// sc-17935 items 1 + 2: local `[patch]` targets and build scripts.
//
// Before this, inference's vendored CUDA kernels were reachable ONLY through the root patch table,
// which the dependency walk never read, and build scripts sat outside the `src/` filter. Every
// `.cu`/`.cuh` file and the `build.rs` that compiles them could change while all five Candle digests
// held still and the calibration reported `current`. Each test below is written as a MUTATION: the
// edit has to move the digest, because a filter that absolves everything is indistinguishable from
// one that works.
// ---------------------------------------------------------------------------------------------

test("a local [patch] target the closure reaches is IN the closure", () => {
  const digests = digestsFor({ r1: workspace() });
  assert.ok(
    digests.get("provider_a").crates.includes("crates/vendor/patched"),
    "provider_a reaches `patched` through the lock, so the patched tree is a compile input",
  );
  assert.ok(
    !digests.get("provider_b").crates.includes("crates/vendor/patched"),
    "provider_b never compiles it — pulling every patch target into every lane re-couples lanes",
  );
});

test("a local [patch] target NO closure reaches is in nobody's closure", () => {
  // `unreachable-patch` is declared in the same table and has a real tree; only lock reachability
  // keeps it out. A filter keyed on "is it a patch entry" rather than "does this closure reach it"
  // would sweep it into both providers.
  for (const provider of Object.keys(PROVIDERS)) {
    assert.ok(!digestsFor({ r1: workspace() }).get(provider).crates.includes("crates/vendor/unreached"));
  }
  const edited = workspace({ "crates/vendor/unreached/src/lib.rs": "pub fn unreached() { /* v2 */ }\n" });
  const before = digestsFor({ r1: workspace() });
  const after = digestsFor({ r1: edited });
  for (const provider of Object.keys(PROVIDERS)) {
    assert.equal(before.get(provider).digest, after.get(provider).digest, provider);
  }
});

test("THE sc-17935 HEADLINE: a kernel edit in a patched tree moves every closure that reaches it", () => {
  const edited = workspace({
    "crates/vendor/patched/src/kernel.cu": "__global__ void k() { /* v2 */ }\n",
  });
  const before = digestsFor({ r1: workspace() });
  const after = digestsFor({ r1: edited });
  assert.notEqual(
    before.get("provider_a").digest,
    after.get("provider_a").digest,
    "a CUDA kernel change must invalidate the calibration that ran on it",
  );
  assert.equal(before.get("provider_b").digest, after.get("provider_b").digest, "provider_b");
});

test("a patched tree's build.rs is hashed", () => {
  const edited = workspace({
    "crates/vendor/patched/build.rs": "fn main() { compile_kernels_with_new_flags(); }\n",
  });
  assert.notEqual(
    digestsFor({ r1: workspace() }).get("provider_a").digest,
    digestsFor({ r1: edited }).get("provider_a").digest,
  );
});

test("an ORDINARY closure crate's build.rs is hashed too", () => {
  // Item 2 costs nothing at the pin only because no first-party closure crate has a build script.
  // That is luck, not design, so the property is pinned on `shared` rather than on the patch target.
  const edited = workspace({
    "crates/shared/build.rs": "fn main() { println!(\"cargo:rustc-cfg=shared_v2\"); }\n",
  });
  const before = digestsFor({ r1: workspace() });
  const after = digestsFor({ r1: edited });
  for (const provider of Object.keys(PROVIDERS)) {
    assert.notEqual(before.get(provider).digest, after.get(provider).digest, provider);
  }
});

test("a build script renamed through `build = \"…\"` is followed", () => {
  const base = workspace({
    "crates/shared/Cargo.toml": '[package]\nname = "shared"\nbuild = "make.rs"\n',
    "crates/shared/make.rs": "fn main() {}\n",
  });
  const edited = {
    ...base,
    "crates/shared/make.rs": "fn main() { different(); }\n",
  };
  assert.notEqual(
    digestsFor({ r1: base }).get("provider_a").digest,
    digestsFor({ r1: edited }).get("provider_a").digest,
  );
});

test("a comment-only edit to a build script is absolved like any other Rust source", () => {
  const edited = workspace({
    "crates/shared/build.rs": "// explain the cfg\nfn main() { println!(\"cargo:rustc-cfg=shared\"); }\n",
  });
  const before = digestsFor({ r1: workspace() });
  const after = digestsFor({ r1: edited });
  for (const provider of Object.keys(PROVIDERS)) {
    assert.equal(before.get(provider).digest, after.get(provider).digest, provider);
  }
});

test("a patched crate's non-compile files stay out of the closure", () => {
  const edited = workspace({ "crates/vendor/patched/README.md": "# vendored, rewritten\n" });
  assert.equal(
    digestsFor({ r1: workspace() }).get("provider_a").digest,
    digestsFor({ r1: edited }).get("provider_a").digest,
  );
});

test("patchTargetPaths reads local replacements only", () => {
  const paths = patchTargetPaths(workspace()["Cargo.toml"]);
  assert.deepEqual(
    [...paths].sort(),
    [["patched", "crates/vendor/patched"], ["unreachable-patch", "crates/vendor/unreached"]],
  );
  // `serde = { git = … }` is a patch entry with no local tree; there is nothing here to hash for it
  // and the lock already carries its identity.
  assert.ok(!paths.has("serde"));
});

test("a renamed patch replacement is matched under BOTH names", () => {
  // `replaced = { path = "…", package = "real-name" }`: the key is what is being replaced and
  // `package` is the replacement's real name. Whichever the lock records, reachability must hit —
  // guessing wrong here fails OPEN, which is the failure mode this whole story is about.
  const paths = patchTargetPaths([
    '[patch."https://example.invalid/upstream"]',
    'replaced = { path = "crates/vendor/patched", package = "real-name" }',
  ].join("\n"));
  assert.equal(paths.get("replaced"), "crates/vendor/patched");
  assert.equal(paths.get("real-name"), "crates/vendor/patched");
});

/**
 * A workspace whose patch table chains: `patched` reaches `deep`, and `deep` is itself a local patch
 * target. `linkThrough` controls WHERE that reach is recorded.
 *
 * - `"lock"` — `patched`'s own lock entry lists `deep`. `lockedClosure` is transitive, so pass 1
 *   already reaches `deep` and both trees are admitted together.
 * - `"manifest"` — the reach exists only as `patched`'s `path =` dependency on `local-helper`, whose
 *   lock entry names `deep`, while `patched`'s lock entry lists nothing. Pass 1 cannot see it; only
 *   walking `patched` in pass 2 puts `local-helper` in `packageNames` and brings `deep` into reach.
 *   That is a lock that has drifted from the manifests — and the case the fixpoint loop exists for.
 */
function chainedPatchWorkspace(linkThrough) {
  const base = workspace();
  return workspace({
    "Cargo.toml": base["Cargo.toml"].replace(
      'unreachable-patch = { path = "crates/vendor/unreached" }',
      'unreachable-patch = { path = "crates/vendor/unreached" }\ndeep = { path = "crates/vendor/deep" }',
    ),
    "Cargo.lock": base["Cargo.lock"].replace(
      '[[package]]\nname = "patched"\nversion = "0.10.2"',
      [
        "[[package]]",
        'name = "patched"',
        'version = "0.10.2"',
        ...(linkThrough === "lock" ? ["dependencies = [", ' "deep",', "]"] : []),
        "",
        "[[package]]",
        'name = "local-helper"',
        'version = "0.0.0"',
        "dependencies = [",
        ' "deep",',
        "]",
        "",
        "[[package]]",
        'name = "deep"',
        'version = "0.0.1"',
      ].join("\n"),
    ),
    "crates/vendor/patched/Cargo.toml": [
      "[package]",
      'name = "patched"',
      "",
      "[dependencies]",
      'local-helper = { path = "../helper" }',
    ].join("\n"),
    "crates/vendor/helper/Cargo.toml": '[package]\nname = "local-helper"\n',
    "crates/vendor/helper/src/lib.rs": "pub fn helper() {}\n",
    "crates/vendor/deep/Cargo.toml": '[package]\nname = "deep"\n',
    "crates/vendor/deep/src/kernel.cu": "__global__ void deep() { /* v1 */ }\n",
  });
}

for (const linkThrough of ["lock", "manifest"]) {
  test(`a patch target reached only through another patch target is admitted (via the ${linkThrough})`, () => {
    const files = chainedPatchWorkspace(linkThrough);
    assert.ok(
      digestsFor({ r1: files }).get("provider_a").crates.includes("crates/vendor/deep"),
      "a chained patch target is still a compile input",
    );
    // Mutation: editing the chained tree must move the digest, or "included" means nothing.
    const edited = {
      ...files,
      "crates/vendor/deep/src/kernel.cu": "__global__ void deep() { /* v2 */ }\n",
    };
    assert.notEqual(
      digestsFor({ r1: files }).get("provider_a").digest,
      digestsFor({ r1: edited }).get("provider_a").digest,
    );
    // And never for a provider that does not reach the chain.
    assert.ok(!digestsFor({ r1: files }).get("provider_b").crates.includes("crates/vendor/deep"));
  });
}

test("the dotted-key spelling of a local patch is read too", () => {
  // `name.path = "…"` is the same declaration in different syntax. Missing it would be a silent
  // false green, not a parse error, which is the worst way for a walk to be incomplete.
  const paths = patchTargetPaths([
    '[patch."https://example.invalid/upstream"]',
    'patched.path = "crates/vendor/patched"',
  ].join("\n"));
  assert.equal(paths.get("patched"), "crates/vendor/patched");
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
