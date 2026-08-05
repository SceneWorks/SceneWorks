import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtempSync, mkdirSync, readFileSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import {
  AUDIT_ARTIFACT_TARGET,
  AUDIT_CLOSURE_BUILD_INPUTS,
  AUDIT_CLOSURE_CRATES,
  AUDIT_CLOSURE_PATHS,
  FEATURE_WITNESS_RESOLUTION,
  assertComparableToolchains,
  assertNoConfigRustflags,
  assertRealCudaCompiler,
  auditRecord,
  buildMeasurementBinary,
  changedClosurePaths,
  closureObjectPairs,
  closureRelativePath,
  coveredClosurePaths,
  digestBytes,
  encodedRustflags,
  featureWitnessText,
  main,
  measurementFeatureDelta,
  normalizePackageSource,
  parseFeatureTree,
  reproducibleLinkFlags,
  resolveFeatureWitness,
  resolveRevision,
  unexplainedMeasurementDrift,
  selectMeasurementExecutable,
} from "./inference-artifact-audit.mjs";

/** A witness block shaped like the one `resolveFeatureWitness` produces, for record assembly. */
const WITNESS = (digest = `sha256:${"a".repeat(64)}`) => ({
  ...FEATURE_WITNESS_RESOLUTION,
  packages: 179,
  measurementDelta: [],
  capturedDigest: digest,
  compatibleDigest: digest,
});

const OBJECT = (seed) => seed.repeat(40).slice(0, 40);

function pairs({ moved = [] } = {}) {
  return AUDIT_CLOSURE_PATHS.map((objectPath, index) => {
    const captured = OBJECT(String(index));
    return {
      path: objectPath,
      capturedObject: captured,
      compatibleObject: moved.includes(objectPath) ? OBJECT("f") : captured,
    };
  });
}

/**
 * A throwaway git repository shaped like the inference closure. The git layer of this audit is the
 * half that has to keep working without a build, so it is exercised against REAL git objects rather
 * than a stubbed runner — a fixture that only ever returns what it was handed proves nothing about
 * `git rev-parse <revision>:<path>`.
 */
function closureRepo() {
  const repo = mkdtempSync(path.join(os.tmpdir(), "sceneworks-audit-repo-"));
  const git = (...args) => execFileSync("git", ["-C", repo, ...args], { encoding: "utf8" }).trim();
  git("init", "-q", "-b", "main");
  git("config", "user.email", "test@example.com");
  git("config", "user.name", "test");
  for (const objectPath of AUDIT_CLOSURE_PATHS) {
    // Build inputs are FILES at the repo root; crate trees are directories. Getting that backwards
    // makes `git rev-parse <rev>:Cargo.lock` name a tree object and the fixture stops resembling
    // the thing under test.
    const file = AUDIT_CLOSURE_BUILD_INPUTS.includes(objectPath)
      ? path.join(repo, objectPath)
      : path.join(repo, objectPath, "src.rs");
    mkdirSync(path.dirname(file), { recursive: true });
    writeFileSync(file, `fn ${objectPath.replace(/[^a-z]/g, "_")}() {}\n`);
  }
  git("add", "-A");
  git("commit", "-qm", "captured");
  const captured = git("rev-parse", "HEAD");
  // The sc-16961 shape verbatim: a doc-comment-only edit inside exactly one closure crate.
  const moved = "crates/media/candle-gen/candle-gen";
  const file = path.join(repo, moved, "src.rs");
  writeFileSync(file, `//! carried-over no-go set\nfn ${moved.replace(/[^a-z]/g, "_")}() {}\n`);
  git("add", "-A");
  git("commit", "-qm", "doc comment only");
  return { repo, git, captured, compatible: git("rev-parse", "HEAD"), moved };
}

test("closure object pairs cover the declared closure and isolate the one path that moved", () => {
  const { repo, captured, compatible, moved } = closureRepo();
  try {
    const observed = closureObjectPairs({ repo, captured, compatible });
    assert.deepEqual(
      observed.map(({ path: objectPath }) => objectPath),
      [...AUDIT_CLOSURE_PATHS],
      "every audited path is reported, in declaration order",
    );
    assert.ok(observed.every(({ capturedObject }) => /^[0-9a-f]{40}$/.test(capturedObject)));
    assert.deepEqual(changedClosurePaths(observed), [moved]);
  } finally {
    rmSync(repo, { recursive: true, force: true });
  }
});

test("a revision that is not a commit is refused rather than recorded as an abbreviation", () => {
  const { repo, captured } = closureRepo();
  try {
    assert.equal(resolveRevision(repo, captured.slice(0, 8)), captured, "an abbreviation resolves to 40 hex");
    assert.throws(() => resolveRevision(repo, "refs/heads/does-not-exist"), /Command failed|did not resolve/);
  } finally {
    rmSync(repo, { recursive: true, force: true });
  }
});

test("the fast path emits no artifact block, and a moved path cannot emit a record without one", () => {
  const clean = auditRecord({
    captured: OBJECT("a"),
    compatible: OBJECT("b"),
    pairs: pairs(),
    featureWitness: WITNESS(),
  });
  assert.equal(clean.schemaVersion, 4);
  assert.deepEqual(clean.changedClosurePaths, []);
  assert.ok(!("auditedArtifact" in clean), "claiming an artifact proof that was never built would be a lie");
  // sc-17606: the witness is the one layer with NO fast path. The change it exists to catch moves no
  // closure object, so a record without it is not a v4 record at all.
  assert.equal(clean.featureWitness.shippedPackage, "runtime-cuda");
  assert.throws(
    () => auditRecord({ captured: OBJECT("a"), compatible: OBJECT("b"), pairs: pairs() }),
    /no resolved-feature witness/,
    "a quiet closure still owes a witness — that is the only path the feature gap ever takes",
  );

  const moved = pairs({ moved: ["crates/media/candle-gen/candle-gen"] });
  assert.throws(
    () => auditRecord({ captured: OBJECT("a"), compatible: OBJECT("b"), pairs: moved, featureWitness: WITNESS() }),
    /closure paths moved .* but no compiled-artifact proof/,
  );
  const proven = auditRecord({
    captured: OBJECT("a"),
    compatible: OBJECT("b"),
    pairs: moved,
    featureWitness: WITNESS(),
    artifact: {
      ...AUDIT_ARTIFACT_TARGET,
      lane: "cuda",
      adjudicates: ["crates/media/candle-gen/candle-gen"],
      capturedDigest: "x",
      compatibleDigest: "x",
    },
  });
  assert.deepEqual(proven.changedClosurePaths, ["crates/media/candle-gen/candle-gen"]);
  assert.equal(proven.auditedArtifact.lane, "cuda");
});

test("a stub CUDA compiler is refused, however convincing its --version output", () => {
  // This is the exact failure mode `scripts/check-candle-build.mjs` creates on purpose: its stub
  // answers `--version` and writes an EMPTY .ptx. candle-kernels `include_str!`s that, so an
  // artifact built against it carries none of the kernels and cannot see a `.cu` change at all.
  assert.throws(
    () => assertRealCudaCompiler({ runProbe: () => ({ ok: true, ptx: "", release: "12.9" }) }),
    /no usable PTX|stub compiler/,
  );
  assert.throws(
    () => assertRealCudaCompiler({ runProbe: () => ({ ok: true, ptx: "// generated by stub\n", release: "12.9" }) }),
    /no usable PTX|stub compiler/,
  );
  assert.throws(
    () => assertRealCudaCompiler({ runProbe: () => ({ ok: false, reason: "nvcc: not found" }) }),
    /needs a real CUDA toolkit/,
  );
  const realPtx = `//\n// Generated by NVIDIA NVVM Compiler\n//\n.version 8.4\n.target sm_90\n.address_size 64\n\n.visible .entry probe()\n{\n\tret;\n}\n`;
  assert.equal(assertRealCudaCompiler({ runProbe: () => ({ ok: true, ptx: realPtx, release: "12.9" }) }), "12.9");
});

test("the hashed file is the one cargo reports for the lib test target, never a guess", () => {
  const libTest = JSON.stringify({
    reason: "compiler-artifact",
    executable: "/t/deps/candle_gen_flux2-abc",
    profile: { test: true },
    target: { kind: ["lib"], name: "candle_gen_flux2" },
  });
  // An integration test and a non-test build of the same crate both sit in the same directory; a
  // glob over target/ would happily hash either.
  const integration = JSON.stringify({
    reason: "compiler-artifact",
    executable: "/t/deps/convert_real_weights-def",
    profile: { test: true },
    target: { kind: ["test"], name: "convert_real_weights" },
  });
  const notATest = JSON.stringify({
    reason: "compiler-artifact",
    executable: "/t/deps/candle_gen_flux2-xyz",
    profile: { test: false },
    target: { kind: ["lib"], name: "candle_gen_flux2" },
  });
  assert.equal(selectMeasurementExecutable([libTest, integration, notATest, "not json"]), "/t/deps/candle_gen_flux2-abc");
  assert.throws(() => selectMeasurementExecutable([integration, notATest]), /got 0/);
  assert.throws(() => selectMeasurementExecutable([libTest, libTest]), /got 2/);
});

test("the digest is taken over the built file's bytes and moves when they do", () => {
  assert.match(digestBytes(Buffer.from("compiled")), /^sha256:[0-9a-f]{64}$/);
  assert.equal(digestBytes(Buffer.from("compiled")), digestBytes(Buffer.from("compiled")));
  assert.notEqual(digestBytes(Buffer.from("compiled")), digestBytes(Buffer.from("compiles")));
});

test("both revisions are built in one worktree, under one remapped, non-incremental toolchain", () => {
  // Two checkouts at different paths bake different `file!()` strings into panic locations and would
  // differ for reasons that have nothing to do with source, so the build discipline IS part of the
  // proof and is asserted here rather than left to the header comment.
  const checkouts = [];
  const invocations = [];
  const report = JSON.stringify({
    reason: "compiler-artifact",
    executable: "/w/target/deps/candle_gen_flux2-1",
    profile: { test: true },
    target: { kind: ["lib"], name: "candle_gen_flux2" },
  });
  const toolchains = [];
  const build = (revision, rustc = "rustc 1.96.0 (ac68faa20 2026-05-25)") =>
    buildMeasurementBinary({
      workdir: "/w",
      revision,
      lane: "cuda",
      cargoTargetDir: "/w-target",
      runGit: (repo, args) => {
        checkouts.push([repo, ...args].join(" "));
        return "";
      },
      runCargo: (call) => {
        invocations.push(call);
        return [report];
      },
      readArtifact: () => Buffer.from(`bytes for ${revision}`),
      readToolchain: (cwd) => {
        toolchains.push([cwd, checkouts.length]);
        return rustc;
      },
      // Injected, not the host's: on the Linux/macOS lanes that run this, `reproducibleLinkFlags()`
      // is `[]`, so asserting against the host value passes with the flags unwired entirely.
      linkFlags: reproducibleLinkFlags("win32"),
    });

  const first = build("5ffd7612");
  const second = build("d2216f6b");
  assert.deepEqual(checkouts, ["/w checkout --detach 5ffd7612", "/w checkout --detach d2216f6b"]);
  assert.ok(invocations.every(({ cwd }) => cwd === "/w"), "one directory, so `file!()` cannot drift");
  for (const { args, env } of invocations) {
    assert.ok(args.includes("--lib") && args.includes("--no-run"), "the lib test binary, not a run");
    assert.deepEqual(args.slice(args.indexOf("--features"), args.indexOf("--features") + 2), ["--features", "cuda"]);
    assert.ok(args.includes("--locked"), "an unlocked resolve could move a dependency between builds");
    assert.equal(env.CARGO_INCREMENTAL, "0", "incremental artifacts are not reproducible");
    assert.equal(env.RUSTFLAGS, undefined, "the encoded form must not be shadowed by the split one");
    const flags = env.CARGO_ENCODED_RUSTFLAGS.split("\u001f");
    assert.ok(flags.includes("--remap-path-prefix=/w=/inference"));
    assert.ok(flags.some((flag) => /^--remap-path-prefix=.*=\/cargo$/.test(flag)));
    assert.deepEqual(
      flags.filter((flag) => flag.startsWith("-Clink-arg=")),
      reproducibleLinkFlags("win32"),
      "the reproducibility flags ride every build, not just the first",
    );
  }
  assert.notEqual(first.digest, second.digest, "different bytes, different digest");
  assert.equal(first.executable, "/w/target/deps/candle_gen_flux2-1");
  // The toolchain is read AFTER each checkout, not once up front: a checkout swaps
  // `rust-toolchain.toml`, so rustup can hand the second build a different compiler and a
  // single up-front read would record one that did not build both artifacts.
  assert.deepEqual(toolchains, [["/w", 1], ["/w", 2]]);
  assert.equal(first.rustc, second.rustc);
});

test("a rustc bump between the two revisions is a hard stop, not a comparison", () => {
  // sc-17524 put `rust-toolchain.toml` in the closure so a rustc bump can no longer take the free
  // path — it forces the build, and this is what the build then does with it. Asserted directly
  // because inside `main` it sits behind two real CUDA builds, so nothing ever reached it.
  const args = { captured: "5ffd7612", compatible: "a4f409ae" };
  assert.equal(
    assertComparableToolchains({ ...args, capturedRustc: "rustc 1.96.0", compatibleRustc: "rustc 1.96.0" }),
    "rustc 1.96.0",
  );
  assert.throws(
    () => assertComparableToolchains({ ...args, capturedRustc: "rustc 1.96.0", compatibleRustc: "rustc 1.97.0" }),
    /different toolchains.*not comparable/s,
    "two compilers cannot produce comparable digests, so this may never fall through to a compare",
  );
  // The message has to name both, or an operator cannot tell which side moved.
  assert.throws(
    () => assertComparableToolchains({ ...args, capturedRustc: "rustc 1.96.0", compatibleRustc: "rustc 1.97.0" }),
    /5ffd7612: rustc 1\.96\.0[\s\S]*a4f409ae: rustc 1\.97\.0/,
  );
});

test("two revisions built under different toolchains are refused, not quietly compared", async () => {
  // Digests from two different compilers are not comparable at all, so this is a hard stop.
  const report = JSON.stringify({
    reason: "compiler-artifact",
    executable: "/w/target/deps/candle_gen_flux2-1",
    profile: { test: true },
    target: { kind: ["lib"], name: "candle_gen_flux2" },
  });
  const build = (rustc) =>
    buildMeasurementBinary({
      workdir: "/w",
      revision: "abc",
      lane: "cuda",
      cargoTargetDir: "/w-target",
      runGit: () => "",
      runCargo: () => [report],
      readArtifact: () => Buffer.from("same bytes"),
      readToolchain: () => rustc,
      linkFlags: [],
    });
  assert.notEqual(build("rustc 1.96.0").rustc, build("rustc 1.97.0").rustc);
});

test("coverage is read off what cargo actually compiled, and a gap is refused not glossed", () => {
  const artifact = (manifest) =>
    JSON.stringify({ reason: "compiler-artifact", manifest_path: `/w/${manifest}`, executable: null });
  const compiled = [
    artifact("crates/media/candle-gen/candle-gen-flux2/Cargo.toml"),
    artifact("crates/media/candle-gen/candle-gen/Cargo.toml"),
    artifact("crates/media/candle-gen/candle-gen-pid/Cargo.toml"),
    artifact("crates/contracts/gen-core/Cargo.toml"),
    artifact("crates/media/candle-gen/vendor/candle-kernels/Cargo.toml"),
    // Out-of-workspace and out-of-closure crates must not widen coverage.
    JSON.stringify({ reason: "compiler-artifact", manifest_path: "/home/x/.cargo/registry/serde/Cargo.toml" }),
    artifact("crates/media/candle-gen/candle-gen-catalog/Cargo.toml"),
    JSON.stringify({ reason: "build-script-executed", manifest_path: "/w/crates/bundles/runtime-cuda/Cargo.toml" }),
    "not json",
  ];
  const covered = coveredClosurePaths(compiled, "/w");
  assert.deepEqual(covered, [
    // sc-17524: the build inputs are covered because they are inputs to THIS build — cargo
    // never names them, so compile coverage alone would leave them permanently unadjudicable.
    "Cargo.toml",
    "Cargo.lock",
    "rust-toolchain.toml",
    ".cargo/config.toml",
    "crates/contracts/gen-core",
    "crates/media/candle-gen/candle-gen",
    "crates/media/candle-gen/candle-gen-pid",
    "crates/media/candle-gen/candle-gen-flux2",
    "crates/media/candle-gen/vendor/candle-kernels",
  ]);
  assert.ok(
    !covered.includes("crates/bundles/runtime-cuda"),
    "runtime-cuda depends on the provider, not the reverse — the measurement binary never links it",
  );

  // The build-input inference is anchored to the audited package, not to "some build happened".
  // Without that anchor an empty or unrecognized cargo report still hands back three adjudicable
  // paths, which is a digest signing for inputs to a build it cannot be shown to have come from.
  const withoutTheTarget = compiled.filter((line) => !line.includes(`${AUDIT_ARTIFACT_TARGET.package}/Cargo.toml`));
  assert.deepEqual(coveredClosurePaths(withoutTheTarget, "/w"), [
    "crates/contracts/gen-core",
    "crates/media/candle-gen/candle-gen",
    "crates/media/candle-gen/candle-gen-pid",
    "crates/media/candle-gen/vendor/candle-kernels",
  ]);
  assert.deepEqual(coveredClosurePaths([], "/w"), [], "a report that compiled nothing covers nothing");
  for (const input of AUDIT_CLOSURE_BUILD_INPUTS) {
    assert.ok(!coveredClosurePaths(withoutTheTarget, "/w").includes(input), `${input} needs the audited target`);
  }

  // Regression: `os.tmpdir()` is `/var/folders/...` on macOS while cargo reports manifests under the
  // canonical `/private/var/folders/...`. Comparing those two spellings resolved EVERY closure path
  // to "outside the worktree" and produced an empty covered set. Caught by running the tool for real.
  const canonical = mkdtempSync(path.join(os.tmpdir(), "sceneworks-audit-real-"));
  const symlinked = `${canonical}-link`;
  symlinkSync(canonical, symlinked);
  // Real files, because the resolution only has anything to resolve when the path exists — which is
  // always true of a manifest cargo just compiled.
  const manifest = path.join(canonical, "crates/media/candle-gen/candle-gen/Cargo.toml");
  mkdirSync(path.dirname(manifest), { recursive: true });
  writeFileSync(manifest, "[package]\n");
  try {
    const viaSymlink = coveredClosurePaths(
      [JSON.stringify({ reason: "compiler-artifact", manifest_path: manifest })],
      symlinked,
    );
    assert.deepEqual(viaSymlink, ["crates/media/candle-gen/candle-gen"]);
  } finally {
    rmSync(symlinked, { force: true });
    rmSync(canonical, { recursive: true, force: true });
  }

  // A move in an unlinked path cannot be signed off by a digest that could not have seen it.
  const moved = pairs({ moved: ["crates/bundles/runtime-cuda"] });
  assert.throws(
    () =>
      auditRecord({
        captured: OBJECT("a"),
        compatible: OBJECT("b"),
        pairs: moved,
        featureWitness: WITNESS(),
        artifact: { lane: "cuda", adjudicates: covered, capturedDigest: "x", compatibleDigest: "x" },
      }),
    /does not link crates\/bundles\/runtime-cuda/,
  );
});

test("a cargo config that declares rustflags is refused, because this script overrides them", () => {
  // sc-17524. `.cargo/config.toml` is a closure path now, which makes this sharper than it looks:
  // cargo REPLACES `build.rustflags` with `CARGO_ENCODED_RUSTFLAGS` rather than merging, and this
  // script must set the latter for the path remaps. So declared flags are dropped from BOTH builds,
  // the digests agree with each other and with nothing that ships — and the audit would then report
  // the config edit as adjudicated. Refusing is the only honest answer until the flags are merged.
  const ok = (config) => assert.doesNotThrow(() => assertNoConfigRustflags(config, "5ffd7612"));
  ok('[env]\nRUST_TEST_THREADS = { value = "1", force = true }\n');
  ok("");
  ok('# rustflags = "-Ctarget-cpu=native" is commented out\n');
  ok('[build]\nrustdocflags = ["-Dwarnings"]\n');
  for (const declared of [
    '[build]\nrustflags = ["-Ctarget-cpu=native"]\n',
    '[build]\nrustflags=["-Ctarget-cpu=native"]\n',
    '[target.x86_64-pc-windows-msvc]\nrustflags = ["-Clink-arg=/STACK:8000000"]\n',
    '[target."cfg(all())"]\n  rustflags = ["-Dwarnings"]\n',
    // TOML spellings a line-anchored regex misses, all of which cargo honours.
    'build.rustflags = ["-Ctarget-cpu=native"]\n',
    'build = { rustflags = ["-C", "target-cpu=native"] }\n',
    'target."cfg(all())".rustflags = ["-Dwarnings"]\n',
    '[build]\n"rustflags" = ["-Dwarnings"]\n',
  ]) {
    assert.throws(
      () => assertNoConfigRustflags(declared, "5ffd7612"),
      /declares rustflags.*REPLACES/s,
      `must refuse: ${declared.split("\n")[0]}`,
    );
  }
});

test("a manifest path is compared in POSIX form on every host, not in the host's own separators", () => {
  // sc-17587. `path.relative` returns NATIVE separators, so on Windows every manifest resolved to
  // "outside the worktree", the covered set came back empty, and the audit refused its own build —
  // on the only box that can produce a `cuda` record. It reached main green because on the
  // Linux/macOS `check` lanes `path.sep` is already "/", so a test using the host implementation
  // passes with or without the normalisation. Driving `path.win32` explicitly is what makes this
  // guard bite everywhere instead of only on the machine that was already broken.
  assert.equal(
    closureRelativePath("D:\\w", "D:\\w\\crates\\contracts\\gen-core\\Cargo.toml", path.win32),
    "crates/contracts/gen-core/Cargo.toml",
  );
  assert.equal(
    closureRelativePath("/w", "/w/crates/contracts/gen-core/Cargo.toml", path.posix),
    "crates/contracts/gen-core/Cargo.toml",
  );
  // Out-of-tree stays out of tree in both dialects — the normalisation must not widen coverage.
  assert.equal(closureRelativePath("D:\\w", "D:\\elsewhere\\Cargo.toml", path.win32), null);
  assert.equal(closureRelativePath("/w", "/elsewhere/Cargo.toml", path.posix), null);
  assert.equal(closureRelativePath("D:\\w", "C:\\w\\crates\\x\\Cargo.toml", path.win32), null);
});

test("the Windows link timestamp is stamped out, because a digest that moves on its own proves nothing", () => {
  // Found by running the audit for real on the RTX box. `link.exe` writes the LINK TIME into the
  // PE header, so the same inference revision built twice hashed differently (5ffd7612 gave
  // sha256:2164e988… then sha256:57f15abb…) and the cuda lane could only ever report ARTIFACTS
  // DIFFER — an unfalsifiable "re-capture on an RTX PRO 6000" for code that had not changed.
  // Asserted per-platform rather than against `process.platform`, so this stays meaningful on the
  // Linux/macOS CI that runs it and cannot pass by agreeing with itself.
  // /Brepro ALONE does not fix it — measured, not assumed: it hashes the image, and the image still
  // carries a varying PDB signature. /DEBUG:NONE is the flag that decides, by stopping link.exe
  // emitting a PDB at all. Asserting both means dropping either one goes red.
  assert.deepEqual(reproducibleLinkFlags("win32"), ["-Clink-arg=/Brepro", "-Clink-arg=/DEBUG:NONE"]);
  assert.deepEqual(reproducibleLinkFlags("darwin"), [], "Mach-O carries no link timestamp");
  assert.deepEqual(reproducibleLinkFlags("linux"), [], "nor does ELF");
});

test("a remap path containing a space survives, because the CUDA box is Windows", () => {
  // cargo splits RUSTFLAGS on WHITESPACE. The only machine that can produce a `cuda` record is the
  // Windows RTX box, whose default temp directory sits under a profile path that routinely contains
  // a space — so a space-joined remap is silently truncated into flags that match nothing, and the
  // digests would then embed real paths. The 0x1f-encoded form takes arbitrary values.
  const spaced = "--remap-path-prefix=C:\\Users\\Michael Trefry\\AppData\\Local\\Temp\\audit=/inference";
  const encoded = encodedRustflags([spaced], "-C debuginfo=0");
  assert.deepEqual(encoded.split("\u001f"), ["-C", "debuginfo=0", spaced]);
  assert.equal(encodedRustflags([spaced], undefined).split("\u001f").length, 1, "no env, one flag");
  assert.equal(encodedRustflags([], "").length, 0, "nothing in, nothing out");
});

test("the audited artifact names the target that actually produced the measurements", () => {
  // sc-15833's approved capture command runs exactly this test in this package; auditing anything
  // else would be auditing code the calibration never ran.
  assert.equal(AUDIT_ARTIFACT_TARGET.package, "candle-gen-flux2");
  assert.equal(AUDIT_ARTIFACT_TARGET.test, "tests::flux2_dev_probed_generate_for_offload_ab");
  assert.equal(AUDIT_ARTIFACT_TARGET.profile, "release");
  assert.ok(
    AUDIT_CLOSURE_CRATES.includes(AUDIT_ARTIFACT_TARGET.closurePath),
    "the build-input gate is anchored to this path, so it has to be one of the audited crate trees",
  );
});

// -------------------------------------------------------------------------------------------
// sc-17606: the resolved-feature witness.
//
// The audited binary is built `-p candle-gen-flux2`, which unifies features over a strictly
// smaller graph than the shipped `-p runtime-cuda` bundle does. A crate outside FLUX.2's closure
// can therefore enable a feature on a dependency FLUX.2 SHARES and change what it executes in the
// shipped runtime — moving no closure object, no lockfile entry (features are not recorded there)
// and no artifact digest. The witness is the only layer that can see it, and the only one that
// runs on the free path, because the free path is where that change always arrives.
// -------------------------------------------------------------------------------------------

const TREE = (...lines) => lines;

test("a package's location is normalized out, or the witness describes the worktree not the code", () => {
  // The Windows case is the reason this is not a one-line `scheme:` test: a path dependency renders
  // as `D:\repos\...`, and `^[a-z+]*:` reads `D:` as a URL scheme — so the absolute path would be
  // hashed verbatim and two runs of the same revision in two directories would disagree. Driven
  // through `path.win32` explicitly, for the sc-17587 reason: on Linux/macOS this cannot arise at
  // all, so a test using the host's implementation passes with the normalisation deleted.
  assert.equal(
    normalizePackageSource("candle-gen v0.0.0 (D:\\w\\crates\\media\\candle-gen\\candle-gen)", "D:\\w", path.win32),
    "candle-gen v0.0.0 (crates/media/candle-gen/candle-gen)",
  );
  assert.equal(
    normalizePackageSource("candle-gen v0.0.0 (/w/crates/media/candle-gen/candle-gen)", "/w", path.posix),
    "candle-gen v0.0.0 (crates/media/candle-gen/candle-gen)",
  );
  // Registry crates carry no source at all, and remote ones carry a rev that IS the identity.
  assert.equal(normalizePackageSource("byteorder v1.5.0", "/w", path.posix), "byteorder v1.5.0");
  const remote = "candle-core v0.10.2 (https://github.com/huggingface/candle?rev=1e6aa85e#1e6aa85e)";
  assert.equal(normalizePackageSource(remote, "/w", path.posix), remote);
  const registry = "serde v1.0.0 (registry+https://github.com/rust-lang/crates.io-index)";
  assert.equal(normalizePackageSource(registry, "/w", path.posix), registry);
  // cargo appends a KIND marker after the source. Read as a source it is neither a URL nor a
  // relativizable path, so every registry proc-macro in the bundle — `paste`, `seq-macro`,
  // `dyn-stack-macros` — came back "outside the worktree". Caught by running this for real.
  assert.equal(normalizePackageSource("paste v1.0.15 (proc-macro)", "/w", path.posix), "paste v1.0.15 (proc-macro)");
  assert.equal(
    normalizePackageSource("dyn-stack-macros v0.1.3 (/w/crates/x) (proc-macro)", "/w", path.posix),
    "dyn-stack-macros v0.1.3 (crates/x) (proc-macro)",
    "…and a path proc-macro still has its location normalized, not just its marker preserved",
  );
  // A `[patch]` at a checkout on the operator's disk cannot be normalized, and hashing it would
  // make the witness a property of that machine. Refused rather than silently kept or dropped.
  assert.throws(
    () => normalizePackageSource("candle-core v0.10.2 (/home/x/candle)", "/w", path.posix),
    /outside the worktree/,
  );
  assert.throws(
    () => normalizePackageSource("candle-macros v0.1.0 (/home/x/candle) (proc-macro)", "/w", path.posix),
    /outside the worktree/,
    "peeling the marker must not become a way to smuggle an unnormalizable path past the guard",
  );
});

test("an empty cargo tree parse is refused, because a witness over nothing never changes", () => {
  // This is not hypothetical: the first run of this implementation omitted `--format`, every line
  // failed the `|` test, and the result was a perfectly stable digest of the empty string that was
  // identical at both revisions. A layer that certifies by producing nothing is worse than absent.
  assert.throws(() => parseFeatureTree(TREE(""), "/w", path.posix), /no packages/);
  // And a line the parser does not understand is refused rather than skipped: skipping one would
  // quietly shrink the hashed set by a package instead of emptying it, which no guard would see.
  assert.throws(
    () => parseFeatureTree(TREE("candle-core v0.10.2|cuda", "runtime-cuda v0.0.0"), "/w", path.posix),
    /does not recognize: runtime-cuda/,
  );
  assert.doesNotThrow(
    () => parseFeatureTree(TREE("candle-core v0.10.2|cuda", "", "   "), "/w", path.posix),
    "blank lines are cargo's trailing newline, not unparsed output",
  );
});

test("cargo's dedupe marker and the host/target feature split are both parsed, not flattened", () => {
  const tree = parseFeatureTree(
    TREE(
      "serde_json v1.0.150|alloc,default,indexmap,preserve_order,std",
      // The same package under the v2 resolver's HOST unification (a build script's dependency) —
      // genuinely a second feature set, and collapsing the two would hide one of them.
      "serde_json v1.0.150|std,default",
      // ` (*)` is cargo's "subtree already shown" marker and is appended AFTER the format string.
      // Left on, it becomes part of the feature list and the same package hashes two ways.
      "serde_json v1.0.150|alloc,default,indexmap,preserve_order,std (*)",
      "byteorder v1.5.0|",
    ),
    "/w",
    path.posix,
  );
  assert.deepEqual(
    [...tree.get("serde_json v1.0.150")].sort(),
    ["alloc,default,indexmap,preserve_order,std", "default,std"],
    "two resolutions, two entries — and the feature list is sorted so cargo's order cannot move it",
  );
  assert.deepEqual([...tree.get("byteorder v1.5.0")], [""], "no features is a value, not a missing package");
});

test("the witness hashes the SHIPPED features of the packages FLUX.2 links, and nothing else", () => {
  const shipped = parseFeatureTree(
    TREE(
      "candle-core v0.10.2|cuda,cudarc,default",
      "serde_json v1.0.150|alloc,default,indexmap,preserve_order,std",
      // Bundle-only: another provider's crate. It must not reach the witness, or this becomes the
      // ~50-crate false-positive trigger sc-17524 measured and rejected.
      "candle-gen-sdxl v0.0.0 (/w/crates/media/candle-gen/candle-gen-sdxl)|cuda",
    ),
    "/w",
    path.posix,
  );
  const members = parseFeatureTree(
    TREE("candle-core v0.10.2|cuda,cudarc,default", "serde_json v1.0.150|default,std"),
    "/w",
    path.posix,
  );
  const text = featureWitnessText({ shipped, members });
  assert.ok(!text.includes("candle-gen-sdxl"), "a provider FLUX.2 does not link cannot move this witness");
  assert.ok(
    text.includes("serde_json v1.0.150|alloc,default,indexmap,preserve_order,std"),
    "the SHIPPED feature set is what is hashed — taking the member tree's own would measure nothing",
  );
  assert.ok(!text.includes("serde_json v1.0.150|default,std"), "…and only the shipped one");
  for (const line of ["# shipped-package: runtime-cuda", "# target: x86_64-pc-windows-msvc", "# edges: normal,build"]) {
    assert.ok(text.includes(line), `${line} is hashed, so two witnesses of different questions cannot compare equal`);
  }

  // The realized mechanism this story exists for: something outside the closure turns on
  // `preserve_order` for a dependency FLUX.2 shares. No member changed; the witness still moves.
  const widened = parseFeatureTree(
    TREE(
      "candle-core v0.10.2|cuda,cudarc,default",
      "serde_json v1.0.150|alloc,default,indexmap,preserve_order,std,unbounded_depth",
      "candle-gen-sdxl v0.0.0 (/w/crates/media/candle-gen/candle-gen-sdxl)|cuda",
    ),
    "/w",
    path.posix,
  );
  assert.notEqual(featureWitnessText({ shipped: widened, members }), text);
  // …while a change confined to a bundle crate FLUX.2 never links does not.
  const unrelated = parseFeatureTree(
    TREE(
      "candle-core v0.10.2|cuda,cudarc,default",
      "serde_json v1.0.150|alloc,default,indexmap,preserve_order,std",
      "candle-gen-sdxl v0.0.0 (/w/crates/media/candle-gen/candle-gen-sdxl)|cuda,flash-attn",
    ),
    "/w",
    path.posix,
  );
  assert.equal(featureWitnessText({ shipped: unrelated, members }), text);

  // A member the bundle does not contain would simply be absent from the hashed lines, which reads
  // exactly like "nothing changed". FLUX.2 not being reachable from what ships is a hard stop.
  const orphaned = parseFeatureTree(TREE("candle-core v0.10.2|cuda"), "/w", path.posix);
  assert.throws(() => featureWitnessText({ shipped: orphaned, members }), /absent from runtime-cuda/);
});

test("the measured-vs-shipped feature delta is reported, and unexplained drift is surfaced", () => {
  const shipped = parseFeatureTree(TREE("candle-gen v0.0.0|cuda,default", "serde v1.0.0|default"), "/w", path.posix);
  const measured = parseFeatureTree(
    // The real shape at both audited revisions: the test binary turns on `candle-gen/testkit` and
    // pulls a testkit crate that never ships. That is the "it is a test binary" caveat, quantified.
    TREE("candle-gen v0.0.0|cuda,default,testkit", "serde v1.0.0|default", "gen-core-testkit v0.1.0|"),
    "/w",
    path.posix,
  );
  assert.deepEqual(measurementFeatureDelta(measured, shipped), [
    "candle-gen v0.0.0: measured [cuda,default,testkit] shipped [cuda,default]",
    "gen-core-testkit v0.1.0: measured only",
  ]);
  assert.deepEqual(measurementFeatureDelta(shipped, shipped), [], "identical resolutions have no delta");

  const args = { captured: "5ffd7612", compatible: "a4f409ae", shippedWitnessMoved: false };
  assert.equal(unexplainedMeasurementDrift({ ...args, capturedDelta: ["a"], compatibleDelta: ["a"] }), null);
  // A delta that grows with nothing on the shipped side behind it means the hashed binary
  // represents the shipped code differently at the two ends.
  assert.match(
    unexplainedMeasurementDrift({ ...args, capturedDelta: ["a"], compatibleDelta: ["a", "b"] }),
    /drifted relative to the shipped bundle[\s\S]*5ffd7612[\s\S]*a4f409ae/,
  );
  assert.match(
    unexplainedMeasurementDrift({ ...args, capturedDelta: [], compatibleDelta: ["b"] }),
    /\(none\)/,
    "an empty side must still print legibly, or the operator cannot tell which end moved",
  );
  // The subtlety, found by running the tool end to end against a real outside-closure feature
  // change: the delta is `measured − shipped`, so widening the SHIPPED side widens the delta too.
  // Reported here, the change this whole layer exists to catch would surface under the wrong name.
  assert.equal(
    unexplainedMeasurementDrift({
      ...args,
      shippedWitnessMoved: true,
      capturedDelta: ["a"],
      compatibleDelta: ["a", "serde_json v1.0.150: measured [x] shipped [x,unbounded_depth]"],
    }),
    null,
    "a delta that moved BECAUSE the shipped resolution moved is arithmetic, not a second finding",
  );
});

test("the witness resolves three trees per revision, all locked, at the pinned shipping target", () => {
  const calls = [];
  const checkouts = [];
  const witness = resolveFeatureWitness({
    workdir: "/w",
    revision: "5ffd7612",
    runGit: (repo, args) => {
      checkouts.push([repo, ...args].join(" "));
      return "";
    },
    runCargoTree: ({ cwd, args }) => {
      calls.push({ cwd, args });
      const selected = args[args.indexOf("-p") + 1];
      const dev = args.includes("normal,build,dev");
      return selected === "runtime-cuda"
        ? ["candle-gen v0.0.0|cuda,default", "candle-gen-sdxl v0.0.0|cuda"]
        : dev
          ? ["candle-gen v0.0.0|cuda,default,testkit"]
          : ["candle-gen v0.0.0|cuda,default"];
    },
    platformPath: path.posix,
  });
  assert.deepEqual(checkouts, ["/w checkout --detach 5ffd7612"]);
  assert.equal(calls.length, 3, "shipped resolution, FLUX.2's members, and the measurement resolution");
  for (const { cwd, args } of calls) {
    assert.equal(cwd, "/w");
    assert.ok(args.includes("--locked"), "an unlocked resolve could move a dependency between revisions");
    assert.deepEqual(args.slice(args.indexOf("--format"), args.indexOf("--format") + 2), ["--format", "{p}|{f}"]);
    assert.deepEqual(
      args.slice(args.indexOf("--target"), args.indexOf("--target") + 2),
      ["--target", "x86_64-pc-windows-msvc"],
      "the calibration is a Windows/CUDA one, so the witness is pinned to that triple on every host",
    );
  }
  assert.match(witness.digest, /^sha256:[0-9a-f]{64}$/);
  assert.equal(witness.packages, 1, "the member count, not the bundle's — sdxl is not FLUX.2's");
  assert.deepEqual(witness.measurementDelta, [
    "candle-gen v0.0.0: measured [cuda,default,testkit] shipped [cuda,default]",
  ]);
});

test("the resolution the witness describes is the bundle SceneWorks actually consumes", () => {
  // `sceneworks-worker` and `sceneworks-memory-adapter` both take `runtime-cuda` with DEFAULT
  // features, so no feature override belongs here; scoping to `candle-gen-flux2` is what keeps a
  // commit to one of ~50 unrelated bundle crates from demanding a 47.6 GB re-capture.
  assert.deepEqual({ ...FEATURE_WITNESS_RESOLUTION }, {
    shippedPackage: "runtime-cuda",
    scopeRoot: "candle-gen-flux2",
    scopeFeatures: "cuda",
    target: "x86_64-pc-windows-msvc",
    edges: "normal,build",
  });
  assert.equal(
    FEATURE_WITNESS_RESOLUTION.scopeRoot,
    AUDIT_ARTIFACT_TARGET.package,
    "the witness and the hashed binary must be scoped to the same provider or they answer about different code",
  );
  assert.ok(
    !FEATURE_WITNESS_RESOLUTION.edges.includes("dev"),
    "dev-dependencies do not ship; they belong in measurementDelta, not in what the witness certifies",
  );
});

/**
 * A fake git + cargo tree pair that answers `main` for a two-revision run with NO build.
 *
 * `runCargoTree` answers from whichever revision `runGit` last checked out, which is the only way
 * to catch a wiring bug that resolves both witnesses at the same revision — the fake has to model
 * the checkout, or every assertion below passes with the two calls collapsed into one.
 */
function fakeAudit({ trees, objects = {} }) {
  const calls = [];
  let head = null;
  const OBJECT_OF = (revision, objectPath) =>
    (objects[revision]?.[objectPath] ?? objectPath).padEnd(40, "0").slice(0, 40).replace(/[^0-9a-f]/g, "a");
  return {
    calls,
    head: () => head,
    runGit(repo, args) {
      calls.push(args.join(" "));
      if (args[0] === "rev-parse" && args[1].endsWith("^{commit}")) {
        return args[1].replace("^{commit}", "").padEnd(40, "0").replace(/[^0-9a-f]/g, "a");
      }
      if (args[0] === "rev-parse") {
        const [revision, objectPath] = args[1].split(":");
        return OBJECT_OF(revision, objectPath);
      }
      if (args[0] === "checkout") head = args[2];
      if (args[0] === "worktree" && args[1] === "add") head = args[4];
      return "";
    },
    runCargoTree({ args }) {
      const selected = args[args.indexOf("-p") + 1];
      calls.push(`tree@${head}:${selected}${args.includes("normal,build,dev") ? ":dev" : ""}`);
      return trees[head][selected];
    },
  };
}

const REV_A = "a".repeat(40);
const REV_B = "b".repeat(40);

test("main resolves one witness per revision and puts each digest on its own side", async () => {
  // sc-17606: every load-bearing part of the witness layer's WIRING lives in `main` — which
  // revision each witness resolves at, whose digest lands where, and the exit code. None of it is
  // reachable from a unit test of the pure functions: swapping `compatibleWitness.digest` for
  // `capturedWitness.digest` makes the layer permanently green and leaves them all passing. This is
  // the same argument that pulled `assertComparableToolchains` out of `main` in sc-17497.
  const out = path.join(mkdtempSync(path.join(os.tmpdir(), "sceneworks-audit-main-")), "record.json");
  const workdir = mkdtempSync(path.join(os.tmpdir(), "sceneworks-audit-main-wd-"));
  const members = ["candle-gen-flux2 v0.0.0|cuda"];
  const fake = fakeAudit({
    trees: {
      // The story's mechanism: something outside FLUX.2's closure widens a dependency it links.
      [REV_A]: {
        "runtime-cuda": ["candle-gen-flux2 v0.0.0|cuda", "serde_json v1.0.150|default,std"],
        "candle-gen-flux2": [...members, "serde_json v1.0.150|default,std"],
      },
      [REV_B]: {
        "runtime-cuda": ["candle-gen-flux2 v0.0.0|cuda", "serde_json v1.0.150|default,preserve_order,std"],
        "candle-gen-flux2": [...members, "serde_json v1.0.150|default,std"],
      },
    },
  });
  try {
    const code = await main(
      ["--repo", "/r", "--captured", REV_A, "--compatible", REV_B, "--workdir", workdir, "--out", out],
      { runGit: fake.runGit, runCargoTree: fake.runCargoTree },
    );
    const record = JSON.parse(readFileSync(out, "utf8"));
    assert.deepEqual(record.changedClosurePaths, [], "no closure object moved — this IS the free path");
    assert.ok(!("auditedArtifact" in record), "and nothing was built");
    assert.notEqual(
      record.featureWitness.capturedDigest,
      record.featureWitness.compatibleDigest,
      "the shipped resolution changed, and the record must say which side is which",
    );
    assert.equal(code, 1, "a free-path run that certified this would be the exact bug sc-17606 exists for");
    // Each side must come from ITS revision, asserted by re-deriving one of them in isolation.
    const only = (revision) =>
      resolveFeatureWitness({
        workdir,
        revision,
        runGit: fake.runGit,
        runCargoTree: fake.runCargoTree,
        platformPath: path.posix,
      }).digest;
    assert.equal(record.featureWitness.capturedDigest, only(REV_A));
    assert.equal(record.featureWitness.compatibleDigest, only(REV_B));
    assert.ok(
      fake.calls.includes(`tree@${REV_A}:runtime-cuda`) && fake.calls.includes(`tree@${REV_B}:runtime-cuda`),
      "both revisions are actually checked out and resolved, not one of them twice",
    );
  } finally {
    rmSync(path.dirname(out), { recursive: true, force: true });
    rmSync(workdir, { recursive: true, force: true });
  }
});

test("main exits 0 and records a witness when the shipped resolution is unchanged", async () => {
  const out = path.join(mkdtempSync(path.join(os.tmpdir(), "sceneworks-audit-main-")), "record.json");
  const workdir = mkdtempSync(path.join(os.tmpdir(), "sceneworks-audit-main-wd-"));
  const tree = {
    "runtime-cuda": ["candle-gen-flux2 v0.0.0|cuda", "serde_json v1.0.150|default,std"],
    "candle-gen-flux2": ["candle-gen-flux2 v0.0.0|cuda", "serde_json v1.0.150|default,std"],
  };
  const fake = fakeAudit({ trees: { [REV_A]: tree, [REV_B]: tree } });
  try {
    const code = await main(
      ["--repo", "/r", "--captured", REV_A, "--compatible", REV_B, "--workdir", workdir, "--out", out],
      { runGit: fake.runGit, runCargoTree: fake.runCargoTree },
    );
    const record = JSON.parse(readFileSync(out, "utf8"));
    assert.equal(code, 0);
    assert.equal(record.schemaVersion, 4);
    assert.equal(record.featureWitness.capturedDigest, record.featureWitness.compatibleDigest);
    assert.equal(record.featureWitness.packages, 2, "the member count, from the member tree");
    assert.equal(record.featureWitness.scopeFeatures, "cuda");
  } finally {
    rmSync(path.dirname(out), { recursive: true, force: true });
    rmSync(workdir, { recursive: true, force: true });
  }
});

test("main puts each BUILD's digest on its own side, and reports both layers independently", async () => {
  // The artifact layer's wiring has the same untestable-from-below shape as the witness layer's.
  // Driven on `--lane metal` so the nvcc probe is not reached; the record is inert by construction.
  const out = path.join(mkdtempSync(path.join(os.tmpdir(), "sceneworks-audit-main-")), "record.json");
  const workdir = mkdtempSync(path.join(os.tmpdir(), "sceneworks-audit-main-wd-"));
  const moved = "crates/media/candle-gen/candle-gen";
  const tree = {
    "runtime-cuda": ["candle-gen-flux2 v0.0.0|metal"],
    "candle-gen-flux2": ["candle-gen-flux2 v0.0.0|metal"],
  };
  const fake = fakeAudit({
    trees: { [REV_A]: tree, [REV_B]: tree },
    // Only `candle-gen` differs between the two revisions, so a build is owed for exactly it.
    objects: { [REV_B]: { [moved]: "f".repeat(40) } },
  });
  const report = [
    JSON.stringify({
      reason: "compiler-artifact",
      executable: "/w/target/deps/candle_gen_flux2-1",
      manifest_path: `${workdir}/crates/media/candle-gen/candle-gen-flux2/Cargo.toml`,
      profile: { test: true },
      target: { kind: ["lib"], name: "candle_gen_flux2" },
    }),
    JSON.stringify({ reason: "compiler-artifact", manifest_path: `${workdir}/${moved}/Cargo.toml` }),
  ];
  let built = 0;
  try {
    const code = await main(
      [
        ...["--repo", "/r", "--captured", REV_A, "--compatible", REV_B],
        ...["--lane", "metal", "--workdir", workdir, "--out", out],
      ],
      {
        runGit: fake.runGit,
        runCargoTree: fake.runCargoTree,
        runCargo: () => report,
        // Bytes keyed to the CHECKED-OUT revision, not to a call counter: a counter makes two
        // builds of the same revision disagree, and a wiring that builds the captured one twice
        // then sails through. (It did — this fixture was rewritten after that mutation survived.)
        readArtifact: () => {
          built += 1;
          return Buffer.from(`build of ${fake.head()}`);
        },
        readToolchain: () => "rustc 1.96.0",
      },
    );
    const record = JSON.parse(readFileSync(out, "utf8"));
    assert.deepEqual(record.changedClosurePaths, [moved]);
    assert.equal(built, 2, "one build per revision");
    assert.notEqual(record.auditedArtifact.capturedDigest, record.auditedArtifact.compatibleDigest);
    assert.equal(code, 1, "artifacts that differ are a re-capture");
    assert.equal(
      record.featureWitness.capturedDigest,
      record.featureWitness.compatibleDigest,
      "the witness is quiet here — the two layers report independently",
    );
    assert.deepEqual(record.auditedArtifact.adjudicates, [
      "Cargo.toml",
      "Cargo.lock",
      "rust-toolchain.toml",
      ".cargo/config.toml",
      "crates/media/candle-gen/candle-gen",
      "crates/media/candle-gen/candle-gen-flux2",
    ]);
  } finally {
    rmSync(path.dirname(out), { recursive: true, force: true });
    rmSync(workdir, { recursive: true, force: true });
  }
});

test("the closure covers the workspace build inputs, not only the crate trees", () => {
  // sc-17524. `Cargo.lock` is the realized gap: it moved between 5ffd7612 and 277f4238 while all
  // seven crate trees stayed byte-identical, so the free path reported "no build needed" over a
  // changed build input. Asserting the membership rather than only the length means dropping one
  // and adding another cannot keep this green.
  assert.deepEqual([...AUDIT_CLOSURE_BUILD_INPUTS], ["Cargo.toml", "Cargo.lock", "rust-toolchain.toml", ".cargo/config.toml"]);
  assert.equal(AUDIT_CLOSURE_PATHS.length, 10);
  assert.deepEqual([...AUDIT_CLOSURE_PATHS], [...AUDIT_CLOSURE_BUILD_INPUTS, ...AUDIT_CLOSURE_CRATES]);
  assert.equal(
    new Set(AUDIT_CLOSURE_PATHS).size,
    AUDIT_CLOSURE_PATHS.length,
    "a duplicated path would be audited twice and validated as a closure one entry short",
  );
});
