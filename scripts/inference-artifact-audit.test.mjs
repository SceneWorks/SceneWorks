import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtempSync, mkdirSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import {
  AUDIT_ARTIFACT_TARGET,
  AUDIT_CLOSURE_BUILD_INPUTS,
  AUDIT_CLOSURE_CRATES,
  AUDIT_CLOSURE_PATHS,
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
  reproducibleLinkFlags,
  resolveRevision,
  selectMeasurementExecutable,
} from "./inference-artifact-audit.mjs";

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
  const clean = auditRecord({ captured: OBJECT("a"), compatible: OBJECT("b"), pairs: pairs() });
  assert.equal(clean.schemaVersion, 3);
  assert.deepEqual(clean.changedClosurePaths, []);
  assert.ok(!("auditedArtifact" in clean), "claiming an artifact proof that was never built would be a lie");

  const moved = pairs({ moved: ["crates/media/candle-gen/candle-gen"] });
  assert.throws(
    () => auditRecord({ captured: OBJECT("a"), compatible: OBJECT("b"), pairs: moved }),
    /closure paths moved .* but no compiled-artifact proof/,
  );
  const proven = auditRecord({
    captured: OBJECT("a"),
    compatible: OBJECT("b"),
    pairs: moved,
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
