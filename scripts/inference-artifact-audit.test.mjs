import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtempSync, mkdirSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import {
  AUDIT_ARTIFACT_TARGET,
  AUDIT_CLOSURE_PATHS,
  assertRealCudaCompiler,
  auditRecord,
  buildMeasurementBinary,
  changedClosurePaths,
  closureObjectPairs,
  coveredClosurePaths,
  digestBytes,
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
    const file = objectPath.endsWith(".toml") ? path.join(repo, objectPath) : path.join(repo, objectPath, "src.rs");
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
  assert.equal(clean.schemaVersion, 2);
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
  const build = (revision) =>
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
    assert.match(env.RUSTFLAGS, /--remap-path-prefix=\/w=\/inference/);
    assert.match(env.RUSTFLAGS, /--remap-path-prefix=.*=\/cargo/);
  }
  assert.notEqual(first.digest, second.digest, "different bytes, different digest");
  assert.equal(first.executable, "/w/target/deps/candle_gen_flux2-1");
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

test("the audited artifact names the target that actually produced the measurements", () => {
  // sc-15833's approved capture command runs exactly this test in this package; auditing anything
  // else would be auditing code the calibration never ran.
  assert.equal(AUDIT_ARTIFACT_TARGET.package, "candle-gen-flux2");
  assert.equal(AUDIT_ARTIFACT_TARGET.test, "tests::flux2_dev_probed_generate_for_offload_ab");
  assert.equal(AUDIT_ARTIFACT_TARGET.profile, "release");
  assert.equal(AUDIT_CLOSURE_PATHS.length, 7);
});
