#!/usr/bin/env node
// sc-17497: prove a FLUX.2 calibration captured at inference revision X still describes the code at
// live pin Y by hashing the COMPILED ARTIFACT, not the source tree.
//
// ## What was wrong with hashing the source tree
//
// sc-15833's audit compares `git rev-parse <revision>:<path>` over the seven objects in FLUX.2's
// Candle/CUDA compile closure. Object identity needs no judgment and cannot be faked, which is why
// it was chosen — but its unit is a CRATE TREE, so any commit into one of those seven invalidates
// the proof, including commits that provably cannot change the compiled code.
//
// That cost a full re-capture once already. inference `35251a88` added 42 lines of `//!` doc comment
// to `candle-gen/src/preview.rs`; six of the seven objects stayed byte-identical, that one moved,
// and five `flux2_dev` q4 cells dropped to `historical` — remediable only by a ~47.6 GB-peak capture
// on an RTX PRO 6000.
//
// ## The unit this script hashes, and why it is the LINKED binary
//
// "The compiled code is identical" is the claim, so hash the thing that runs. Three candidate units
// were measured (sc-17497 working notes; the fixture is reproduced in the test file):
//
//   - the RLIB           — WRONG. rustc metadata carries doc strings, so `libdep.rlib` moved on the
//                          exact doc-comment edit this story exists to absorb. Hashing rlibs would
//                          reproduce the false positive verbatim.
//   - the rlib's `.o`s   — WRONG, and dangerously so. A `#[inline]`/generic body is codegen'd in the
//                          CONSUMER, not its own crate: mutating `#[inline] fn scale` from `*7` to
//                          `*8` left the defining crate's object members byte-identical. A per-crate
//                          object hash is a false green over most of a numerics crate.
//   - the LINKED binary  — RIGHT. Invariant across the doc comment, and it moved on the `*7 -> *8`
//                          mutation. Code that link-time DCE removes is by definition code that does
//                          not run, so excluding it does not weaken the claim.
//
// The binary hashed is the one that PRODUCED the measurements: the `candle-gen-flux2` lib test
// binary carrying `tests::flux2_dev_probed_generate_for_offload_ab`, the exact target
// `scripts/sc-15833-flux2-evidence.mjs` prints as the approved capture command.
//
// ## Why a stub `nvcc` is refused rather than tolerated
//
// `scripts/check-candle-build.mjs` can typecheck the candle lane on a GPU-less box because
// `cargo check` never links. Its stub `nvcc` writes EMPTY `.ptx` files. That is fine for a typecheck
// and catastrophic here: `candle-kernels` `include_str!`s the PTX it was handed, so a stub-built
// artifact embeds nothing of the kernels and is blind to every `.cu` change in the closure — a
// silent false green over one of the seven audited objects. `--lane cuda` therefore probes the
// compiler with a real compile and refuses anything that does not emit real PTX.
//
// ## What the closure is, and why it grew (sc-17524)
//
// sc-15833 declared it as seven crate trees. Two root-level siblings that feed every one of those
// builds were not among them, and one of the gaps was already realized inside the authorized
// window: `Cargo.lock` moved between `5ffd7612` and `277f4238` while all seven trees stayed
// byte-identical, so the free path printed "no build needed" over a changed build input. A
// `cargo update` bumping a transitive dependency the measurement binary links moves only that file.
// `rust-toolchain.toml` is the same shape. Both are in the closure now, so either one moving forces
// the artifact layer instead of sailing through the free path.
//
// They are also, unlike a crate tree, adjudicable by ANY successful build of the audited target —
// see `coveredClosurePaths` for why, and for why switching the audited artifact to `runtime-cuda`'s
// bundle test binary (the other candidate) was measured and rejected.
//
// ## The fast path is the point
//
// When all nine objects are byte-identical there is nothing to compile and the script says so
// without building anything, exactly as before. The build only happens when a path actually moved —
// "cheap and automatic when nothing compiled changed, loud only when something did".
//
// ## Build discipline (why the two builds share one directory)
//
// `file!()` bakes source paths into panic locations, so the two revisions must be compiled at the
// SAME path or the digests differ for reasons unrelated to source. The script checks both revisions
// out into one detached worktree and reuses it, and additionally remaps the worktree and CARGO_HOME
// out of the artifact. `CARGO_INCREMENTAL=0` because incremental artifacts are not reproducible, and
// the toolchain is whatever `rust-toolchain.toml` pins (1.96.0 today) — recorded, and required to be
// identical across both builds, because a rustc bump genuinely does change the compiled code.

import { createHash } from "node:crypto";
import { execFileSync, spawnSync } from "node:child_process";
import { mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { pathToFileURL } from "node:url";

/**
 * Workspace-level build inputs (sc-17524). Not packages, so cargo never names them.
 *
 * sc-15833 declared the closure as crate trees only, which left these outside it. `Cargo.lock` is
 * the realized gap: it moved between `5ffd7612` and `277f4238` while all seven audited crate trees
 * stayed byte-identical, so the free path reported "no build needed" over a changed build input. A
 * `cargo update` bumping a semver-compatible transitive dependency the measurement binary actually
 * links moves ONLY this file. `rust-toolchain.toml` is the same shape — a rustc bump genuinely does
 * change the compiled code, and nothing on the free path looked at it.
 */
export const AUDIT_CLOSURE_BUILD_INPUTS = Object.freeze([
  "Cargo.toml",
  "Cargo.lock",
  "rust-toolchain.toml",
]);

/** The crate trees. These reach the audited binary exactly one way: by being compiled into it. */
export const AUDIT_CLOSURE_CRATES = Object.freeze([
  "crates/contracts/gen-core",
  "crates/bundles/runtime-cuda",
  "crates/media/candle-gen/candle-gen",
  "crates/media/candle-gen/candle-gen-pid",
  "crates/media/candle-gen/candle-gen-flux2",
  "crates/media/candle-gen/vendor/candle-kernels",
]);

/** FLUX.2's complete Candle/CUDA compile closure: sc-15833's seven paths plus sc-17524's two. */
export const AUDIT_CLOSURE_PATHS = Object.freeze([
  ...AUDIT_CLOSURE_BUILD_INPUTS,
  ...AUDIT_CLOSURE_CRATES,
]);

// v3 is sc-17524's nine-path closure. v1/v2 records describe the seven-path one and are refused
// outright rather than re-graded: a record that never looked at `Cargo.lock` cannot be re-read as
// evidence about it, and a silent re-grade is how an audit ends up asserting more than it measured.
export const SCHEMA_VERSION = 3;
export const AUDIT_STORY = "SC-15833";
export const AUDIT_METHOD =
  "compiled artifact identity for changed paths, git object identity for unchanged paths, across " +
  "the complete Candle FLUX.2 runtime dependency closure and its workspace build inputs";

/**
 * The measurement target. `tests::` is a lib-inline module, so this is the `--lib` test binary.
 *
 * `closurePath` is the crate tree this package lives in, and it is the gate on adjudicating build
 * inputs: their coverage is inferred from "a real build of this target happened", so it has to be
 * anchored to cargo having reported compiling this very package. See `coveredClosurePaths`.
 */
export const AUDIT_ARTIFACT_TARGET = Object.freeze({
  package: "candle-gen-flux2",
  kind: "lib test binary",
  test: "tests::flux2_dev_probed_generate_for_offload_ab",
  profile: "release",
  closurePath: "crates/media/candle-gen/candle-gen-flux2",
});

/** `cuda` is the only lane that can authorize a FLUX.2 calibration; `metal` exists so this script
 *  can be exercised end to end on a Mac, and its records are inert by construction — every consumer
 *  requires `lane === "cuda"`. */
export const LANE_FEATURES = Object.freeze({
  cuda: "cuda",
  metal: "metal",
});

const CUDA_PROBE_SOURCE = "__global__ void sceneworks_audit_probe(float *x) { x[0] = 1.0f; }\n";

function git(repo, args) {
  return execFileSync("git", ["-C", repo, ...args], { encoding: "utf8", maxBuffer: 64 * 1024 * 1024 }).trim();
}

/** Full 40-hex revision for `revision`, so a record can never name an abbreviation. */
export function resolveRevision(repo, revision, { runGit = git } = {}) {
  const resolved = runGit(repo, ["rev-parse", `${revision}^{commit}`]);
  if (!/^[0-9a-f]{40}$/.test(resolved)) {
    throw new Error(`${revision}: did not resolve to a 40-hex commit in ${repo}`);
  }
  return resolved;
}

/**
 * `{ path, capturedObject, compatibleObject }` for every closure path, in declaration order.
 *
 * `git rev-parse <revision>:<path>` is the same command the v1 records were produced with; keeping
 * it means the cheap layer of the proof stays independently re-derivable by anyone with a checkout.
 */
export function closureObjectPairs({ repo, captured, compatible, runGit = git }) {
  return AUDIT_CLOSURE_PATHS.map((objectPath) => ({
    path: objectPath,
    capturedObject: runGit(repo, ["rev-parse", `${captured}:${objectPath}`]),
    compatibleObject: runGit(repo, ["rev-parse", `${compatible}:${objectPath}`]),
  }));
}

/** The closure paths whose objects moved. Empty means the fast path applies and nothing is built. */
export function changedClosurePaths(pairs) {
  return pairs.filter((pair) => pair.capturedObject !== pair.compatibleObject).map((pair) => pair.path);
}

/**
 * Refuse a `nvcc` that cannot actually compile. The stub in `check-candle-build.mjs` answers
 * `--version` convincingly and then writes an EMPTY `.ptx`, which would silently erase
 * `candle-kernels` from the audited artifact — so version output is not evidence. Compile a real
 * kernel and require real PTX back.
 */
export function assertRealCudaCompiler({ runProbe = defaultCudaProbe } = {}) {
  const probe = runProbe(CUDA_PROBE_SOURCE);
  if (!probe.ok) {
    throw new Error(
      "--lane cuda needs a real CUDA toolkit: `nvcc` is absent or failed to compile the probe kernel " +
        `(${probe.reason}). This audit cannot be produced on a box without one — the stub compiler ` +
        "used by `npm run rust:check:candle` emits empty PTX, which would drop candle-kernels out of " +
        "the hashed artifact entirely.",
    );
  }
  if (!/\.target\s+sm_\d+/.test(probe.ptx) || probe.ptx.trim().length < 64) {
    throw new Error(
      "`nvcc` produced no usable PTX for the probe kernel. This is the signature of the stub compiler " +
        "written by scripts/check-candle-build.mjs; an artifact built against it is blind to every " +
        "kernel change in the closure. Run this on a box with a real CUDA toolkit.",
    );
  }
  return probe.release;
}

function defaultCudaProbe(source) {
  const dir = mkdtempSync(path.join(os.tmpdir(), "sceneworks-audit-nvcc-"));
  try {
    const cu = path.join(dir, "probe.cu");
    const ptx = path.join(dir, "probe.ptx");
    writeFileSync(cu, source);
    const compiled = spawnSync("nvcc", ["--ptx", cu, "-o", ptx], { encoding: "utf8" });
    if (compiled.error || compiled.status !== 0) {
      return { ok: false, reason: compiled.error ? compiled.error.message : compiled.stderr.trim().slice(0, 200) };
    }
    const version = spawnSync("nvcc", ["--version"], { encoding: "utf8" });
    const release = /release (\d+\.\d+)/.exec(version.stdout ?? "")?.[1] ?? "unknown";
    return { ok: true, ptx: readFileSync(ptx, "utf8"), release };
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

export function digestBytes(bytes) {
  return `sha256:${createHash("sha256").update(bytes).digest("hex")}`;
}

/**
 * Build the measurement binary for `revision` inside `workdir` and return its digest.
 *
 * `--message-format=json` rather than a glob over `target/`: cargo names test binaries with a
 * metadata hash, and picking the wrong file (an integration test, a stale binary from the other
 * revision) would compare something that is not the measurement target. The emitted record is only
 * as good as this selection, so it is taken from cargo's own report of what it just built.
 */
export function buildMeasurementBinary({
  workdir,
  revision,
  lane,
  cargoTargetDir,
  runGit = git,
  runCargo = defaultCargoRunner,
  readArtifact = (file) => readFileSync(file),
  readToolchain = defaultToolchain,
}) {
  runGit(workdir, ["checkout", "--detach", revision]);
  // AFTER the checkout, because the checkout also swaps `rust-toolchain.toml` and rustup will
  // silently hand the second build a different compiler. Reading this once up front — as this did
  // originally — records a toolchain that may not be the one that built both artifacts.
  const rustc = readToolchain(workdir);
  // Both spellings: on macOS `os.tmpdir()` hands back `/var/folders/...` while cargo and rustc see
  // the canonical `/private/var/folders/...`, and a remap that does not match the path rustc is
  // given silently does nothing.
  const remap = [...new Set([workdir, resolvedPath(workdir)])]
    .map((root) => `--remap-path-prefix=${root}=/inference`)
    .concat(`--remap-path-prefix=${cargoHome()}=/cargo`)
    .concat(reproducibleLinkFlags());
  const report = runCargo({
    cwd: workdir,
    args: [
      "test",
      "-p",
      AUDIT_ARTIFACT_TARGET.package,
      "--lib",
      "--release",
      "--locked",
      "--no-run",
      "--features",
      LANE_FEATURES[lane],
      "--message-format=json",
    ],
    env: {
      ...process.env,
      CARGO_TARGET_DIR: cargoTargetDir,
      CARGO_INCREMENTAL: "0",
      // CARGO_ENCODED_RUSTFLAGS, not RUSTFLAGS: cargo splits the latter on WHITESPACE, and the box
      // this has to run on is Windows, where the default temp directory sits under a profile path
      // that routinely contains a space ("C:\Users\Michael Trefry\AppData\Local\Temp\..."). A
      // space-joined remap there is silently truncated into arguments that match nothing. The
      // encoded form is 0x1f-separated and takes arbitrary values.
      CARGO_ENCODED_RUSTFLAGS: encodedRustflags(remap),
      RUSTFLAGS: undefined,
    },
  });
  const executable = selectMeasurementExecutable(report);
  return {
    executable,
    rustc,
    digest: digestBytes(readArtifact(executable)),
    covered: coveredClosurePaths(report, workdir),
  };
}

function defaultToolchain(workdir) {
  return execFileSync("rustc", ["--version"], { cwd: workdir, encoding: "utf8" }).trim();
}

/**
 * Flags that make the linked artifact a function of the code rather than of the clock.
 *
 * `link.exe` stamps the LINK TIME into the PE image, so two builds of identical source, at the same
 * path, under the same compiler, produce different bytes. That is fatal here and it is not
 * theoretical: running this for real on the RTX box, `5ffd7612` built twice gave `sha256:2164e988…`
 * and `sha256:57f15abb…`, and `06e0c5e9` gave `sha256:feb9ea68…` and `sha256:b5fbcd64…`. Every
 * CUDA-lane comparison was therefore guaranteed to report ARTIFACTS DIFFER — a permanent,
 * unfalsifiable "the compiled code changed, re-capture on an RTX PRO 6000", which is worse than the
 * false positive this audit exists to remove.
 *
 * Byte-diffing two such builds located it exactly: 21 bytes out of 181 MB, and not one of them in
 * `.text`. The COFF `TimeDateStamp` at `0x138`, its copy in each debug-directory entry, and the
 * 17-byte CodeView PDB GUID+age. Codegen was already deterministic; only the stamps were not.
 *
 * BOTH flags are needed, which a two-link experiment on a hello-world binary settled in seconds
 * rather than by reasoning:
 *
 *   baseline                         DIFFERS
 *   /Brepro                          DIFFERS   <- turns the timestamps into a hash OF THE IMAGE,
 *                                                 which still contains the varying PDB signature
 *   /Brepro -Cstrip=symbols          DIFFERS
 *   /Brepro -Cdebuginfo=0            DIFFERS   <- rustc passes /DEBUG on MSVC regardless
 *   /DEBUG:NONE                      IDENTICAL
 *   /Brepro /DEBUG:NONE              IDENTICAL <- what this returns
 *
 * `/DEBUG:NONE` is the one that decides it: it stops `link.exe` emitting a PDB at all, so there is
 * no signature left to vary. `/Brepro` is kept because it is the documented way to ask for a
 * content-derived timestamp, and it costs nothing to keep the COFF stamp pinned explicitly rather
 * than relying on a side effect of not writing debug info.
 *
 * Dropping the PDB does not weaken the claim. The audited binary is built `--no-run` and never
 * executes; debug info is not code, `.text` is byte-identical with and without it, and the one
 * thing it can hide — a symbol or local renamed with no codegen change — is not a change in the
 * compiled code either.
 *
 * The Metal lane needs nothing: Mach-O carries no link timestamp, and sc-17497's `277f -> d2216f6b
 * -> 277f` round trip on macOS was already byte-stable, which is exactly why this went unseen.
 *
 * Reproducibility is not a nicety in this script — it IS the proof. A digest that moves on its own
 * cannot distinguish "the code changed" from "the clock did".
 */
export function reproducibleLinkFlags(platform = process.platform) {
  return platform === "win32" ? ["-Clink-arg=/Brepro", "-Clink-arg=/DEBUG:NONE"] : [];
}

/**
 * The closure paths the built binary can speak for — the only ones its digest may adjudicate.
 *
 * Two kinds of entry, adjudicated two different ways:
 *
 *   - **Crate trees** are covered when cargo reports having COMPILED a package whose manifest lives
 *     under them. This is not a detail. `runtime-cuda` depends on `candle-gen-flux2`, NOT the other
 *     way round, so a commit into `crates/bundles/runtime-cuda` leaves the measurement binary
 *     byte-identical, and reading an unchanged digest as proof over it would be a false green — the
 *     one failure mode strictly worse than the false positive this audit set out to remove.
 *   - **Build inputs** (`Cargo.lock`, `rust-toolchain.toml`, the virtual root `Cargo.toml`) can
 *     never appear here, because cargo's `compiler-artifact` stream only ever names *package*
 *     manifests. They are covered anyway, and soundly: they are inputs to THIS build. A lockfile
 *     bump that moves a dependency the binary links, a `[workspace.dependencies]` edit, a rustc
 *     bump — each of them recompiles the binary, so each of them shows up in its digest. One that
 *     leaves the digest byte-identical (sc-17524's real example: `sha2` added to mlx crates and
 *     `candle-gen-sensenova`) provably did not reach the measured code.
 *
 * The build-input inference is gated on the audited package's own tree being covered, so it cannot
 * be reached by an empty or unrecognized cargo report: without that gate a build that compiled
 * nothing would still hand back three adjudicable paths.
 *
 * sc-17524 considered switching the audited artifact to `runtime-cuda`'s test binary instead, to
 * make the whole closure adjudicable by compile coverage alone. Measured and rejected: that binary
 * links the ENTIRE CUDA bundle, so its digest moves on any commit to any of ~50 provider crates.
 * Over the very window this had to adjudicate (`5ffd7612` -> `06e0c5e9`) `candle-gen-catalog`,
 * `candle-gen-chroma`, `candle-gen-sdxl` and `candle-gen-sensenova` all moved, none of them in
 * FLUX.2's closure. It would have reported "the compiled code changed, re-capture" for FLUX.2 code
 * that did not change — the 47.6 GB false positive this epic exists to remove, with a wider trigger.
 *
 * Crate coverage is derived from cargo's own report of what it compiled (`compiler-artifact` is
 * emitted for fresh units too, so a warm cache does not shrink the set) rather than from a
 * hand-maintained list, so it cannot drift away from the dependency graph.
 */
/**
 * A manifest path as a repo-relative POSIX path, or `null` if it is outside `root` (sc-17587).
 *
 * Separators here are not cosmetics. `path.relative` returns NATIVE ones, so on Windows this was
 * handing back `crates\contracts\gen-core\Cargo.toml` and comparing it against the forward-slash
 * `AUDIT_CLOSURE_PATHS` entries. Nothing ever matched, the covered set came back EMPTY, and the
 * tool refused its own build — on the one box that can produce a `cuda` record at all.
 *
 * `platformPath` is injectable for exactly one reason: on Linux and macOS `path.sep` is already
 * `/`, so a test that uses the host's own implementation passes whether or not the normalisation
 * is there. That is how the bug reached main green. Driving `path.win32` explicitly makes the
 * regression test fail on ANY host if this is dropped.
 */
export function closureRelativePath(root, manifestPath, platformPath = path) {
  const native = platformPath.relative(root, manifestPath);
  if (native.startsWith("..") || platformPath.isAbsolute(native)) return null;
  return native.split(platformPath.sep).join("/");
}

export function coveredClosurePaths(cargoJsonLines, workdir) {
  const covered = new Set();
  // `realpathSync`, not the raw path: `os.tmpdir()` is `/var/folders/...` on macOS and cargo reports
  // manifests under the canonical `/private/var/folders/...`. Comparing the two spellings resolves
  // every closure path to "outside the worktree" and reports an EMPTY covered set — which the caller
  // then correctly refuses, turning a path bug into a hard stop. Found by running this for real.
  const root = resolvedPath(workdir);
  for (const line of cargoJsonLines) {
    let message;
    try {
      message = JSON.parse(line);
    } catch {
      continue;
    }
    if (message.reason !== "compiler-artifact" || typeof message.manifest_path !== "string") continue;
    const relative = closureRelativePath(root, resolvedPath(message.manifest_path));
    if (relative === null) continue;
    for (const objectPath of AUDIT_CLOSURE_CRATES) {
      if (relative === objectPath || relative.startsWith(`${objectPath}/`)) covered.add(objectPath);
    }
  }
  // Anchored to the audited package itself: "this really was a build of the target we hash".
  if (covered.has(AUDIT_ARTIFACT_TARGET.closurePath)) {
    for (const objectPath of AUDIT_CLOSURE_BUILD_INPUTS) covered.add(objectPath);
  }
  return [...AUDIT_CLOSURE_PATHS].filter((objectPath) => covered.has(objectPath));
}

/**
 * The one executable cargo reports for the package's LIB test target. Anything else — zero matches,
 * or more than one — is a build whose output this script cannot identify, and guessing is how an
 * audit ends up hashing the wrong file.
 */
export function selectMeasurementExecutable(cargoJsonLines) {
  const matches = [];
  for (const line of cargoJsonLines) {
    let message;
    try {
      message = JSON.parse(line);
    } catch {
      continue;
    }
    if (
      message.reason === "compiler-artifact" &&
      message.executable &&
      message.profile?.test === true &&
      Array.isArray(message.target?.kind) &&
      message.target.kind.includes("lib") &&
      message.target?.name === AUDIT_ARTIFACT_TARGET.package.replace(/-/g, "_")
    ) {
      matches.push(message.executable);
    }
  }
  if (matches.length !== 1) {
    throw new Error(
      `expected exactly one ${AUDIT_ARTIFACT_TARGET.package} lib test executable in the cargo report, ` +
        `got ${matches.length}`,
    );
  }
  return matches[0];
}

function defaultCargoRunner({ cwd, args, env }) {
  // stderr inherited, not captured: this is a release build of the FLUX.2 closure and an operator
  // watching a silent terminal for an hour cannot tell it apart from a hang.
  const result = spawnSync("cargo", args, {
    cwd,
    env,
    encoding: "utf8",
    maxBuffer: 512 * 1024 * 1024,
    stdio: ["ignore", "pipe", "inherit"],
  });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    throw new Error(`cargo ${args.slice(0, 4).join(" ")} failed in ${cwd} (see the output above)`);
  }
  return result.stdout.split("\n");
}

/**
 * Flags in cargo's 0x1f-separated encoding, carrying through anything already in the environment.
 *
 * An inherited `RUSTFLAGS` is whitespace-separated by definition, so it is re-split on whitespace;
 * only the flags this script adds — the ones that may contain a space — get to be atomic.
 */
export function encodedRustflags(flags, inherited = process.env.RUSTFLAGS) {
  const carried = (inherited ?? "").split(/\s+/).filter(Boolean);
  return [...carried, ...flags].join("\u001f");
}

/** Canonical form where the path exists, best-effort otherwise (so tests can pass synthetic paths). */
function resolvedPath(target) {
  try {
    return realpathSync(target);
  } catch {
    return path.resolve(target);
  }
}

function cargoHome() {
  return process.env.CARGO_HOME || path.join(os.homedir(), ".cargo");
}

/**
 * Two builds under two compilers are not comparable at all, so this is a hard stop rather than a
 * note in the record. A checkout swaps `rust-toolchain.toml`, and rustup will silently hand the
 * second build a different compiler.
 *
 * sc-17524 made this reachable on purpose. `rust-toolchain.toml` is now a closure path, so a rustc
 * bump can no longer take the free path — it forces the build, which lands here. Extracted from
 * `main` so that can be asserted directly: inside `main` it sat behind two real CUDA builds and
 * nothing exercised it, which for a rule this load-bearing is indistinguishable from not having it.
 */
export function assertComparableToolchains({ captured, compatible, capturedRustc, compatibleRustc }) {
  if (capturedRustc !== compatibleRustc) {
    throw new Error(
      "the two revisions built under different toolchains, so their digests are not comparable:\n" +
        `  ${captured}: ${capturedRustc}\n  ${compatible}: ${compatibleRustc}`,
    );
  }
  return capturedRustc;
}

/**
 * Assemble the record. `auditedArtifact` is present exactly when a path moved: on the fast path
 * there is no artifact to name, and inventing one would claim a build that never ran.
 */
export function auditRecord({ captured, compatible, pairs, artifact = null }) {
  const changed = changedClosurePaths(pairs);
  if (changed.length > 0 && !artifact) {
    throw new Error(`closure paths moved (${changed.join(", ")}) but no compiled-artifact proof was supplied`);
  }
  const unadjudicable = changed.filter((objectPath) => !(artifact?.adjudicates ?? []).includes(objectPath));
  if (unadjudicable.length > 0) {
    throw new Error(
      `the audited artifact does not link ${unadjudicable.join(", ")}, so its digest cannot speak for ` +
        "those paths. They must be object-identical, or a different artifact must be audited.",
    );
  }
  const record = {
    schemaVersion: SCHEMA_VERSION,
    story: AUDIT_STORY,
    capturedInferenceRevision: captured,
    compatibleInferenceRevision: compatible,
    method: AUDIT_METHOD,
    command: "node scripts/inference-artifact-audit.mjs --repo PATH --captured SHA40 --compatible SHA40",
    changedClosurePaths: changed,
    auditedObjects: pairs,
  };
  if (artifact) record.auditedArtifact = artifact;
  return record;
}

function usage() {
  return (
    "usage: node scripts/inference-artifact-audit.mjs --repo PATH --captured SHA --compatible SHA\n" +
    "                                                [--lane cuda|metal] [--workdir PATH] [--out FILE]\n" +
    "\n" +
    "  --lane cuda    (default) the only lane whose record can authorize a FLUX.2 calibration.\n" +
    "  --lane metal   builds the same closure on Apple Silicon. Produces a deliberately inert record:\n" +
    "                 every consumer requires lane \"cuda\". For exercising this script, not for proof.\n" +
    "  --workdir PATH build here instead of a throwaway temp directory, and KEEP the derived\n" +
    "                 `<PATH>-target` afterwards. A cold CUDA build is hours and the remapped paths\n" +
    "                 make the fingerprints path-specific, so this is what makes a re-run warm.\n"
  );
}

export async function main(argv) {
  const value = (flag) => {
    const index = argv.indexOf(flag);
    return index === -1 ? undefined : argv[index + 1];
  };
  const repo = value("--repo");
  const capturedArg = value("--captured");
  const compatibleArg = value("--compatible");
  const lane = value("--lane") ?? "cuda";
  const out = value("--out");
  if (!repo || !capturedArg || !compatibleArg || !LANE_FEATURES[lane]) {
    process.stderr.write(usage());
    return 2;
  }

  const captured = resolveRevision(repo, capturedArg);
  const compatible = resolveRevision(repo, compatibleArg);
  const pairs = closureObjectPairs({ repo, captured, compatible });
  const changed = changedClosurePaths(pairs);

  let artifact = null;
  if (changed.length === 0) {
    process.stderr.write(
      `[audit] all ${AUDIT_CLOSURE_PATHS.length} closure objects are byte-identical; no build needed.\n`,
    );
  } else {
    process.stderr.write(`[audit] closure paths moved: ${changed.join(", ")}\n`);
    const cudaRelease = lane === "cuda" ? assertRealCudaCompiler() : null;
    // An operator-supplied `--workdir` is theirs, and so is the target directory derived from it.
    // A cold CUDA build of this closure is hours (candle-kernels alone is an nvcc pass over the
    // vendored `.cu` set), and `--remap-path-prefix` bakes the worktree path into every rustc
    // fingerprint — so the ONLY way a second run reuses the first one's work is to build at the
    // same path with the same target directory. Deleting it unconditionally, as this did, made
    // every re-run cold and effectively made the tool single-shot on the one box that can run it.
    const requestedWorkdir = value("--workdir");
    const workdir = requestedWorkdir ?? mkdtempSync(path.join(os.tmpdir(), "sceneworks-audit-build-"));
    const cargoTargetDir = path.join(workdir, "..", `${path.basename(workdir)}-target`);
    try {
      git(repo, ["worktree", "add", "--detach", workdir, captured]);
      process.stderr.write(`[audit] building ${captured} (${lane})\n`);
      const capturedBuild = buildMeasurementBinary({ workdir, revision: captured, lane, cargoTargetDir });
      process.stderr.write(`[audit] building ${compatible} (${lane})\n`);
      const compatibleBuild = buildMeasurementBinary({ workdir, revision: compatible, lane, cargoTargetDir });
      assertComparableToolchains({
        captured,
        compatible,
        capturedRustc: capturedBuild.rustc,
        compatibleRustc: compatibleBuild.rustc,
      });
      artifact = {
        ...AUDIT_ARTIFACT_TARGET,
        lane,
        features: [LANE_FEATURES[lane]],
        rustc: capturedBuild.rustc,
        cudaRelease,
        // Both builds must agree on what was linked; a feature or resolution change between the two
        // revisions could otherwise silently shrink what the digest speaks for.
        adjudicates: capturedBuild.covered.filter((objectPath) => compatibleBuild.covered.includes(objectPath)),
        capturedDigest: capturedBuild.digest,
        compatibleDigest: compatibleBuild.digest,
      };
    } finally {
      // `|| true`-style tolerance: if `worktree add` was what failed there is nothing to remove, and
      // masking that error with a second one would hide the real cause.
      try {
        git(repo, ["worktree", "remove", "--force", workdir]);
      } catch {
        rmSync(workdir, { recursive: true, force: true });
      }
      if (requestedWorkdir) {
        // Say so. This is tens of gigabytes of release artifacts, and an operator who does not know
        // it survived will neither reuse it nor delete it.
        process.stderr.write(
          `[audit] kept ${cargoTargetDir} so a re-run at --workdir ${requestedWorkdir} is warm; delete it when done.\n`,
        );
      } else {
        rmSync(cargoTargetDir, { recursive: true, force: true });
      }
    }
    if (artifact.capturedDigest !== artifact.compatibleDigest) {
      process.stderr.write(
        `[audit] ARTIFACTS DIFFER\n  ${captured}: ${artifact.capturedDigest}\n` +
          `  ${compatible}: ${artifact.compatibleDigest}\n` +
          "  The compiled code changed. The calibration must be re-captured; it cannot be extended.\n",
      );
    }
  }

  const record = auditRecord({ captured, compatible, pairs, artifact });
  const body = `${JSON.stringify(record, null, 2)}\n`;
  if (out) {
    writeFileSync(out, body);
    process.stderr.write(`[audit] wrote ${out}\n`);
  } else {
    process.stdout.write(body);
  }
  return artifact && artifact.capturedDigest !== artifact.compatibleDigest ? 1 : 0;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main(process.argv.slice(2)).then(
    (code) => process.exit(code),
    (error) => {
      process.stderr.write(`${error.message}\n`);
      process.exit(1);
    },
  );
}
