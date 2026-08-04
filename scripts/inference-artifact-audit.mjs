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
// ## The fast path is the point
//
// When all seven objects are byte-identical there is nothing to compile and the script says so
// without building anything, exactly as today. The build only happens when a path actually moved —
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

/** FLUX.2's complete Candle/CUDA compile closure — the sc-15833 audit set, unchanged. */
export const AUDIT_CLOSURE_PATHS = Object.freeze([
  "Cargo.toml",
  "crates/contracts/gen-core",
  "crates/bundles/runtime-cuda",
  "crates/media/candle-gen/candle-gen",
  "crates/media/candle-gen/candle-gen-pid",
  "crates/media/candle-gen/candle-gen-flux2",
  "crates/media/candle-gen/vendor/candle-kernels",
]);

export const SCHEMA_VERSION = 2;
export const AUDIT_STORY = "SC-15833";
export const AUDIT_METHOD =
  "compiled artifact identity for changed paths, git object identity for unchanged paths, across " +
  "the complete Candle FLUX.2 runtime dependency closure";

/** The measurement target. `tests::` is a lib-inline module, so this is the `--lib` test binary. */
export const AUDIT_ARTIFACT_TARGET = Object.freeze({
  package: "candle-gen-flux2",
  kind: "lib test binary",
  test: "tests::flux2_dev_probed_generate_for_offload_ab",
  profile: "release",
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
}) {
  runGit(workdir, ["checkout", "--detach", revision]);
  // Both spellings: on macOS `os.tmpdir()` hands back `/var/folders/...` while cargo and rustc see
  // the canonical `/private/var/folders/...`, and a remap that does not match the path rustc is
  // given silently does nothing.
  const remap = [
    ...new Set([workdir, resolvedPath(workdir)]),
  ]
    .map((root) => `--remap-path-prefix=${root}=/inference`)
    .concat(`--remap-path-prefix=${cargoHome()}=/cargo`)
    .join(" ");
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
      RUSTFLAGS: [process.env.RUSTFLAGS, remap].filter(Boolean).join(" "),
    },
  });
  const executable = selectMeasurementExecutable(report);
  return {
    executable,
    digest: digestBytes(readArtifact(executable)),
    covered: coveredClosurePaths(report, workdir),
  };
}

/**
 * The closure paths the built binary actually links — the only ones its digest can adjudicate.
 *
 * This is not a detail. `runtime-cuda` depends on `candle-gen-flux2`, NOT the other way round, so a
 * commit into `crates/bundles/runtime-cuda` leaves the measurement binary byte-identical. Treating
 * an unchanged digest as proof over the whole closure would be a false green on that path — the one
 * failure mode strictly worse than the false positive this story set out to remove.
 *
 * Derived from cargo's own report of what it compiled (`compiler-artifact` is emitted for fresh
 * units too, so a warm cache does not shrink the set) rather than from a hand-maintained list, so it
 * cannot drift away from the dependency graph.
 */
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
    const relative = path.relative(root, resolvedPath(message.manifest_path));
    if (relative.startsWith("..") || path.isAbsolute(relative)) continue;
    for (const objectPath of AUDIT_CLOSURE_PATHS) {
      if (relative === objectPath || relative.startsWith(`${objectPath}/`)) covered.add(objectPath);
    }
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
  const result = spawnSync("cargo", args, { cwd, env, encoding: "utf8", maxBuffer: 512 * 1024 * 1024 });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    throw new Error(`cargo ${args.slice(0, 4).join(" ")} failed in ${cwd}:\n${result.stderr.slice(-4000)}`);
  }
  return result.stdout.split("\n");
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
    "  --lane cuda   (default) the only lane whose record can authorize a FLUX.2 calibration.\n" +
    "  --lane metal  builds the same closure on Apple Silicon. Produces a deliberately inert record:\n" +
    "                every consumer requires lane \"cuda\". For exercising this script, not for proof.\n"
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
    const workdir = value("--workdir") ?? mkdtempSync(path.join(os.tmpdir(), "sceneworks-audit-build-"));
    const cargoTargetDir = path.join(workdir, "..", `${path.basename(workdir)}-target`);
    git(repo, ["worktree", "add", "--detach", workdir, captured]);
    try {
      const rustc = execFileSync("rustc", ["--version"], { cwd: workdir, encoding: "utf8" }).trim();
      process.stderr.write(`[audit] building ${captured} (${lane}, ${rustc})\n`);
      const capturedBuild = buildMeasurementBinary({ workdir, revision: captured, lane, cargoTargetDir });
      process.stderr.write(`[audit] building ${compatible} (${lane}, ${rustc})\n`);
      const compatibleBuild = buildMeasurementBinary({ workdir, revision: compatible, lane, cargoTargetDir });
      artifact = {
        ...AUDIT_ARTIFACT_TARGET,
        lane,
        features: [LANE_FEATURES[lane]],
        rustc,
        cudaRelease,
        // Both builds must agree on what was linked; a feature or resolution change between the two
        // revisions could otherwise silently shrink what the digest speaks for.
        adjudicates: capturedBuild.covered.filter((objectPath) => compatibleBuild.covered.includes(objectPath)),
        capturedDigest: capturedBuild.digest,
        compatibleDigest: compatibleBuild.digest,
      };
    } finally {
      git(repo, ["worktree", "remove", "--force", workdir]);
      rmSync(cargoTargetDir, { recursive: true, force: true });
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

if (import.meta.url === new URL(`file://${process.argv[1]}`).href) {
  main(process.argv.slice(2)).then(
    (code) => process.exit(code),
    (error) => {
      process.stderr.write(`${error.message}\n`);
      process.exit(1);
    },
  );
}
