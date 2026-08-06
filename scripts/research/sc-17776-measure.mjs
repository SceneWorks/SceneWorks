#!/usr/bin/env node
// sc-17776 RESEARCH INSTRUMENT — the driver that turns `sc-17776-reachability-probe.mjs` into a
// measurement of a real revision pair. Not a validator and not wired into any gate.
//
// It reproduces `inference-artifact-audit.mjs`'s build discipline exactly — same worktree, same
// `--remap-path-prefix` set, same `/Brepro /DEBUG:NONE`, same `CARGO_INCREMENTAL=0`, same target
// directory — because a reachability digest computed under different flags is not comparable with
// the whole-binary digest it is being weighed against. The one addition is `-Csave-temps`, passed
// as a TRAILING `cargo rustc` argument rather than through `CARGO_ENCODED_RUSTFLAGS`: trailing args
// apply to the selected unit only, so the ~700 dependency crates stay warm instead of rebuilding.
//
// The dependencies' objects need no such trick — they are already `.o` members inside the `.rlib`
// files cargo left in `target/release/deps`. Only the audited crate's own codegen units are
// transient, because the lib TEST crate is linked into a binary rather than archived.

import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { pathToFileURL } from "node:url";
import { existsSync, mkdirSync, readFileSync, readdirSync, rmSync, utimesSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";

import { reproducibleLinkFlags, encodedRustflags } from "../inference-artifact-audit.mjs";
import { probe } from "./sc-17776-reachability-probe.mjs";

const PACKAGE = "candle-gen-flux2";

function run(command, args, options = {}) {
  return execFileSync(command, args, {
    encoding: "utf8",
    maxBuffer: 1 << 28,
    shell: false,
    ...options,
  });
}

function cargoHome() {
  return process.env.CARGO_HOME || path.join(os.homedir(), ".cargo");
}

/**
 * The rlibs and objects `link.exe` was actually handed, recovered from `--print=link-args`.
 *
 * Globbing `target/release/deps/*.rlib` instead would be wrong in a way that is hard to see: that
 * directory accumulates every rlib any build in this target dir ever produced, including two
 * versions of the same crate (`safetensors` 0.4 and 0.7 both live there) and rlibs left behind by
 * the OTHER revision. Feeding those to the walk adds duplicate definitions of the same symbol, and
 * whichever copy wins decides what the digest describes.
 */
export function linkInputs(report) {
  const tokens = report.match(/"[^"]+\.(?:rlib|o|obj)"|\S+\.(?:rlib|o|obj)\b/g) ?? [];
  const files = new Set();
  for (const token of tokens) {
    const file = token.replace(/^"|"$/g, "");
    if (existsSync(file)) files.add(path.resolve(file));
  }
  return [...files].sort();
}

/** The one lib-test executable cargo reports, selected the way the shipped audit selects it. */
function selectExecutable(report) {
  const executables = report
    .split("\n")
    .filter(Boolean)
    .flatMap((line) => {
      try {
        return [JSON.parse(line)];
      } catch {
        return [];
      }
    })
    .filter(
      (message) =>
        message?.reason === "compiler-artifact" &&
        message.executable &&
        message.target?.test === true &&
        message.target?.name === PACKAGE.replace(/-/g, "_"),
    );
  if (executables.length !== 1) {
    throw new Error(`expected exactly one ${PACKAGE} lib test executable, got ${executables.length}`);
  }
  return executables[0].executable;
}

export function buildEnv({ workdir, cargoTargetDir }) {
  const remap = [...new Set([workdir, path.resolve(workdir)])]
    .map((root) => `--remap-path-prefix=${root}=/inference`)
    .concat(`--remap-path-prefix=${cargoHome()}=/cargo`)
    .concat(reproducibleLinkFlags());
  return {
    ...process.env,
    CARGO_TARGET_DIR: cargoTargetDir,
    CARGO_INCREMENTAL: "0",
    CARGO_ENCODED_RUSTFLAGS: encodedRustflags(remap),
    RUSTFLAGS: undefined,
  };
}

/**
 * Build the lib test crate at `revision` with its codegen units preserved, and return where they
 * landed alongside the whole-binary digest the shipped audit would have computed.
 *
 * `save-temps` writes `<crate>-<hash>.<cgu>.rcgu.o` into the deps directory and leaves them there,
 * so the previous revision's objects have to be swept first or the walk would see both revisions'
 * code at once — which reads as a spurious DIFFER in one direction and, worse, can hide a deletion
 * in the other.
 */
export function buildWithObjects({ workdir, revision, cargoTargetDir, lane = "cuda", patch }) {
  // `reset --hard` first: the previous case may have left a mutation patch applied, and `checkout`
  // refuses rather than clobbering it. This worktree is scratch — the audit tool creates and deletes
  // it — so there is nothing here worth preserving between cases.
  run("git", ["-C", workdir, "reset", "--hard"]);
  run("git", ["-C", workdir, "clean", "-fd", "--", "crates"]);
  run("git", ["-C", workdir, "checkout", "--detach", revision]);
  if (patch) run("git", ["-C", workdir, "apply", path.resolve(patch)]);
  // Force the audited crate to recompile even when the checkout put back byte-identical sources.
  // Without this, cargo takes a fingerprint hit, `-Csave-temps` emits nothing, and the walk would
  // silently re-read whichever revision's objects happened to survive on disk. Safe for the digest:
  // nothing timestamp-derived is hashed — only section contents and relocation target names.
  const touched = path.join(workdir, "crates/media/candle-gen/candle-gen-flux2/src/lib.rs");
  const now = new Date();
  utimesSync(touched, now, now);
  const deps = path.join(cargoTargetDir, "release", "deps");
  try {
    for (const entry of readdirSync(deps)) {
      if (/^candle_gen_flux2-.*\.(o|obj|bc|ll|s)$/.test(entry)) rmSync(path.join(deps, entry));
    }
  } catch {
    /* first run: the directory may not exist yet */
  }
  const env = buildEnv({ workdir, cargoTargetDir });
  // Build the audited artifact with the SHIPPED audit's own command first, so every row carries a
  // whole-binary digest directly comparable with the checked-in records. The object-emitting build
  // below adds `-Csave-temps --print=link-args` to the unit, which changes its rustc command line
  // and therefore its bytes: a digest from that build could not be checked against `d80844f2…`.
  const shipped = run(
    "cargo",
    [
      "test",
      "-p",
      PACKAGE,
      "--lib",
      "--release",
      "--locked",
      "--no-run",
      "--features",
      lane,
      "--message-format=json",
    ],
    { cwd: workdir, env },
  );
  const shippedExecutable = selectExecutable(shipped);
  const report = run(
    "cargo",
    [
      "rustc",
      "-p",
      PACKAGE,
      "--lib",
      "--profile",
      "bench",
      "--locked",
      "--features",
      lane,
      "--message-format=json",
      "--",
      "-Csave-temps",
      "--print=link-args",
    ],
    { cwd: workdir, env },
  );
  const objects = readdirSync(deps)
    .filter((entry) => /^candle_gen_flux2-.*\.(o|obj)$/.test(entry))
    .map((entry) => path.join(deps, entry));
  if (objects.length === 0) throw new Error("-Csave-temps produced no objects for the audited crate");
  return {
    executable: selectExecutable(report),
    shippedExecutable,
    objects,
    deps,
    linkInputs: linkInputs(report),
  };
}

/**
 * Whole-image digest of a NON-TEST artifact built from the same tree — candidate 4's unit.
 *
 * `examples/` targets are the only non-test binaries `candle-gen-flux2` has. They compile the crate
 * without `#[cfg(test)]`, which is the class the accepted false positive comes from, so hashing one
 * across the same mutations says directly what narrowing the audited target would buy — and what it
 * gives up, since an example is not the binary that produced the measurements.
 */
export function buildExample({ workdir, cargoTargetDir, lane, example }) {
  const report = run(
    "cargo",
    [
      "build",
      "-p",
      PACKAGE,
      "--example",
      example,
      "--profile",
      "bench",
      "--locked",
      "--features",
      lane,
      "--message-format=json",
    ],
    { cwd: workdir, env: buildEnv({ workdir, cargoTargetDir }) },
  );
  const artifact = report
    .split("\n")
    .filter(Boolean)
    .flatMap((line) => {
      try {
        return [JSON.parse(line)];
      } catch {
        return [];
      }
    })
    .find(
      (message) =>
        message?.reason === "compiler-artifact" &&
        message.executable &&
        message.target?.kind?.includes("example") &&
        message.target?.name === example,
    );
  if (!artifact) throw new Error(`no executable reported for example ${example}`);
  return {
    executable: artifact.executable,
    digest: `sha256:${createHash("sha256").update(readFileSync(artifact.executable)).digest("hex")}`,
  };
}

export function measure({ workdir, revision, cargoTargetDir, lane, patch, label, example, explain }) {
  const { executable, shippedExecutable, objects, linkInputs: inputs } = buildWithObjects({
    workdir,
    revision,
    cargoTargetDir,
    lane,
    patch,
  });
  const result = probe({ inputs: [...new Set([...inputs, ...objects])], explain });
  const exampleArtifact = example ? buildExample({ workdir, cargoTargetDir, lane, example }) : null;
  return {
    exampleBinary: exampleArtifact?.digest ?? null,
    label: label ?? revision,
    revision,
    patch: patch ?? null,
    executable,
    // `shippedUnit` is what `inference-artifact-audit.mjs` would have printed for this tree;
    // `probeUnit` is the same binary rebuilt with the object-emitting flags, kept only so the
    // reachability rows can be tied to a build that actually happened.
    shippedUnit: `sha256:${createHash("sha256").update(readFileSync(shippedExecutable)).digest("hex")}`,
    probeUnit: `sha256:${createHash("sha256").update(readFileSync(executable)).digest("hex")}`,
    auditedObjects: objects.length,
    ...result,
  };
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const args = process.argv.slice(2);
  const value = (flag, fallback) => {
    const at = args.indexOf(flag);
    return at === -1 ? fallback : args[at + 1];
  };
  const workdir = value("--workdir", "D:/repos/inference-audit/sc-17524");
  const cargoTargetDir = value("--target-dir", `${workdir}-target`);
  const out = value("--out");
  // Each entry is `label=revision[+patchfile]`, so one invocation can walk a revision pair and the
  // mutations layered on it without the driver ever holding a mutated tree between runs.
  const results = value("--cases", "")
    .split(",")
    .filter(Boolean)
    .map((entry) => {
      const [label, spec] = entry.includes("=") ? entry.split("=") : [entry, entry];
      const [revision, patch] = spec.split("+");
      return measure({
        workdir,
        revision,
        cargoTargetDir,
        lane: value("--lane", "cuda"),
        patch,
        label,
        example: value("--example"),
        explain: value("--explain") ? new RegExp(value("--explain")) : null,
      });
    });
  const text = JSON.stringify({ workdir, results }, null, 2);
  if (out) {
    mkdirSync(path.dirname(out), { recursive: true });
    writeFileSync(out, `${text}\n`);
  }
  process.stdout.write(`${text}\n`);
}
