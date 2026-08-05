#!/usr/bin/env node
// Bump SceneWorks' inference pin to a commit on inference `main`, so day-to-day development tracks
// inference by SHA instead of waiting on a cut `runtime-*` release. Inference is co-developed and
// SceneWorks is its only consumer, so a formal release does not belong on the critical path between
// the two repos -- releases stay for durable/shareable snapshots (inference's cut_release.py).
//
// The pins live in crates/sceneworks-worker/Cargo.toml and
// crates/sceneworks-memory-adapter/Cargo.toml. The root Cargo.toml additionally `[patch]`es
// candle-kernels to the multi-arch vendored copy inside the same inference revision (sc-7544 /
// sc-13510) — that rev must move in lockstep or the patched kernels skew against candle-core. This
// rewrites every `tag = "..."` / `rev = "..."` pin in those manifests and regenerates the lockfile.
//
// The direct `mlx-rs` pin (michaeltrefry/mlx-rs, a DIFFERENT url) is intentionally left alone -- but
// it must resolve to the same fork the pinned inference uses or Cargo builds two mlx-rs and the
// Array types diverge. After the bump this verifies no gen-core OR mlx-rs skew (reusing the repo's
// own check-gen-core-skew.sh) and fails loudly if the direct mlx-rs pin needs realigning.
//
//   node scripts/bump-inference.mjs                 # bump to latest inference main, update lock, verify
//   node scripts/bump-inference.mjs --dry-run       # show the target SHA, write nothing
//   node scripts/bump-inference.mjs --sha <sha40>   # pin a specific inference revision
//   node scripts/bump-inference.mjs --self-test     # exercise the pin rewrite + the facts checks
//
// `--self-test` runs in `npm run check`. It used to be a manual npm script only, which made it a
// place to add an assertion and never learn whether it fired; sc-17593 wired it in when it grew the
// engine-capability-facts coverage checks, whose whole subject is a guard that was never exercised.

import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

// The stage-2 derivation owns both halves of the facts-file contract, so the bump's fail-closed
// check reuses them rather than re-implementing a second, drifting copy (the same cross-import
// shape as scripts/check-scaffold.mjs).
import {
  assertBackendCoverage,
  parseSceneworksAudioBackends,
  parseSceneworksBackends,
} from "../apps/web/src/data/previewSupportDerivation.js";

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const MANIFEST = join(repoRoot, "crates/sceneworks-worker/Cargo.toml");
const MEMORY_MANIFEST = join(repoRoot, "crates/sceneworks-memory-adapter/Cargo.toml");
const LOCKFILE = join(repoRoot, "Cargo.lock");
// Root workspace manifest: holds the candle-kernels [patch] pin (same repo, same rev).
const ROOT_MANIFEST = join(repoRoot, "Cargo.toml");
const INFERENCE_GIT = "https://github.com/SceneWorks/inference";
// Every inference crate any workspace manifest depends on, so `cargo update -p` refreshes ALL of
// their lock entries. `mlx-gen` / `candle-gen` come in through
// `crates/sceneworks-memory-adapter` -- omitting them left that crate's four deps stranded on
// the previous revision in `Cargo.lock` even after its manifest was rewritten, which is the same
// two-revisions-in-one-lockfile skew `inferenceManifests()` exists to prevent one step earlier.
const INFERENCE_CRATES = [
  "sceneworks-gen-core",
  "runtime-macos",
  "runtime-cuda",
  "mlx-gen",
  "candle-gen",
];
// Resolved through the root [patch], not a direct dependency — still pinned to the inference
// repo, so its lock entry must be refreshed on every bump.
const PATCHED_CRATES = ["candle-kernels"];
// The worker stamps every catalog semantic analysis with the inference revision it was produced
// under, and `semantic_provenance_matches_linked_inference_revision` asserts that constant equals
// the Cargo pin. So it is PART of the pin, not a separate knob: a bump that leaves it behind is a
// red test (sc-14958 had to fix exactly that up after the previous bump). Moving it deliberately
// marks prior analyses stale so they re-run under the new runtime.
// Every workspace manifest that pins the inference repo is DISCOVERED, not listed. This script used
// to name the worker + root manifests explicitly, and silently missed
// `crates/sceneworks-memory-adapter/Cargo.toml` -- which carries four of its own inference deps
// (mlx-gen / runtime-macos / candle-gen / runtime-cuda) and was added long after this script. The
// result was a lockfile with TWO `sceneworks-gen-core` entries at different revisions, caught only
// downstream by `candle_kernels_patch_guard`. A hardcoded list rots every time a crate takes an
// inference dependency; discovery does not.
function inferenceManifests() {
  const found = [join(repoRoot, "Cargo.toml")].filter((path) =>
    readFileSync(path, "utf8").includes(INFERENCE_GIT),
  );
  const cratesDir = join(repoRoot, "crates");
  for (const entry of readdirSync(cratesDir, { withFileTypes: true })) {
    if (!entry.isDirectory()) continue;
    const path = join(cratesDir, entry.name, "Cargo.toml");
    let text;
    try {
      text = readFileSync(path, "utf8");
    } catch {
      continue;
    }
    if (text.includes(INFERENCE_GIT)) found.push(path);
  }
  if (!found.length) throw new Error(`no manifest pins ${INFERENCE_GIT}`);
  return found;
}

const SEMANTIC_PROVENANCE = join(repoRoot, "crates/sceneworks-worker/src/catalog_semantic_jobs.rs");
const SEMANTIC_PROVENANCE_RE = /(const INFERENCE_RUNTIME_REVISION: &str = ")[0-9a-f]{40}(";)/;
const MEMORY_PROVENANCE = join(repoRoot, "crates/sceneworks-memory-adapter/src/lib.rs");
const MEMORY_PROVENANCE_RE = /(pub const INFERENCE_PIN: &str = ")[0-9a-f]{40}(";)/;
// The FLUX.2 compatibility audit's frozen end-of-window (sc-17497 / sc-17524 / sc-17606). Read, never
// rewritten: unlike the two provenance stamps above, this constant is a claim about what a BUILD
// proved, so substituting the string would fake the proof — the same reason
// config/inference-third-party-source.json is left manual.
// Both are relative-plus-joined rather than one absolute path because `--self-test` drives the same
// code over a fixture tree; nothing else in this script needs a second root.
const FLUX2_AUDIT_FROZEN_RELATIVE = "crates/sceneworks-worker/src/inference_compatibility_audit.rs";
const FLUX2_AUDIT_FROZEN = join(repoRoot, FLUX2_AUDIT_FROZEN_RELATIVE);
const FLUX2_RECORD_DIR_RELATIVE = "docs/calibration/sc-15833";
// Mirrors `FLUX2_AUDIT_SCHEMA_VERSION` in the Rust module above. Only a record of this schema is
// evidence here, for exactly the reason both validators refuse the older ones.
const FLUX2_ACCEPTED_SCHEMA_VERSION = 5;
const FLUX2_COMPATIBLE_RE =
  /const FLUX2_COMPATIBLE_INFERENCE_REVISION: &str =\s*"([0-9a-f]{40})"\s*;/;
const SHA_RE = /^[0-9a-f]{40}$/;

// --- pure: rewrite the inference pins to rev=<sha> (self-tested; no fs/network) ---------------

// The revision the tree is pinned at BEFORE the rewrite, read off the same lines `repin` rewrites.
// Only `verifyFlux2AuditWindow` needs it, and only to tell "this bump moved the pin out of the
// audited window" from "it was already out" — so a manifest pinned by `tag`, or one whose lines
// disagree, is reported as `null` (unknown) rather than guessed at: the unknown case degrades to the
// warning, never to a spurious refusal.
function pinnedRevision(manifestText) {
  const revisions = new Set(
    manifestText
      .split("\n")
      .filter((line) => line.includes(INFERENCE_GIT))
      .map((line) => /\brev\s*=\s*"([0-9a-f]{40})"/.exec(line)?.[1])
      .filter(Boolean),
  );
  return revisions.size === 1 ? [...revisions][0] : null;
}

function repin(manifestText, sha, manifestPath = MANIFEST) {
  let inferenceLines = 0;
  let rewrote = 0;
  const out = manifestText.split("\n").map((line) => {
    // Only lines that pin the inference git repo. The direct mlx-rs pin uses another url, so it is
    // never matched here -- alignment of that pin is verified after the bump, not rewritten.
    if (!line.includes(INFERENCE_GIT)) return line;
    inferenceLines += 1;
    return line.replace(/\b(?:tag|rev)\s*=\s*"[^"]*"/, () => {
      rewrote += 1;
      return `rev = "${sha}"`;
    });
  });
  if (inferenceLines === 0) {
    throw new Error(`no inference pins found (looked for ${INFERENCE_GIT} in ${manifestPath})`);
  }
  if (rewrote !== inferenceLines) {
    throw new Error(`expected to rewrite ${inferenceLines} inference pin(s), rewrote ${rewrote}`);
  }
  return out.join("\n");
}

function repinSemanticProvenance(source, sha) {
  if (!SEMANTIC_PROVENANCE_RE.test(source)) {
    throw new Error(
      `no INFERENCE_RUNTIME_REVISION constant found in ${SEMANTIC_PROVENANCE} -- the semantic ` +
        "provenance stamp must move with the pin or its lockstep test goes red",
    );
  }
  return source.replace(SEMANTIC_PROVENANCE_RE, `$1${sha}$2`);
}

function repinMemoryProvenance(source, sha) {
  if (!MEMORY_PROVENANCE_RE.test(source)) {
    throw new Error(
      `no INFERENCE_PIN constant found in ${MEMORY_PROVENANCE} -- the calibration adapter ` +
        "provenance stamp must move with the pin",
    );
  }
  return source.replace(MEMORY_PROVENANCE_RE, `$1${sha}$2`);
}

// --- git / cargo orchestration ----------------------------------------------------------------

function latestInferenceSha() {
  const out = execFileSync("git", ["ls-remote", INFERENCE_GIT, "HEAD"], { encoding: "utf8" });
  const sha = (out.split(/\s+/)[0] || "").trim();
  if (!SHA_RE.test(sha)) throw new Error(`git ls-remote returned no SHA: ${out.trim()}`);
  return sha;
}

function packageSpecsForUpdate(crate, sha, lockText = readFileSync(LOCKFILE, "utf8")) {
  const packages = lockText
    .split("[[package]]")
    .slice(1)
    .map((block) => ({
      name: block.match(/\nname = "([^"]+)"/)?.[1],
      version: block.match(/\nversion = "([^"]+)"/)?.[1],
      source: block.match(/\nsource = "([^"]+)"/)?.[1],
    }))
    .filter(({ name, source }) => name === crate && source?.includes(INFERENCE_GIT));
  if (packages.length <= 1) return [crate];

  const stale = packages.filter(({ source }) => !source.includes(`?rev=${sha}#`));
  if (stale.length === 0) return [crate];
  return stale.map(({ name, version, source }) => {
    const sourceWithoutCommit = source.slice(0, source.lastIndexOf("#"));
    return `${sourceWithoutCommit}#${name}@${version}`;
  });
}

function lockHasStaleInferenceRevision(sha, lockText = readFileSync(LOCKFILE, "utf8")) {
  return lockText
    .split("\n")
    .some(
      (line) =>
        line.includes(`source = "git+${INFERENCE_GIT}?rev=`) &&
        !line.includes(`?rev=${sha}#`),
    );
}

function cargoUpdate(sha) {
  const packages = [...new Set([...INFERENCE_CRATES, ...PATCHED_CRATES])];
  const spec = packages.flatMap((crate) =>
    packageSpecsForUpdate(crate, sha).flatMap((packageSpec) => ["-p", packageSpec]),
  );
  console.log(`$ cargo update ${spec.join(" ")}`);
  execFileSync("cargo", ["update", ...spec], { cwd: repoRoot, stdio: "inherit" });
}

function distinctResolutions(crate) {
  // One `cargo tree` over both platform bundles (--target all), so macOS + CUDA resolutions are
  // visible even off-macOS -- the same data source check-gen-core-skew.sh uses.
  const tree = execFileSync(
    "cargo",
    ["tree", "-p", "sceneworks-worker", "--features", "backend-candle", "--target", "all", "--prefix", "none"],
    { cwd: repoRoot, encoding: "utf8" },
  );
  return new Set(
    tree
      .split("\n")
      .map((l) => l.replace(/\s*\(\*\)\s*$/, "").trim())
      .filter((l) => l.includes(crate)),
  );
}

// `config/inference-third-party-source.json` is ALSO keyed to the pin: check-license-coverage.mjs
// fails closed when its `inferenceRevision` / `provenanceScan.revision` / `crateCoverage.revision` do
// not equal the pinned rev. Unlike the memory matrix this cannot be regenerated blindly — a bump that
// brings a NEW crate or a new embedded third-party source needs a human classification decision, and
// the rescan needs a local inference clone (`--repo`) that this script does not otherwise require. So
// run the guard and let its own remediation text (which names the exact scanner invocations) speak; the
// point is that a stale audit surfaces HERE, at bump time, instead of in parity CI ten minutes later.
// Fail-closed on purpose: the manifests and lockfile are already written by now, so the bump is
// genuinely incomplete until the audit is refreshed and this passes.
function verifyLicenseAudit() {
  console.log("$ node scripts/check-license-coverage.mjs");
  try {
    execFileSync("node", ["scripts/check-license-coverage.mjs"], {
      cwd: repoRoot,
      stdio: "inherit",
    });
  } catch {
    throw new Error(
      "the inference source/license audit is stale for the new pin (see the check output above). " +
        "Re-run the scanner against a local inference clone, refresh " +
        "config/inference-third-party-source.json (inferenceRevision, provenanceScan, crateCoverage, " +
        "auditDigest), then re-run this bump to verify. The pin itself is already written.",
    );
  }
}

// `docs/generated/memory-matrix.{json,md}` is DERIVED from the inference pin: the generator stamps
// `inferenceRevision` on the document (`generatedFrom`; sc-16268 removed the per-cell copy that
// used to duplicate it into every row), and each cell's derived `calibrationFingerprint` includes
// the inference pin alongside the provider ABI and semantic cell identity. So a pin bump makes the checked-in artifact stale by construction, and
// `tests/test_memory_matrix.py`
// (parity CI) fails with "generated memory matrix is stale". Regenerating here keeps everything derived
// from the pin moving in ONE commit, the same reason the lockfile regen lives in this script rather than
// in the caller's hands.
function regenerateMemoryMatrix() {
  console.log("$ node scripts/generate-memory-matrix.mjs");
  execFileSync("node", ["scripts/generate-memory-matrix.mjs"], {
    cwd: repoRoot,
    stdio: "inherit",
  });
}

// `docs/generated/calibration-cost-model.{json,md}` is derived from the memory matrix regenerated
// immediately above — it stamps `generatedFrom.inferenceRevision` and re-derives its counts from the
// matrix cells — so the matrix regen makes it stale by construction. `npm run check` (via
// `check:rust-derived-docs` -> `check:calibration-cost-model --check`) then fails with "generated
// calibration cost model is stale", one step further down the SAME generated-artifact cascade this
// script exists to keep in one commit. It was missing here: sc-16962's bump regenerated the matrix,
// reported success, and left the cost model behind for `npm run check` to catch minutes later.
function regenerateCalibrationCostModel() {
  console.log("$ node scripts/calibration-cost-model.mjs");
  execFileSync("node", ["scripts/calibration-cost-model.mjs"], {
    cwd: repoRoot,
    stdio: "inherit",
  });
}

// `config/engine-capabilities/capabilities.<backend>.json` is keyed to the pin exactly like the
// licence audit above: each file stamps `generatedFrom.inferenceRevision`, and it is a dump of
// `Capabilities.supports_preview` read off the LINKED provider registry at that revision (sc-16965,
// epic 16948). A bump can move any descriptor's flag — every remaining family story in that epic
// flips more of them — so the checked-in dumps go stale by construction.
//
// Like the licence audit, and unlike the memory matrix, this CANNOT be regenerated blindly from
// here: the dumper needs a lane that actually links engines (macOS for mlx,
// `--features backend-candle` off-Mac for candle), which a bump does not otherwise require and
// which no single host can supply for both backends. So verify, and let the remediation text name
// the exact invocation. Fail-closed on purpose: the pins are already written by now, so the bump is
// genuinely incomplete until the dumps are refreshed and the stage-2 artifacts regenerated.
//
// Downstream cascade, same shape as memory-matrix -> calibration-cost-model: re-dumping a facts file
// makes `config/manifests/builtin.preview-support.jsonc` + `apps/web/src/data/previewSupport.json`
// stale, and `apps/web/src/data/previewSupportCatalog.test.js` (web vitest, every PR) fails until
// `npm run gen:preview-support` is re-run.
// `root` is a parameter purely so `--self-test` can drive this against fixture trees. It is the one
// check in this script with real branching over the FILESYSTEM rather than over text, and its
// audio half (sc-17593) guards a hole that stayed invisible for four pins, so "it is wired" is not
// the same as "it fires".
function verifyEngineCapabilityFacts(sha, root = repoRoot) {
  const dir = join(root, "config/engine-capabilities");
  // The audio registry's dumps live one level down (sc-17593), so `readdirSync` — not recursive —
  // keeps the two sets apart on its own. They are validated for staleness together and for coverage
  // separately, because their declared backend sets are different consts.
  const audioDir = join(dir, "audio");
  const factsFileNames = (from) => {
    try {
      return readdirSync(from).filter((name) => /^capabilities\.[a-z0-9_-]+\.json$/.test(name));
    } catch {
      return [];
    }
  };
  const names = factsFileNames(dir);
  const audioNames = factsFileNames(audioDir);
  if (names.length === 0) {
    throw new Error(
      `no engine-capability facts under ${dir}. Dump them on a lane that links engines: ` +
        "`cargo run -p sceneworks-worker --bin dump-engine-capabilities --features backend-candle` " +
        "(off-Mac) or the same command with no features (macOS), then re-run " +
        "`npm run gen:preview-support` from apps/web.",
    );
  }
  // Coverage BEFORE staleness (sc-17119). The loop below can only judge files that exist, so a
  // backend that was never dumped is not "stale" — it is absent, and absence passed this check
  // forever. `capabilities.mlx.json` was absent for four consecutive pins beginning with the one
  // sc-16965 itself shipped on, and nothing in this function could have objected -- it validates
  // only the files that are there.
  const dumpedBackendsIn = (from, fileNames, constName, lane) =>
    fileNames.map((name) => {
      const facts = JSON.parse(readFileSync(join(from, name), "utf8"));
      const backend = facts?.backend;
      if (typeof backend !== "string" || backend.length === 0) {
        // Named here rather than left to `assertBackendCoverage`, which sees only the values: a
        // nameless backend reads there as an unidentifiable entry, and the operator needs the file.
        throw new Error(
          `${name} carries no \`backend\` field, so it cannot be matched against ${constName}. ` +
            "Re-dump it on the lane that owns it rather than hand-editing it.",
        );
      }
      // The `registry` discriminator (sc-17593). Checking it here and not only in the stage-2
      // derivation matters because a media and an audio dump can carry the SAME `backend` —
      // `candle` on every platform for audio — so `backend` alone cannot tell a file that landed in
      // the wrong directory from a correct one, and this check would happily count it as coverage.
      const registry = facts?.registry ?? "media";
      if (registry !== lane) {
        throw new Error(
          `${name} is a ${JSON.stringify(registry)} dump but sits in the ${JSON.stringify(lane)} ` +
            "facts directory. Media dumps belong in config/engine-capabilities/, audio dumps in " +
            "config/engine-capabilities/audio/ — counting this one would report coverage for a " +
            "registry that was never dumped.",
        );
      }
      return backend;
    });
  const factsDeclarationSource = readFileSync(
    join(root, "crates/sceneworks-worker/src/engine_capability_facts.rs"),
    "utf8",
  );
  assertBackendCoverage(
    parseSceneworksBackends(factsDeclarationSource),
    dumpedBackendsIn(dir, names, "SCENEWORKS_BACKENDS", "media"),
  );
  // sc-17593. The audio registry is dumped separately and its coverage must be asserted separately:
  // `candle` in SCENEWORKS_BACKENDS is satisfied by the media dump alone, so an undumped audio
  // registry cleared the check above every time — which is how it stayed invisible.
  assertBackendCoverage(
    parseSceneworksAudioBackends(factsDeclarationSource),
    dumpedBackendsIn(audioDir, audioNames, "SCENEWORKS_AUDIO_BACKENDS", "audio"),
    "audio",
  );

  const stale = [];
  for (const [from, fileNames, label] of [
    [dir, names, ""],
    [audioDir, audioNames, "audio/"],
  ]) {
    for (const name of fileNames) {
      const facts = JSON.parse(readFileSync(join(from, name), "utf8"));
      const revision = facts?.generatedFrom?.inferenceRevision;
      if (revision !== sha) stale.push(`${label}${name} (dumped at ${revision ?? "unknown"})`);
    }
  }
  if (stale.length) {
    throw new Error(
      `engine-capability facts are stale for ${sha}:\n  ${stale.join("\n  ")}\n` +
        "Each file must be re-dumped on the lane that owns it — one file per backend, so no lane " +
        "overwrites another's:\n" +
        "  macOS  : cargo run -p sceneworks-worker --bin dump-engine-capabilities\n" +
        "  off-Mac: cargo run -p sceneworks-worker --bin dump-engine-capabilities " +
        "--no-default-features --features backend-candle\n" +
        "Either command also rewrites audio/ — the audio registry is candle-native on both lanes.\n" +
        "then regenerate the derived catalog: (cd apps/web && npm run gen:preview-support). " +
        "The pin itself is already written.",
    );
  }
  // Silent when driven from `--self-test`, which runs in `npm run check`: the fixture tree is a
  // temp dir at a fake SHA, and an "OK: … dumped at aaaa…" line there reads exactly like a real
  // verification of the repo's own facts files.
  if (root === repoRoot) {
    console.log(
      `OK: ${names.length} engine-capability facts file(s) + ${audioNames.length} audio file(s) ` +
        `dumped at ${sha}`,
    );
  }
}

// The FLUX.2 compatibility audit is pin-keyed like the two checks above, and until sc-17760 this
// script did not know it existed.
//
// `FLUX2_COMPATIBILITY_AUDIT` (scripts/generate-memory-matrix.mjs) authorizes the sc-15833 FLUX.2
// measurements to be read as `Runtime verified` at a revision NEWER than the one they were captured
// at, and `compatibilityAuthorizes` only matches while the live pin equals the audited
// `compatibleInferenceRevision`. Move the pin past it and the five `flux2_dev` q4 cells fall back to
// `Implemented/unverified` — quietly, because a cell's current/historical state is DERIVED from the
// live pin at generate time. Every test stays green and simply records the demotion, which is why
// this needs a guard rather than a reader.
//
// That is the same staleness class the licence audit and the capability-facts dumps have. It is NOT
// fail-closed on the same terms, and the difference is the whole design of this check.
//
// The other two guards refuse because their artifact can always be refreshed: re-scan, or re-dump on
// a lane that links engines. This one cannot promise that. The audit's exit 1 — ARTIFACTS DIFFER —
// is a legitimate, terminal verdict meaning the compiled code the measurements ran genuinely changed
// and the cells are CORRECTLY demoted until the calibration is re-captured (sc-15922, a campaign
// this script cannot start). sc-17760 ran the `5ffd7612 -> fbb00d6b` window and got exactly that. A
// guard that refused unconditionally would therefore block every future pin bump, for an unbounded
// period, over a demotion no future bump caused — and a guard that stands between the operator and a
// bump they cannot satisfy is a guard that gets deleted.
//
// So it keys on the TRANSITION, which is the event worth refusing:
//
//   in-window  -> out-of-window : THROW. This bump is what demotes the cells, and re-running the
//                                 audit first is cheap (~15 min warm, no GPU, nothing executes).
//                                 This is #2120's case, the regression this exists for.
//   out         -> out          : WARN. The cells were already demoted before this bump; refusing
//                                 fixes nothing and stops unrelated work.
//   any         -> in-window    : OK.
//
// It runs BEFORE `regenerateMemoryMatrix`, so on the throwing path the bump refuses instead of first
// writing a generated matrix with the demotion baked into it.
//
// `root` is a parameter for the same reason `verifyEngineCapabilityFacts` has one: `--self-test`
// drives it over a fixture tree, so the check that fires here is the one that is exercised.
// Has this exact pin already been probed? The audit writes its record on EVERY path, including the
// two that exit 1, and superseded/negative records stay on disk — so the answer to "is it worth
// spending 15 minutes on this window" is usually already checked in.
//
// Returns the advice line for a record that names `sha`, or `null` when there is none and the
// operator really should run it. A record whose digests MATCH is a different failure: the window
// does extend and someone stopped halfway through the checklist, which is worth saying plainly
// rather than sending them back to the tool that would just print the same thing.
function priorProbe(sha, root) {
  const dir = join(root, FLUX2_RECORD_DIR_RELATIVE);
  let names;
  try {
    names = readdirSync(dir).filter((name) => /^inference-compatibility-.*\.json$/.test(name));
  } catch {
    return null;
  }
  for (const name of names) {
    let record;
    try {
      record = JSON.parse(readFileSync(join(dir, name), "utf8"));
    } catch {
      continue; // an unreadable record is not evidence either way; the generic remedy still applies.
    }
    if (record?.compatibleInferenceRevision !== sha) continue;
    const artifact = record?.auditedArtifact ?? {};
    const witness = record?.featureWitness ?? {};
    const artifactMoved =
      !!artifact.capturedDigest && artifact.capturedDigest !== artifact.compatibleDigest;
    const witnessMoved =
      !!witness.capturedDigest && witness.capturedDigest !== witness.compatibleDigest;
    // A record this repo's validators would REFUSE cannot be read as evidence in either direction,
    // and reading one as the "extends" case is the dangerous half: `inference-compatibility-277f.json`
    // is a v1 record sitting in this very directory with no digests at all, which without this
    // check computes `moved = false` and advises moving the frozen constants. Superseded records
    // are kept as history, not as fallbacks — see the schema-versions note in
    // docs/inference-artifact-audit-sc-17497.md — so anything but the accepted schema, or a record
    // missing either digest pair, falls back to the generic remedy.
    const complete =
      record?.schemaVersion === FLUX2_ACCEPTED_SCHEMA_VERSION &&
      !!artifact.capturedDigest &&
      !!artifact.compatibleDigest &&
      !!witness.capturedDigest &&
      !!witness.compatibleDigest;
    if (!complete) continue;
    if (artifactMoved || witnessMoved) {
      return (
        `This pin has ALREADY been probed — ${FLUX2_RECORD_DIR_RELATIVE}/${name} records ` +
        `${artifactMoved ? "ARTIFACTS DIFFER" : "SHIPPED FEATURE SETS DIFFER"} across ` +
        `${record.capturedInferenceRevision} -> ${sha}. Do NOT re-run it and do NOT move the frozen ` +
        "constants: the window does not extend, and the demotion stands until sc-15922 re-captures " +
        "the calibration."
      );
    }
    return (
      `${FLUX2_RECORD_DIR_RELATIVE}/${name} says this window DOES extend, but the frozen constants ` +
      "were never moved to match it. Finish the `After running it` checklist in " +
      "docs/inference-artifact-audit-sc-17497.md rather than re-running the audit."
    );
  }
  return null;
}

function verifyFlux2AuditWindow(sha, previousSha, root = repoRoot) {
  const path = root === repoRoot ? FLUX2_AUDIT_FROZEN : join(root, FLUX2_AUDIT_FROZEN_RELATIVE);
  const source = readFileSync(path, "utf8");
  const match = FLUX2_COMPATIBLE_RE.exec(source);
  if (!match) {
    throw new Error(
      `FLUX2_COMPATIBLE_INFERENCE_REVISION not found in ${path}. It is the end of the audited ` +
        "FLUX.2 window and this bump cannot tell whether the new pin is inside it. If the constant " +
        "was renamed or moved, update FLUX2_COMPATIBLE_RE in scripts/bump-inference.mjs.",
    );
  }
  const audited = match[1];
  if (audited === sha) {
    // Silent under `--self-test` for the reason `verifyEngineCapabilityFacts` is: an "OK: … covers
    // aaaa…" line over a fixture tree reads exactly like a verification of the repo's own constant.
    if (root === repoRoot) console.log(`OK: the FLUX.2 compatibility audit already covers ${sha}`);
    return;
  }
  // A record for THIS pin already on disk changes the advice completely, and it is the difference
  // between a useful warning and one that costs 15 minutes to act on. sc-17760 probed `fbb00d6b`
  // and got ARTIFACTS DIFFER: re-running that window is pure waste, and the operator needs to be
  // told the verdict is terminal rather than pointed at the same command again. Records are kept on
  // disk precisely so this is answerable without a build.
  const priorVerdict = priorProbe(sha, root);
  const remedy =
    `The FLUX.2 compatibility audit ends at ${audited}, so it does not cover ${sha}.\n` +
    "Nothing goes red either way: it demotes the five flux2_dev q4 cells from `Runtime verified` " +
    "in the generated memory matrix, because a cell's current/historical state is DERIVED from the " +
    "live pin at generate time.\n" +
    (priorVerdict ??
      "Re-run the audit on the Windows CUDA box (~15 min warm; both builds are " +
        "`cargo test --no-run`, so no GPU and no weights are needed):\n" +
        "  node scripts/inference-artifact-audit.mjs --repo <inference clone> \\\n" +
        "    --captured 5ffd7612e7de4e76b6db00a7148ed3d9c15b4c0d \\\n" +
        `    --compatible ${sha} --lane cuda --workdir <persistent workdir> \\\n` +
        `    --out ${FLUX2_RECORD_DIR_RELATIVE}/inference-compatibility-<short>.json\n` +
        "Exit 0 -> follow the `After running it` checklist in " +
        "docs/inference-artifact-audit-sc-17497.md. Exit 1 (ARTIFACTS DIFFER / SHIPPED FEATURE " +
        "SETS DIFFER) is a re-capture verdict rather than a repin: the measurements no longer " +
        "describe the code at this pin, the constants must NOT be moved, and the demotion stands " +
        "until sc-15922 re-captures.");
  if (audited === previousSha) {
    throw new Error(
      `${remedy}\nThis bump is what moves the pin out of the audited window. Nothing has been ` +
        "written: unlike the other verifiers here, this one runs BEFORE the rewrite, so re-running " +
        "the bump unchanged refuses identically instead of inheriting its own half-applied state.",
    );
  }
  // Already outside before this bump — say so at full volume, but do not stand in the way. Printed
  // in the self-test too: a warning nobody sees is the failure mode this whole check exists for.
  console.warn(
    `WARNING: the pin was ALREADY outside the audited FLUX.2 window before this bump ` +
      `(was ${previousSha ?? "unknown"}, audited window ends at ${audited}).\n${remedy}\n` +
      "This bump did not cause the demotion, so it is not refused — but it does not clear it either.",
  );
}

function verifyNoSkew() {
  // gen-core: reuse the repo's own CI-wired guard verbatim.
  console.log("$ bash scripts/check-gen-core-skew.sh sceneworks-worker --features backend-candle");
  execFileSync("bash", ["scripts/check-gen-core-skew.sh", "sceneworks-worker", "--features", "backend-candle"], {
    cwd: repoRoot,
    stdio: "inherit",
  });
  // mlx-rs: SceneWorks pins pmetal-mlx-rs directly; it has no dedicated guard, so confirm the bumped
  // inference did not pull a different fork revision.
  const mlx = distinctResolutions("pmetal-mlx-rs");
  if (mlx.size > 1) {
    throw new Error(
      `mlx-rs skew: ${mlx.size} pmetal-mlx-rs resolutions after the bump:\n  ${[...mlx].join("\n  ")}\n` +
        "Align the direct mlx-rs pin in crates/sceneworks-worker/Cargo.toml with the fork this " +
        "inference revision uses.",
    );
  }
  console.log(`OK: one pmetal-mlx-rs resolution (${[...mlx][0] ?? "not found"})`);
}

// --- self-test --------------------------------------------------------------------------------

function selfTest() {
  let rc = 0;
  const SHA = "a".repeat(40);
  const check = (name, ok) => {
    console.log(`  ${ok ? "ok" : "FAIL"}: ${name}`);
    if (!ok) rc = 1;
  };

  check(
    "tag pin becomes rev",
    repin(`gc = { git = "${INFERENCE_GIT}", tag = "runtime-2026.07.7" }`, SHA) ===
      `gc = { git = "${INFERENCE_GIT}", rev = "${SHA}" }`,
  );
  check(
    "rev pin bumps and keeps trailing options",
    repin(`rt = { git = "${INFERENCE_GIT}", rev = "bbbb", optional = true }`, SHA) ===
      `rt = { git = "${INFERENCE_GIT}", rev = "${SHA}", optional = true }`,
  );
  const mlx = `mlx-rs = { package = "pmetal-mlx-rs", git = "https://github.com/michaeltrefry/mlx-rs", rev = "38e1cc17" }`;
  check(
    "direct mlx-rs pin (other url) is left untouched",
    repin(`rt = { git = "${INFERENCE_GIT}", tag = "x" }\n${mlx}`, SHA).includes(mlx),
  );
  check(
    "every inference pin moves",
    (repin(
      `a = { git = "${INFERENCE_GIT}", tag = "x" }\nb = { git = "${INFERENCE_GIT}", rev = "y" }`,
      SHA,
    ).match(new RegExp(`rev = "${SHA}"`, "g")) || []).length === 2,
  );
  check(
    "root-manifest candle-kernels [patch] pin bumps",
    repin(`candle-kernels = { git = "${INFERENCE_GIT}", rev = "d68b8b45" }`, SHA) ===
      `candle-kernels = { git = "${INFERENCE_GIT}", rev = "${SHA}" }`,
  );
  let threw = false;
  try {
    repin(`foo = "bar"`, SHA);
  } catch {
    threw = true;
  }
  check("throws when no inference pin is present", threw);

  // The fixture deliberately carries a DECOY 40-hex constant: the real file declares
  // CLIP_MODEL_REVISION (an upstream HF model revision) two lines above the stamp, and nothing else
  // guards it -- it is only ever compared against itself. A loosened SEMANTIC_PROVENANCE_RE would
  // silently overwrite it on every bump, so the fixture must prove the rewrite is anchored to the
  // stamp's own name, not merely that the stamp moved.
  const CLIP_DECOY = `const CLIP_MODEL_REVISION: &str = "32bd64288804d66eefd0ccbe215aa642df71cc41";`;
  const stampBefore =
    `${CLIP_DECOY}\nconst CLIP_SPACE: &str = "clip-vit-l14";\nconst INFERENCE_RUNTIME_REVISION: &str = "${"b".repeat(40)}";\nconst DEFAULT_BATCH_SIZE: usize = 16;`;
  const stampAfter =
    `${CLIP_DECOY}\nconst CLIP_SPACE: &str = "clip-vit-l14";\nconst INFERENCE_RUNTIME_REVISION: &str = "${SHA}";\nconst DEFAULT_BATCH_SIZE: usize = 16;`;
  const stampBumped = repinSemanticProvenance(stampBefore, SHA);
  check("semantic provenance stamp bumps with the pin", stampBumped === stampAfter);
  check(
    "neighbouring 40-hex CLIP_MODEL_REVISION is left byte-identical",
    stampBumped.includes(CLIP_DECOY) && !stampBumped.includes(`CLIP_MODEL_REVISION: &str = "${SHA}"`),
  );
  check(
    "memory-strategy adapter provenance stamp bumps with the pin",
    repinMemoryProvenance(
      `pub const INFERENCE_PIN: &str = "${"b".repeat(40)}";`,
      SHA,
    ) === `pub const INFERENCE_PIN: &str = "${SHA}";`,
  );
  const OLD_SHA = "b".repeat(40);
  const OLD_COMMIT = "c".repeat(40);
  const CURRENT_COMMIT = "d".repeat(40);
  const duplicateLock = `
[[package]]
name = "mlx-gen"
version = "0.0.0"
source = "git+${INFERENCE_GIT}?rev=${OLD_SHA}#${OLD_COMMIT}"

[[package]]
name = "mlx-gen"
version = "0.0.0"
source = "git+${INFERENCE_GIT}?rev=${SHA}#${CURRENT_COMMIT}"
`;
  check(
    "duplicate lock revisions select only the stale source-qualified package",
    packageSpecsForUpdate("mlx-gen", SHA, duplicateLock).join("\n") ===
      `git+${INFERENCE_GIT}?rev=${OLD_SHA}#mlx-gen@0.0.0`,
  );
  check(
    "stale inference lock revisions are detected",
    lockHasStaleInferenceRevision(SHA, duplicateLock),
  );
  check(
    "a lock containing only the requested revision is clean",
    !lockHasStaleInferenceRevision(
      SHA,
      duplicateLock.replaceAll(OLD_SHA, SHA).replaceAll(OLD_COMMIT, CURRENT_COMMIT),
    ),
  );
  let stampThrew = false;
  try {
    repinSemanticProvenance(`const OTHER: &str = "x";`, SHA);
  } catch {
    stampThrew = true;
  }
  check("throws when the semantic provenance stamp is missing", stampThrew);

  // --- engine-capability facts, over a real fixture tree (sc-17593) --------------------------
  //
  // Everything above is text in, text out. This check is here because the facts verification is the
  // one part of the script that branches on FILES — including files that are not there, which is the
  // failure mode that hid an undumped registry for four consecutive pins. Driven over a temp tree so
  // it exercises the same code the bump runs, rather than a paraphrase of it.
  const factsDeclaration = [
    "/// doc comment naming candle and mlx in prose, which the parser must not anchor on.",
    'pub const SCENEWORKS_BACKENDS: &[&str] = &["candle", "mlx"];',
    'pub const SCENEWORKS_AUDIO_BACKENDS: &[&str] = &["candle"];',
    "",
  ].join("\n");
  const factsFile = (backend, revision, extra = {}) =>
    JSON.stringify({
      ...extra,
      backend,
      generatedFrom: { inferenceRevision: revision, dumper: "self-test" },
      engines: [{ id: "x", modality: "image", supportsPreview: false }],
    });
  const fixture = ({ audio = true, revision = SHA, swapped = false } = {}) => {
    const root = mkdtempSync(join(tmpdir(), "bump-inference-facts-"));
    mkdirSync(join(root, "crates/sceneworks-worker/src"), { recursive: true });
    writeFileSync(
      join(root, "crates/sceneworks-worker/src/engine_capability_facts.rs"),
      factsDeclaration,
    );
    const dir = join(root, "config/engine-capabilities");
    mkdirSync(dir, { recursive: true });
    for (const backend of ["candle", "mlx"]) {
      writeFileSync(
        join(dir, `capabilities.${backend}.json`),
        // `swapped` puts an AUDIO dump where a media one belongs — the realistic mistake, and the
        // one `backend` alone cannot detect, since both registries are `candle`.
        factsFile(backend, SHA, swapped && backend === "candle" ? { registry: "audio" } : {}),
      );
    }
    if (audio) {
      mkdirSync(join(dir, "audio"), { recursive: true });
      writeFileSync(
        join(dir, "audio/capabilities.candle.json"),
        factsFile("candle", revision, swapped ? {} : { registry: "audio" }),
      );
    }
    return root;
  };
  const verifyFacts = (options) => {
    const root = fixture(options);
    try {
      verifyEngineCapabilityFacts(SHA, root);
      return null;
    } catch (error) {
      return error?.message ?? String(error);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  };

  check("a complete facts tree verifies", verifyFacts() === null);
  // THE regression this exists for. Before sc-17593 this exact tree — every media backend dumped,
  // the audio registry dumped nowhere — passed, because `candle` in SCENEWORKS_BACKENDS is satisfied
  // by the MEDIA dump alone. The check must name the audio lane, not merely fail.
  const missingAudio = verifyFacts({ audio: false });
  check("a missing audio dump fails", missingAudio !== null);
  check(
    "the missing-audio failure names the audio lane and how to dump it",
    !!missingAudio &&
      /audio engine-capability facts are missing/.test(missingAudio) &&
      /dump-engine-capabilities/.test(missingAudio),
  );
  // Staleness must reach into the subdirectory too: an audio dump left at the previous pin describes
  // descriptors that may have moved, exactly as a stale media dump does.
  const staleAudio = verifyFacts({ revision: "b".repeat(40) });
  check(
    "a stale audio dump is reported, named by its subdirectory path",
    !!staleAudio && /audio\/capabilities\.candle\.json/.test(staleAudio),
  );
  // Both dumps carry `backend: "candle"`, so swapping the two directories is invisible to a check
  // that reads `backend` alone — it would count an audio file as media coverage and vice versa, and
  // report full coverage with one registry dumped twice and the other not at all.
  const swapped = verifyFacts({ swapped: true });
  check(
    "a dump sitting in the other registry's directory is refused",
    !!swapped && /sits in the/.test(swapped),
  );

  // --- the FLUX.2 audited window (sc-17760) --------------------------------------------------
  //
  // Driven over a fixture tree for the same reason as the facts checks: this guard exists because a
  // pin moved out of the audited window and NOTHING objected, so "it is wired" is not the same as
  // "it fires". Both sides are exercised — a matching window must pass, or the failing side proves
  // nothing but that the function throws.
  //
  // The fixture deliberately carries a DECOY 40-hex constant. `FLUX2_CAPTURED_INFERENCE_REVISION`
  // sits two lines above the one this reads and must never be mistaken for it: anchoring on the
  // wrong end of the window would compare the pin against the revision the measurements were TAKEN
  // at, which never moves, so the guard would fire on every bump forever.
  const auditSource = (compatible) =>
    [
      'pub(crate) const FLUX2_CAPTURED_INFERENCE_REVISION: &str =',
      `    "${"c".repeat(40)}";`,
      "pub(crate) const FLUX2_COMPATIBLE_INFERENCE_REVISION: &str =",
      `    "${compatible}";`,
      "pub(crate) const FLUX2_AUDIT_SCHEMA_VERSION: u64 = 5;",
      "",
    ].join("\n");
  // `warned` captures console.warn so the "already outside" branch can be asserted on its OUTPUT
  // rather than merely on "it did not throw" — which a no-op would also satisfy.
  const auditFixture = (source, previousSha, record) => {
    const root = mkdtempSync(join(tmpdir(), "bump-inference-flux2-"));
    mkdirSync(join(root, "crates/sceneworks-worker/src"), { recursive: true });
    writeFileSync(
      join(root, "crates/sceneworks-worker/src/inference_compatibility_audit.rs"),
      source,
    );
    if (record) {
      const dir = join(root, "docs/calibration/sc-15833");
      mkdirSync(dir, { recursive: true });
      writeFileSync(join(dir, "inference-compatibility-fixture.json"), JSON.stringify(record));
    }
    const realWarn = console.warn;
    const warnings = [];
    console.warn = (...parts) => warnings.push(parts.join(" "));
    try {
      verifyFlux2AuditWindow(SHA, previousSha, root);
      return { threw: null, warned: warnings.join("\n") };
    } catch (error) {
      return { threw: error?.message ?? String(error), warned: warnings.join("\n") };
    } finally {
      console.warn = realWarn;
      rmSync(root, { recursive: true, force: true });
    }
  };
  const IN_WINDOW = SHA;
  const OUT_OF_WINDOW = "b".repeat(40);
  const AUDITED_END = "e".repeat(40);

  const covered = auditFixture(auditSource(IN_WINDOW), OUT_OF_WINDOW);
  check("a pin inside the audited FLUX.2 window passes", covered.threw === null && !covered.warned);
  // THE regression this exists for: #2120 moved the pin to fbb00d6b with the record still ending at
  // a4f409ae, and every check in this script, and every test downstream of it, stayed green. The
  // fixture's audited end IS the previous pin, which is what makes this bump the cause.
  const movedOut = auditFixture(auditSource(OUT_OF_WINDOW), OUT_OF_WINDOW);
  check("a bump that moves the pin OUT of the audited FLUX.2 window is refused", !!movedOut.threw);
  check(
    "the refusal names both revisions, the demotion and how to re-run the audit",
    !!movedOut.threw &&
      movedOut.threw.includes(OUT_OF_WINDOW) &&
      movedOut.threw.includes(SHA) &&
      /flux2_dev q4 cells/.test(movedOut.threw) &&
      /inference-artifact-audit\.mjs/.test(movedOut.threw),
  );
  check(
    "the refusal says the tree is untouched, which is what makes a re-run refuse identically",
    /Nothing has been written/.test(`${movedOut.threw}`),
  );
  // The other half of the transition, and the reason this is not a plain equality check. When the
  // window ends somewhere neither pin touches, the demotion predates this bump: blocking it would
  // stop unrelated work over an ARTIFACTS DIFFER verdict only a re-capture campaign can clear.
  const alreadyOut = auditFixture(auditSource(AUDITED_END), OUT_OF_WINDOW);
  check(
    "a pin already outside the window warns instead of refusing",
    alreadyOut.threw === null && /ALREADY outside/.test(alreadyOut.warned),
  );
  check(
    "the warning still names the remedy, so it is not a silent pass",
    /inference-artifact-audit\.mjs/.test(alreadyOut.warned) &&
      /flux2_dev q4 cells/.test(alreadyOut.warned),
  );
  // An unreadable previous pin (a `tag =` manifest, or lines that disagree) must degrade to the
  // warning. Guessing here would refuse a bump on no evidence.
  const unknownPrevious = auditFixture(auditSource(AUDITED_END), null);
  check(
    "an unknown previous pin degrades to the warning, never to a refusal",
    unknownPrevious.threw === null && /ALREADY outside/.test(unknownPrevious.warned),
  );
  // Anchoring on the captured end instead would read the decoy: the refusal would name `ccc…` as
  // the end of the window (and, because `ccc… !== previousSha`, would not be a refusal at all).
  // Asserting the message NAMES the compatible revision and never the decoy is what discriminates —
  // a bare `!includes(...)` passes vacuously when the branch stops throwing, which is exactly how
  // this check read before it was rewritten.
  check(
    "the neighbouring FLUX2_CAPTURED_INFERENCE_REVISION is not what is compared",
    !!movedOut.threw &&
      movedOut.threw.includes(`ends at ${OUT_OF_WINDOW}`) &&
      !movedOut.threw.includes("c".repeat(40)),
  );
  // Everything above runs over fixtures, so a rustfmt reflow or a rename of the constant is
  // invisible to `npm run check` and would surface only as a hard bump failure months later. This
  // is the one assertion against the REAL file — the same argument this script's own header makes
  // for wiring the self-test into `npm run check` at all.
  const frozenInRepo = FLUX2_COMPATIBLE_RE.exec(readFileSync(FLUX2_AUDIT_FROZEN, "utf8"));
  check(
    "the regex still matches the real inference_compatibility_audit.rs",
    !!frozenInRepo && SHA_RE.test(frozenInRepo[1]),
  );
  const missingConstant = auditFixture('pub(crate) const SOMETHING_ELSE: &str = "x";\n', SHA);
  check(
    "a missing FLUX2_COMPATIBLE_INFERENCE_REVISION is refused rather than skipped",
    !!missingConstant.threw && /not found in/.test(missingConstant.threw),
  );

  // A record already on disk for this pin. The advice has to CHANGE, or the guard sends an operator
  // to spend 15 minutes reproducing a verdict that is checked in — which is what sc-17760's own
  // negative record would have caused on the very next bump.
  const probeRecord = (compatibleDigest, witnessDigest = "sha256:w") => ({
    schemaVersion: FLUX2_ACCEPTED_SCHEMA_VERSION,
    capturedInferenceRevision: "c".repeat(40),
    compatibleInferenceRevision: SHA,
    featureWitness: { capturedDigest: "sha256:w", compatibleDigest: witnessDigest },
    auditedArtifact: { capturedDigest: "sha256:a", compatibleDigest },
  });
  const alreadyProbed = auditFixture(
    auditSource(AUDITED_END),
    "d".repeat(40),
    probeRecord("sha256:b"),
  );
  check(
    "a pin with a checked-in ARTIFACTS DIFFER record is told not to re-run the audit",
    /ALREADY been probed/.test(alreadyProbed.warned) &&
      /ARTIFACTS DIFFER/.test(alreadyProbed.warned) &&
      /sc-15922/.test(alreadyProbed.warned) &&
      !/--compatible/.test(alreadyProbed.warned),
  );
  const witnessProbed = auditFixture(
    auditSource(AUDITED_END),
    "d".repeat(40),
    probeRecord("sha256:a", "sha256:x"),
  );
  // Both verdict strings also occur in the GENERIC remedy, so matching one proves nothing on its
  // own — the probed-branch marker has to be there too, and the generic invocation gone.
  check(
    "a feature-witness verdict is named as itself, not as ARTIFACTS DIFFER",
    /ALREADY been probed/.test(witnessProbed.warned) &&
      /SHIPPED FEATURE SETS DIFFER/.test(witnessProbed.warned) &&
      !/ARTIFACTS DIFFER/.test(witnessProbed.warned) &&
      !/--compatible/.test(witnessProbed.warned),
  );
  // The third state: a record that says the window extends, sitting next to constants that were
  // never moved. Re-running the audit would print the same thing; the checklist is what is unfinished.
  const unfinished = auditFixture(
    auditSource(AUDITED_END),
    "d".repeat(40),
    probeRecord("sha256:a"),
  );
  check(
    "an extending record with unmoved constants points at the checklist, not the audit",
    /DOES extend/.test(unfinished.warned) && /After running it/.test(unfinished.warned),
  );
  // The dangerous fail-open. `inference-compatibility-277f.json` is a REAL v1 record in this very
  // directory with no digests at all; without the schema/completeness gate it computes "nothing
  // moved" and advises going and moving the frozen constants — an authorization derived from a
  // record both validators refuse outright.
  const staleSchema = auditFixture(auditSource(AUDITED_END), "d".repeat(40), {
    schemaVersion: 1,
    compatibleInferenceRevision: SHA,
    capturedInferenceRevision: "c".repeat(40),
  });
  check(
    "a record of a refused schema is not read as an authorization",
    !/DOES extend/.test(staleSchema.warned) &&
      !/ALREADY been probed/.test(staleSchema.warned) &&
      /inference-artifact-audit\.mjs/.test(staleSchema.warned),
  );
  // Same gate from the other side: the accepted schema but a missing digest pair is still not a
  // verdict about anything.
  const halfRecord = auditFixture(auditSource(AUDITED_END), "d".repeat(40), {
    ...probeRecord("sha256:a"),
    featureWitness: {},
  });
  check(
    "an accepted-schema record missing a digest pair is not read as an authorization",
    !/DOES extend/.test(halfRecord.warned) && /inference-artifact-audit\.mjs/.test(halfRecord.warned),
  );
  // A record for a DIFFERENT revision must not be read as evidence about this pin.
  const otherRecord = auditFixture(auditSource(AUDITED_END), "d".repeat(40), {
    ...probeRecord("sha256:b"),
    compatibleInferenceRevision: "f".repeat(40),
  });
  check(
    "a record for another revision is ignored and the generic remedy stands",
    !/ALREADY been probed/.test(otherRecord.warned) &&
      /inference-artifact-audit\.mjs/.test(otherRecord.warned),
  );

  // `pinnedRevision` is what supplies the previous pin, so its ambiguous cases are the ones that
  // decide between the two branches above.
  check(
    "the previous pin is read off the inference lines",
    pinnedRevision(
      `a = { git = "${INFERENCE_GIT}", rev = "${OUT_OF_WINDOW}" }\nb = { git = "${INFERENCE_GIT}", rev = "${OUT_OF_WINDOW}", optional = true }`,
    ) === OUT_OF_WINDOW,
  );
  check(
    "a tag-pinned manifest reports an unknown previous pin",
    pinnedRevision(`a = { git = "${INFERENCE_GIT}", tag = "runtime-2026.07.7" }`) === null,
  );
  check(
    "disagreeing inference lines report an unknown previous pin rather than the first one",
    pinnedRevision(
      `a = { git = "${INFERENCE_GIT}", rev = "${OUT_OF_WINDOW}" }\nb = { git = "${INFERENCE_GIT}", rev = "${AUDITED_END}" }`,
    ) === null,
  );
  check(
    "the direct mlx-rs pin (other url) is not mistaken for the inference pin",
    pinnedRevision(
      `mlx = { git = "https://github.com/michaeltrefry/mlx-rs", rev = "${AUDITED_END}" }\na = { git = "${INFERENCE_GIT}", rev = "${OUT_OF_WINDOW}" }`,
    ) === OUT_OF_WINDOW,
  );

  console.log(rc === 0 ? "self-test: PASS" : "self-test: FAIL");
  process.exit(rc);
}

// --- entrypoint -------------------------------------------------------------------------------

function main() {
  const args = process.argv.slice(2);
  if (args.includes("--self-test")) return selfTest();

  const dryRun = args.includes("--dry-run");
  const shaIdx = args.indexOf("--sha");
  const sha = shaIdx >= 0 ? (args[shaIdx + 1] || "") : latestInferenceSha();
  if (!SHA_RE.test(sha)) {
    console.error(`bump-inference: not a 40-char commit SHA: ${sha}`);
    process.exit(2);
  }

  // Three files the tool can safely rewrite: the worker's direct deps, the root's candle-kernels
  // [patch], and the worker's semantic-provenance stamp. They must land on the same rev, so bump
  // them as one unit. `cargo update` below refreshes a fourth, Cargo.lock.
  //
  // Two further files also carry the revision and are DELIBERATELY left manual, not overlooked:
  // config/inference-third-party-source.json and scripts/scan-inference-provenance.mjs. Those are
  // audit sites -- their revision is a label on a scan result (candidate inventory, population
  // hash, audit digest). Substituting the string without re-running the scan would fake an audit,
  // so a bump must re-run scan-inference-provenance.mjs and recompute the digest by hand:
  //
  //   node scripts/scan-inference-provenance.mjs --repo <inference> \
  //     --write config/inference-provenance-candidates.tsv \
  //     --write-crates config/inference-crate-prefixes.txt
  //
  // Both inventories matter. The candidate TSV is heuristic (a marker regex over doc comments); the
  // crate-prefix list is not, and every production-Rust crate it lists must be classified — ported
  // area or explicit `crateDispositions` decision — or check-license-coverage.mjs FAILS. That guard
  // exists because a whole new crate once slipped the marker vocabulary and shipped unclassified
  // (sc-15138 -> sc-15191).
  const manifests = [
    // DISCOVERED, not enumerated. A hand-written list is exactly how
    // `crates/sceneworks-memory-adapter` got stranded on the previous revision: the crate was
    // added later and nobody remembered to add its manifest here, so the bump rewrote two of the
    // three and left one behind. Discovery makes the next such crate correct by default.
    ...inferenceManifests().map((path) => ({
      path,
      rewrite: (text) => repin(text, sha, path),
    })),
    { path: SEMANTIC_PROVENANCE, rewrite: (text) => repinSemanticProvenance(text, sha) },
    {
      path: MEMORY_PROVENANCE,
      rewrite: (text) => repinMemoryProvenance(text, sha),
    },
  ].map(({ path, rewrite }) => {
    const current = readFileSync(path, "utf8");
    return { path, current, bumped: rewrite(current) };
  });
  // NOTE: "manifests already say `sha`" is NOT the same as "the lockfile agrees". This used to
  // early-return on the manifests alone, so a tree whose manifests were correct but whose
  // `Cargo.lock` still carried the previous revision could never self-heal -- re-running the script
  // just said "already pinned" and did nothing. That is precisely the state a partially-applied bump
  // leaves behind. Fall through to `cargoUpdate()` + `verifyNoSkew()` instead: both are cheap, and
  // running them unconditionally makes the script idempotent AND verifying rather than idempotent
  // and blind. `lockHasStaleInferenceRevision` narrows the report to the case that actually needs
  // repair; it deliberately does NOT gate the skew check, which catches divergences (gen-core,
  // pmetal-mlx-rs) that no revision-string comparison can see.
  // Captured before anything is written: after the rewrite loop below the previous revision is gone
  // from the tree, and `verifyFlux2AuditWindow` needs it to tell a bump that MOVES the pin out of
  // the audited FLUX.2 window from one that merely inherits a pin already outside it.
  const previousPin = pinnedRevision(manifests.find((m) => m.path === MANIFEST)?.current ?? "");
  const manifestsAlreadyPinned = manifests.every((m) => m.bumped === m.current);
  if (manifestsAlreadyPinned) {
    console.log(
      lockHasStaleInferenceRevision(sha)
        ? `bump-inference: manifests already pinned at ${sha}, but the lockfile is STALE; repairing`
        : `bump-inference: manifests already pinned at ${sha}; verifying the lockfile agrees`,
    );
  }
  console.log(
    `bump-inference: pinning inference (${[...INFERENCE_CRATES, ...PATCHED_CRATES].join(", ")}) -> ${sha}`,
  );
  // BEFORE anything is written, unlike every other verifier in this script — and that placement is
  // the difference between a guard and a formality. The sibling guards run post-write and still
  // fail on a re-run, because re-running does not refresh a licence scan or a facts dump. This one
  // keys on a TRANSITION, so a post-write refusal would launder itself: the refused run has already
  // moved the manifests, the next invocation sees `previousPin === sha`, and the warn branch lets
  // the demotion through. Refusing here leaves the tree untouched, so the second run refuses
  // identically. It also costs nothing to move — the check reads two strings and a JSON file.
  //
  // Under `--dry-run` the same call reports instead of throwing: with nothing written there is no
  // incomplete state to protect, and this is the cheapest place to learn that a bump you are
  // considering owes a ~15 min audit re-run on the CUDA box.
  if (dryRun) {
    try {
      verifyFlux2AuditWindow(sha, previousPin);
    } catch (error) {
      console.warn(`bump-inference: this bump WOULD be refused —\n${error?.message ?? error}`);
    }
    console.log("bump-inference: dry run, no files written");
    return;
  }
  verifyFlux2AuditWindow(sha, previousPin);
  for (const m of manifests) {
    if (m.bumped === m.current) {
      console.log(`  unchanged ${m.path} (already at ${sha})`);
      continue;
    }
    writeFileSync(m.path, m.bumped);
    console.log(`  wrote ${m.path}`);
  }
  cargoUpdate(sha);
  verifyNoSkew();
  regenerateMemoryMatrix();
  // Downstream of the matrix, so it must follow it — see the note on the function.
  regenerateCalibrationCostModel();
  verifyLicenseAudit();
  // Beside the licence re-scan for the same reason: both are pin-keyed artifacts that need an input
  // this script cannot synthesize (a local inference clone / a lane that links engines), so both
  // fail closed here rather than in parity CI ten minutes later. sc-16965.
  verifyEngineCapabilityFacts(sha);
  console.log("bump-inference: done");
}

try {
  main();
} catch (err) {
  console.error(`bump-inference: ${err?.message ?? err}`);
  process.exit(1);
}
