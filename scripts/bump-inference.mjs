#!/usr/bin/env node
// Bump SceneWorks' inference pin to a commit on inference `main`, so day-to-day development tracks
// inference by SHA instead of waiting on a cut `runtime-*` release. Inference is co-developed and
// SceneWorks is its only consumer, so a formal release does not belong on the critical path between
// the two repos -- releases stay for durable/shareable snapshots (inference's cut_release.py).
//
// The pins live in crates/sceneworks-worker/Cargo.toml and
// crates/sceneworks-image-memory-adapter/Cargo.toml. The root Cargo.toml additionally `[patch]`es
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
//   node scripts/bump-inference.mjs --self-test     # exercise the pin rewrite on canned input

import { execFileSync } from "node:child_process";
import { readdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const MANIFEST = join(repoRoot, "crates/sceneworks-worker/Cargo.toml");
const IMAGE_MEMORY_MANIFEST = join(repoRoot, "crates/sceneworks-image-memory-adapter/Cargo.toml");
const LOCKFILE = join(repoRoot, "Cargo.lock");
// Root workspace manifest: holds the candle-kernels [patch] pin (same repo, same rev).
const ROOT_MANIFEST = join(repoRoot, "Cargo.toml");
const INFERENCE_GIT = "https://github.com/SceneWorks/inference";
// Every inference crate any workspace manifest depends on, so `cargo update -p` refreshes ALL of
// their lock entries. `mlx-gen` / `candle-gen` come in through
// `crates/sceneworks-image-memory-adapter` -- omitting them left that crate's four deps stranded on
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
// `crates/sceneworks-image-memory-adapter/Cargo.toml` -- which carries four of its own inference deps
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
const IMAGE_MEMORY_PROVENANCE = join(repoRoot, "crates/sceneworks-image-memory-adapter/src/lib.rs");
const IMAGE_MEMORY_PROVENANCE_RE = /(pub const INFERENCE_PIN: &str = ")[0-9a-f]{40}(";)/;
const SHA_RE = /^[0-9a-f]{40}$/;

// --- pure: rewrite the inference pins to rev=<sha> (self-tested; no fs/network) ---------------

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

function repinImageMemoryProvenance(source, sha) {
  if (!IMAGE_MEMORY_PROVENANCE_RE.test(source)) {
    throw new Error(
      `no INFERENCE_PIN constant found in ${IMAGE_MEMORY_PROVENANCE} -- the calibration adapter ` +
        "provenance stamp must move with the pin",
    );
  }
  return source.replace(IMAGE_MEMORY_PROVENANCE_RE, `$1${sha}$2`);
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
    "image-memory adapter provenance stamp bumps with the pin",
    repinImageMemoryProvenance(
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
    // `crates/sceneworks-image-memory-adapter` got stranded on the previous revision: the crate was
    // added later and nobody remembered to add its manifest here, so the bump rewrote two of the
    // three and left one behind. Discovery makes the next such crate correct by default.
    ...inferenceManifests().map((path) => ({
      path,
      rewrite: (text) => repin(text, sha, path),
    })),
    { path: SEMANTIC_PROVENANCE, rewrite: (text) => repinSemanticProvenance(text, sha) },
    {
      path: IMAGE_MEMORY_PROVENANCE,
      rewrite: (text) => repinImageMemoryProvenance(text, sha),
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
  if (dryRun) {
    console.log("bump-inference: dry run, no files written");
    return;
  }
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
  console.log("bump-inference: done");
}

try {
  main();
} catch (err) {
  console.error(`bump-inference: ${err?.message ?? err}`);
  process.exit(1);
}
