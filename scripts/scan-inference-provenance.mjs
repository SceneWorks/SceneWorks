#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";

// 🔴 The revision to scan is DERIVED from the shipped Cargo pin, never hand-maintained here.
//
// It used to be a literal, and that is exactly how the inventory silently went stale (sc-15017):
// a pin bump updated `config/inference-third-party-source.json`'s revision LABEL by hand, re-ran
// this scanner — which scanned the untouched literal, an older revision — saw no diff, and
// committed an inventory that described a revision it was not generated from. The audit was then
// self-consistent and GREEN while missing a whole crate that had entered in between
// (`candle-audio-stable-audio-3`). Two revisions kept in sync by discipline is the defect; there
// is now only one.
const WORKER_MANIFEST = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../crates/sceneworks-worker/Cargo.toml",
);
const INFERENCE_GIT = "https://github.com/SceneWorks/inference";

/** The single inference revision the worker's Cargo manifest pins. Throws if it is not unique. */
export function pinnedRevision(manifestPath = WORKER_MANIFEST) {
  const pins = new Set(
    fs
      .readFileSync(manifestPath, "utf8")
      .split("\n")
      .filter((line) => line.includes(INFERENCE_GIT))
      .map((line) => line.match(/\brev\s*=\s*"([0-9a-f]{40})"/)?.[1])
      .filter(Boolean),
  );
  if (pins.size !== 1) {
    throw new Error(
      `expected exactly one 40-hex inference rev in ${manifestPath}, found: ${[...pins].join(", ") || "none"}`,
    );
  }
  return [...pins][0];
}

// Nouns that turn an otherwise-ordinary English verb into a PROVENANCE claim. "mirrors the buffer"
// is prose; "mirrors the reference" is an admission that upstream source exists. Every broadened
// alternative below is anchored to one of these, because the unanchored forms are common technical
// English in this codebase and would classify first-party crates as ports (measured at the pinned
// rev: bare `mirror(s|ed|ing)` alone matched 275 extra files and dragged 10 crates — catalogs,
// bundle composition roots, the gen-core testkit — into the ported inventory on the strength of
// sentences like "mirrors the `gen_core` lib-name convention"). Over-matching is not free here: a
// candidate file forces a ported-source AREA whose disposition asserts an upstream port, so a false
// positive makes the audit state something untrue.
const PORT_OBJECT =
  "reference|upstream|original|python|pytorch|torch|diffusers|comfy(?:ui)?|transformers|implementation";

// 🔴 This regex is a HEURISTIC and is known to have holes — it is NOT the thing that makes the
// audit complete. `mlx-gen-krea-realtime` was a whole new crate whose headers honestly said
// `mirroring` / `from the reference`, matched nothing here, and so was invisible to the audit while
// the license-coverage gate went green (sc-15138 → sc-15191). Broadening the vocabulary (below)
// narrows the hole; it cannot close it, because the next crate may describe its port in words
// nobody anticipated. The FAIL-CLOSED half of the fix is the crate-prefix coverage guard
// (`scanCrates` + `crateDispositions` in config/inference-third-party-source.json): every
// production-Rust crate in the pinned revision must be classified explicitly, marker or no marker.
// Treat this regex as a labour-saver for classification, never as the detector of record.
export const MARKER = new RegExp(
  "\\b(?:" +
    // Long-standing markers (pre-sc-15191).
    "faithful(?:\\s+\\w+){0,3}\\s+ports?|ported\\s+from|ports?\\s+of|vendors?|vendored" +
    "|transcribed|copied(?:\\s+\\w+){0,3}\\s+verbatim|adapted\\s+from" +
    // sc-15191: honest port phrasings the original vocabulary missed.
    `|mirror(?:s|ed|ing)(?:\\s+\\w+){0,4}\\s+(?:${PORT_OBJECT})` +
    `|derived\\s+from(?:\\s+\\w+){0,3}\\s+(?:${PORT_OBJECT})` +
    "|reimplement(?:ed|ation)(?:\\s+\\w+){0,3}\\s+(?:from|of)" +
    "|translated\\s+from|from\\s+the\\s+reference" +
    `|based\\s+on(?:\\s+\\w+){0,3}\\s+(?:${PORT_OBJECT})` +
    "|follow(?:s|ing|ed)(?:\\s+\\w+){0,2}\\s+(?:reference|upstream)" +
    ")\\b",
  "giu",
);

const SPECIAL_AREAS = new Map([
  ["crates/media/candle-gen/candle-gen-flux/src/vae/native.rs", "candle-transformers-source"],
  ["crates/media/candle-gen/candle-gen-flux/src/vae/diffusers.rs", "candle-transformers-source"],
  ["crates/media/candle-gen/candle-gen-flux/src/packed_te.rs", "candle-transformers-source"],
  ["crates/media/candle-gen/candle-gen-kolors/src/unet.rs", "candle-transformers-source"],
  ["crates/media/candle-gen/candle-gen-instantid/src/kps.rs", "opencv-source"],
  ["crates/media/mlx-gen/mlx-gen-instantid/src/kps.rs", "opencv-source"],
  ["crates/audio/candle-audio-acestep/src/vae.rs", "diffusers-source"],
  ["crates/contracts/gen-core/src/sampling/solvers.rs", "comfy-kdiffusion-solvers"],
  ["crates/contracts/gen-core/src/sampling/cfgpp.rs", "cfgpp-formula"],
  // The Mage latent module transcribes Cephes (ndtri/erf/erfc) under BSD-3 — a distinct third-party
  // source obligation from the surrounding architecture port (sc-14432). Route it to its own
  // `cephes-source` area (component `cephes`) so the notice is attributed, not folded into the
  // `mlx-gen-mage` architecture prefix.
  ["crates/media/mlx-gen/mlx-gen-mage/src/latent.rs", "cephes-source"],
]);

function sha256(text) {
  return crypto.createHash("sha256").update(text).digest("hex");
}

function normalizeMarkers(source) {
  return [...source.matchAll(MARKER)]
    .map((match) => match[0].toLowerCase().replace(/\s+/g, " ").trim())
    .sort()
    .join("\n");
}

function productionRustPath(file) {
  return file.endsWith(".rs") &&
    !/(^|\/)(?:tests?|testdata|benches|examples|_vendor)(?:\/|$)/.test(file);
}

function areaFor(file) {
  if (SPECIAL_AREAS.has(file)) return SPECIAL_AREAS.get(file);
  const src = file.indexOf("/src/");
  return `architecture:${src < 0 ? path.posix.dirname(file) : file.slice(0, src)}`;
}

export function scan(repo, revision = pinnedRevision()) {
  const files = execFileSync("git", ["-C", repo, "ls-tree", "-r", "--name-only", revision], {
    encoding: "utf8",
  }).trim().split("\n").filter(productionRustPath);
  const candidates = [];
  for (const file of files) {
    const source = execFileSync("git", ["-C", repo, "show", `${revision}:${file}`], {
      encoding: "utf8",
      maxBuffer: 32 * 1024 * 1024,
    });
    const markers = normalizeMarkers(source);
    if (!markers) continue;
    const blob = execFileSync("git", ["-C", repo, "rev-parse", `${revision}:${file}`], {
      encoding: "utf8",
    }).trim();
    candidates.push({ path: file, blob, markerSha256: sha256(markers), area: areaFor(file) });
  }
  return candidates.sort((a, b) => a.path.localeCompare(b.path));
}

/**
 * Every crate prefix in `revision` that owns at least one production `.rs` file.
 *
 * This is the population the FAIL-CLOSED coverage guard runs over (sc-15191). Unlike
 * {@link scan}, it does not read a single byte of source: a crate is present because it exists and
 * ships production Rust, not because someone wrote a recognizable sentence in a doc comment. That
 * is the whole point — the marker regex can only find crates that describe themselves in words we
 * anticipated, so it can never prove the inventory is complete. This can.
 *
 * A file belongs to the NEAREST enclosing `Cargo.toml`, so a nested crate is its own prefix rather
 * than being absorbed by its parent.
 */
export function scanCrates(repo, revision = pinnedRevision()) {
  const files = execFileSync("git", ["-C", repo, "ls-tree", "-r", "--name-only", revision], {
    encoding: "utf8",
  }).trim().split("\n");
  const crateDirs = files
    .filter((file) => file === "Cargo.toml" || file.endsWith("/Cargo.toml"))
    .map((file) => (file === "Cargo.toml" ? "" : file.slice(0, -"/Cargo.toml".length)))
    .sort((a, b) => b.length - a.length);
  const owning = new Set();
  for (const file of files.filter(productionRustPath)) {
    const owner = crateDirs.find((dir) => dir === "" || file.startsWith(`${dir}/`));
    if (owner) owning.add(owner);
  }
  return [...owning].sort((a, b) => a.localeCompare(b));
}

export function serializeCrates(crates) {
  return [
    "# Production-Rust crate prefixes in the pinned inference revision (scan-inference-provenance.mjs).",
    "# Every prefix here must be classified in config/inference-third-party-source.json — either by a",
    "# portedSourceAreas entry covering it, or by an explicit crateDispositions decision. Unclassified",
    "# prefixes FAIL check-license-coverage.mjs; there is no silent default.",
    ...crates,
    "",
  ].join("\n");
}

export function parseCrates(text) {
  return text.split(/\r?\n/).filter((line) => line && !line.startsWith("#")).map((line) => {
    if (/\s/.test(line)) throw new Error(`malformed crate prefix row: ${line}`);
    return line;
  });
}

export function cratePopulationSha256(crates) {
  return sha256(serializeCrates(crates));
}

export function serialize(candidates) {
  return [
    "# path\tgit_blob_sha1\tnormalized_marker_sha256\tsource_area",
    ...candidates.map((item) =>
      [item.path, item.blob, item.markerSha256, item.area].join("\t")),
    "",
  ].join("\n");
}

export function parse(text) {
  return text.split(/\r?\n/).filter((line) => line && !line.startsWith("#")).map((line) => {
    const [candidatePath, blob, markerSha256, area, ...extra] = line.split("\t");
    if (extra.length || !candidatePath || !/^[0-9a-f]{40}$/.test(blob ?? "") ||
        !/^[0-9a-f]{64}$/.test(markerSha256 ?? "") || !area) {
      throw new Error(`malformed provenance candidate row: ${line}`);
    }
    return { path: candidatePath, blob, markerSha256, area };
  });
}

export function populationSha256(candidates) {
  return sha256(serialize(candidates));
}

if (import.meta.url === new URL(`file://${process.argv[1]}`).href) {
  const args = process.argv.slice(2);
  const value = (flag) => {
    const index = args.indexOf(flag);
    return index < 0 ? undefined : args[index + 1];
  };
  const repo = value("--repo");
  const output = value("--write");
  const compare = value("--compare");
  const crateOutput = value("--write-crates");
  const crateCompare = value("--compare-crates");
  // Defaults to the shipped Cargo pin, so an audit can never scan a revision the product does not
  // build. `--revision` is for auditing a candidate rev BEFORE bumping, not for routine use.
  const revision = value("--revision") ?? pinnedRevision();
  const wants = output || compare || crateOutput || crateCompare;
  if (!repo || !wants || !/^[0-9a-f]{40}$/.test(revision)) {
    console.error("usage: node scripts/scan-inference-provenance.mjs --repo PATH [--revision SHA40] [--write FILE|--compare FILE] [--write-crates FILE|--compare-crates FILE]");
    process.exit(2);
  }
  const resolvedRepo = path.resolve(repo);
  let failed = false;
  if (output || compare) {
    const candidates = scan(resolvedRepo, revision);
    const rendered = serialize(candidates);
    if (output) fs.writeFileSync(path.resolve(output), rendered);
    if (compare && fs.readFileSync(path.resolve(compare), "utf8") !== rendered) {
      console.error("pinned inference provenance population differs from committed inventory");
      failed = true;
    }
    console.log(`${revision}: ${candidates.length} candidates; population sha256 ${populationSha256(candidates)}`);
  }
  if (crateOutput || crateCompare) {
    const crates = scanCrates(resolvedRepo, revision);
    const rendered = serializeCrates(crates);
    if (crateOutput) fs.writeFileSync(path.resolve(crateOutput), rendered);
    if (crateCompare && fs.readFileSync(path.resolve(crateCompare), "utf8") !== rendered) {
      console.error("pinned inference crate-prefix population differs from committed inventory");
      failed = true;
    }
    console.log(`${revision}: ${crates.length} production-Rust crate prefixes; population sha256 ${cratePopulationSha256(crates)}`);
  }
  if (failed) process.exit(1);
}
