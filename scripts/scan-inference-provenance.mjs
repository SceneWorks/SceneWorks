#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";

export const REVISION = "981c80508103c2af264cab05c149605f60a341d2";
export const MARKER =
  /\b(?:faithful(?:\s+\w+){0,3}\s+ports?|ported\s+from|ports?\s+of|vendors?|vendored|transcribed|copied(?:\s+\w+){0,3}\s+verbatim|adapted\s+from)\b/giu;

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

export function scan(repo, revision = REVISION) {
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
  if (!repo || (!output && !compare)) {
    console.error("usage: node scripts/scan-inference-provenance.mjs --repo PATH [--write FILE|--compare FILE]");
    process.exit(2);
  }
  const candidates = scan(path.resolve(repo));
  const rendered = serialize(candidates);
  if (output) fs.writeFileSync(path.resolve(output), rendered);
  if (compare) {
    const committed = fs.readFileSync(path.resolve(compare), "utf8");
    if (committed !== rendered) {
      console.error("pinned inference provenance population differs from committed inventory");
      process.exit(1);
    }
  }
  console.log(`${candidates.length} candidates; population sha256 ${populationSha256(candidates)}`);
}
