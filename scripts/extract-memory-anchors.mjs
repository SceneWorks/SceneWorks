#!/usr/bin/env node
/**
 * sc-22510 — extract the measured memory ANCHOR store from the retained evidence, and classify
 * every routing-catalog cell the evidence cannot anchor as explicitly ANALYTIC-ONLY.
 *
 * Epic 22505 replaces grid measurement with `anchor + analytic derivation`
 * (`crates/sceneworks-core/src/memory_anchor.rs`). An anchor is the measured per-phase peak
 * decomposition of ONE retained render; everything else is derived from it. This script is the
 * migration half: it walks the evidence the repo already retains — ZERO new renders, zero new
 * measurements — and writes `config/memory-anchors.json` for the FULL catalog, image and video,
 * both backend lanes.
 *
 * The store has exactly two kinds of row and no third state:
 *
 * * `anchors` — a cell whose retained corpus contains a render with the three phase peaks
 *   (`conditioningActivePeak` / `denoiseActivePeak` / `decodeActivePeak`) and an overall allocator
 *   envelope. Byte-exact source provenance is recorded and re-checked by `memory_anchor.rs`
 *   against the compiled-in evidence, so the store cannot drift from the corpus it cites.
 * * `analyticOnly` — a cell with no such render. It is NOT a gap: the row names WHY (`basis`) and
 *   carries whatever weaker evidence does exist, so "nobody measured this" is distinguishable from
 *   "this was measured and the record was dropped".
 *
 * Sources walked (all read-only, all repo-retained except the last):
 *
 *   1. every `{records: [...]}` corpus under `docs/calibration/` and `docs/generated/` — the
 *      committed evidence bundle, the per-story receipt seeds and the geometry sweeps;
 *   2. `docs/generated/video-memory-curves.json` — the fitted curves' `evidence.sources` name the
 *      record corpora the video lane's identities were measured from, which is how the LTX-2.3
 *      sweep is reached (its paths are asserted to be part of the discovered corpus set);
 *   3. `config/manifests/builtin.models.jsonc` — `measured: true` `vramGbByTier` /
 *      `sequentialPeakGb` tier tables, a per-tier declared envelope but not a phase decomposition;
 *   4. the pinned inference revision's per-provider `mlx-gen-<family>/src/memory_strategy.rs` measured
 *      byte constants, read from the cargo checkout OF THE PIN (never a working tree, never a
 *      different revision). When that checkout is absent the previously extracted values are
 *      carried forward from the existing store, so the output is identical either way.
 *
 * The catalog population is NOT re-invented here: it comes from `buildMatrix()` in
 * `scripts/generate-memory-matrix.mjs`, the same resolution the memory matrix publishes, so a
 * catalog change moves both artifacts or neither.
 *
 * DETERMINISM is a contract, not an accident: every collection is sorted by an explicit code-unit
 * comparison, every selection rule is total, and re-running over the same corpora reproduces the
 * file byte-identically (`scripts/extract-memory-anchors.test.mjs`).
 *
 * IDEMPOTENCE / UNION: an anchor already in the store whose identity this run does not produce is
 * carried forward rather than dropped, so a concurrently landed anchor (sc-22509's candle proving
 * models) survives a re-run instead of being clobbered by it. Authority is scoped to the corpora
 * this run WALKED: a row citing one of them that the run did not re-produce was rejected on
 * purpose — an overlay render, a record that lost its phase peaks — and is withdrawn rather than
 * made immortal by the carry-forward.
 *
 * Usage: node scripts/extract-memory-anchors.mjs [--check] [--inference-root <dir>]
 */
import { createHash } from "node:crypto";
import { readFile, writeFile, rename, readdir } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

import { stripJsoncComments } from "./lib/jsonc.mjs";
import { buildMatrix } from "./generate-memory-matrix.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

export const STORE_PATH = "config/memory-anchors.json";
export const MEMORY_ANCHOR_SCHEMA_VERSION = 1;

/** Directories whose `{records: [...]}` JSON files are retained calibration corpora. */
export const CORPUS_ROOTS = ["docs/calibration", "docs/generated"];

/** The fitted video curves, whose `evidence.sources` name the corpora the video lane measured. */
export const VIDEO_CURVES_PATH = "docs/generated/video-memory-curves.json";

export const MANIFEST_PATH = "config/manifests/builtin.models.jsonc";

/** Where the inference pin is declared; the extractor reads the mlx providers AT THIS REVISION. */
export const PIN_PATH = "crates/sceneworks-worker/Cargo.toml";

/** The three phase peaks an anchor is made of. Absent any one of them, the record cannot anchor. */
const PHASE_MEASUREMENTS = {
  conditioning: "conditioningActivePeak",
  denoise: "denoiseActivePeak",
  decode: "decodeActivePeak",
};

/**
 * Analytic-only bases, strongest evidence first. The order IS the precedence: a cell is classified
 * by the best evidence that exists for it, and `no_retained_evidence` is the honest floor rather
 * than a silent gap.
 */
export const ANALYTIC_BASES = [
  "measured_envelope",
  "provider_measured_constants",
  "manifest_tier_declaration",
  "no_retained_evidence",
];

// Artifact ordering must not depend on the host's ICU locale: every persisted identifier is a
// UTF-8 protocol string, so compare code units directly (same rule as fit-ltx-temporal-form.mjs).
const compareText = (left, right) => (left < right ? -1 : left > right ? 1 : 0);

const sha256 = (body) => createHash("sha256").update(body).digest("hex");

const toPosix = (value) => value.split(path.sep).join("/");

/** Every JSON file under the corpus roots, in a stable repo-relative order. */
async function listCorpusFiles(root) {
  const found = [];
  for (const relative of CORPUS_ROOTS) {
    const base = path.join(root, relative);
    let entries;
    try {
      entries = await readdir(base, { withFileTypes: true, recursive: true });
    } catch {
      continue;
    }
    for (const entry of entries) {
      if (!entry.isFile() || !entry.name.endsWith(".json")) continue;
      const absolute = path.join(entry.parentPath ?? entry.path ?? base, entry.name);
      found.push(toPosix(path.relative(root, absolute)));
    }
  }
  return found.sort(compareText);
}

/**
 * Read every discovered corpus: a file qualifies when it parses and carries a `records` array.
 * Anything else under those trees (plans, audits, request receipts) is skipped by SHAPE rather
 * than by a hardcoded allowlist that a new story's seed would have to be added to by hand.
 */
export async function loadCorpora(root = ROOT, files = null) {
  const paths = files ?? (await listCorpusFiles(root));
  const corpora = [];
  for (const relative of paths) {
    let body;
    try {
      body = await readFile(path.join(root, relative), "utf8");
    } catch {
      continue;
    }
    let parsed;
    try {
      parsed = JSON.parse(body);
    } catch {
      continue;
    }
    if (!parsed || typeof parsed !== "object" || !Array.isArray(parsed.records)) continue;
    corpora.push({ path: relative, sha256: sha256(body), records: parsed.records });
  }
  return corpora;
}

const measurementsOf = (record) => {
  const values = new Map();
  const entries = record?.diagnostics?.measurements;
  if (!Array.isArray(entries)) return values;
  for (const entry of entries) {
    if (typeof entry?.name === "string" && Number.isInteger(entry?.value)) {
      values.set(entry.name, entry.value);
    }
  }
  return values;
};

const envelopeOf = (record) => {
  const bytes = record?.observedMemory?.overall?.allocatorBytes;
  return Number.isInteger(bytes) && bytes > 0 ? bytes : null;
};

export const cellKey = (modelId, backend, tier) => `${modelId}:${backend}:${tier}`;

/**
 * The anchor identity cell: `(model, tier, backend lane, transformer variant, decoder)` — the same
 * key `memory_anchor.rs` refuses duplicates on. A corpus that carries no pipeline axes (every
 * image record, and the LTX-2.3 sweep) keys on `null` for both rather than inventing a variant the
 * record does not state.
 */
export const identityKey = (candidate) =>
  [
    candidate.modelId,
    candidate.backend,
    candidate.tier,
    candidate.transformerVariant ?? "-",
    candidate.decoder ?? "-",
  ].join(":");

/** One retained record's anchor candidacy. `null` when the record cannot anchor anything. */
export function anchorCandidate(record, corpus) {
  const target = record?.target;
  if (!target || typeof target !== "object") return null;
  const modelId = target.modelId;
  const tier = target.tier;
  const backend = record.backend;
  const recordId = record.id;
  if (typeof modelId !== "string" || typeof tier !== "string") return null;
  if (backend !== "mlx" && backend !== "candle") return null;
  if (typeof recordId !== "string") return null;
  const measured = measurementsOf(record);
  const phases = {};
  for (const [phase, name] of Object.entries(PHASE_MEASUREMENTS)) {
    const value = measured.get(name);
    if (!Number.isInteger(value) || value <= 0) return null;
    phases[phase] = value;
  }
  const envelope = envelopeOf(record);
  if (envelope === null) return null;
  const geometry = target.geometry ?? {};
  const width = geometry.width;
  const height = geometry.height;
  const frames = geometry.frames;
  if (!Number.isInteger(width) || !Number.isInteger(height) || !Number.isInteger(frames)) {
    return null;
  }
  if (width <= 0 || height <= 0 || frames <= 0) return null;
  const loadShape = record.loadShape;
  if (loadShape !== "eager_materialization" && loadShape !== "deferred_materialization") {
    return null;
  }
  const engaged = Array.isArray(record.strategy?.engagedRungs) ? record.strategy.engagedRungs : [];
  const fps = measured.get("outputFps");
  const provider = typeof target.provider === "string" ? target.provider : null;
  if (provider === null) return null;
  return {
    modelId,
    backend,
    tier,
    mode: typeof target.mode === "string" ? target.mode : "text_to_image",
    provider,
    // `target.route` is absent from every retained corpus; `video_memory_curves::source_route` and
    // `memory_anchor::validate_anchor` both read `target.provider` as the route spelling there, so
    // this must not invent a different one.
    route: typeof target.route === "string" ? target.route : provider,
    transformerVariant:
      typeof target.transformerVariant === "string" ? target.transformerVariant : null,
    decoder: typeof target.decoder === "string" ? target.decoder : null,
    overlay: target.overlay === "none" || target.overlay === undefined ? null : target.overlay,
    referenceCount: Number.isInteger(target.referenceCount) ? target.referenceCount : 0,
    loadShape,
    measuredRegime: {
      decodeTiled: engaged.includes("bounded_decode"),
      transformerWindowed: engaged.includes("bounded_transformer_residency"),
    },
    calibrationFingerprint:
      typeof record.calibrationFingerprint === "string" ? record.calibrationFingerprint : null,
    geometry: {
      width,
      height,
      frames,
      fps: Number.isInteger(fps) && fps > 0 ? fps : null,
    },
    phaseActivePeakBytes: phases,
    overallAllocatorEnvelopeBytes: envelope,
    sourcePath: corpus.path,
    sourceSha256: corpus.sha256,
    recordId,
  };
}

/**
 * The representative render for one identity cell, chosen mechanically: the LARGEST measured
 * allocator envelope, tie-broken by (source path, record id). The largest envelope is the most
 * binding retained observation of that cell, and every tie-break term is a stable string, so the
 * choice cannot move with corpus iteration order.
 */
export function selectRepresentative(candidates) {
  return candidates.reduce((best, candidate) => {
    if (best === null) return candidate;
    if (candidate.overallAllocatorEnvelopeBytes !== best.overallAllocatorEnvelopeBytes) {
      return candidate.overallAllocatorEnvelopeBytes > best.overallAllocatorEnvelopeBytes
        ? candidate
        : best;
    }
    const byPath = compareText(candidate.sourcePath, best.sourcePath);
    if (byPath !== 0) return byPath < 0 ? candidate : best;
    return compareText(candidate.recordId, best.recordId) < 0 ? candidate : best;
  }, null);
}

const anchorId = (candidate) =>
  [
    candidate.modelId,
    candidate.backend,
    candidate.tier,
    candidate.transformerVariant ?? "base",
    candidate.decoder ?? "base",
    candidate.calibrationFingerprint ?? "unfingerprinted",
    candidate.recordId,
  ].join(":");

/** Serialise one candidate into the store's anchor shape (field order is the file's field order). */
function anchorRow(candidate, catalogCell) {
  return {
    id: anchorId(candidate),
    modelId: candidate.modelId,
    modelFamily: catalogCell.modelFamily,
    route: candidate.route,
    provider: candidate.provider,
    backend: candidate.backend,
    tier: candidate.tier,
    transformerVariant: candidate.transformerVariant,
    decoder: candidate.decoder,
    mode: candidate.mode,
    overlay: candidate.overlay,
    referenceCount: candidate.referenceCount,
    loadShape: candidate.loadShape,
    measuredRegime: {
      decodeTiled: candidate.measuredRegime.decodeTiled,
      transformerWindowed: candidate.measuredRegime.transformerWindowed,
    },
    source: {
      path: candidate.sourcePath,
      sha256: candidate.sourceSha256,
      recordId: candidate.recordId,
      calibrationFingerprint: candidate.calibrationFingerprint,
    },
    geometry: {
      width: candidate.geometry.width,
      height: candidate.geometry.height,
      frames: candidate.geometry.frames,
      fps: candidate.geometry.fps,
    },
    phaseActivePeakBytes: {
      conditioning: candidate.phaseActivePeakBytes.conditioning,
      denoise: candidate.phaseActivePeakBytes.denoise,
      decode: candidate.phaseActivePeakBytes.decode,
    },
    overallAllocatorEnvelopeBytes: candidate.overallAllocatorEnvelopeBytes,
  };
}

/**
 * The routing catalog's `(model, backend lane, tier)` population, resolved by the memory matrix's
 * own catalog pass. Re-deriving it here from the manifest would be a second spelling of the model
 * universe that could drift from the published one.
 */
export async function catalogCells(matrix) {
  const cells = [];
  for (const model of matrix.models) {
    for (const backend of model.backends) {
      const axes = model.axes?.[backend];
      if (!axes) continue;
      for (const tier of axes.tiers) {
        cells.push({
          modelId: model.id,
          modelFamily: model.family ?? model.familyGroup,
          route: model.resolvedRoutes?.[backend] ?? model.resolvedRoute,
          modality: model.modality,
          backend,
          tier,
        });
      }
    }
  }
  return cells.sort((left, right) =>
    compareText(
      cellKey(left.modelId, left.backend, left.tier),
      cellKey(right.modelId, right.backend, right.tier),
    ),
  );
}

/**
 * The weakest measured evidence a cell can still carry: a retained render with NO phase
 * decomposition (every candle corpus is this shape — its four `observedMemory` blocks are one
 * driver-level figure repeated), so it bounds the envelope without anchoring the derivation.
 */
export function envelopeEvidence(corpora, cell) {
  let best = null;
  for (const corpus of corpora) {
    for (const record of corpus.records) {
      if (record?.backend !== cell.backend) continue;
      if (record?.target?.modelId !== cell.modelId || record?.target?.tier !== cell.tier) continue;
      const envelope = envelopeOf(record);
      if (envelope === null || typeof record.id !== "string") continue;
      // The overlay and provider of the cited render are carried, not elided: an envelope measured
      // under a control overlay is evidence ABOUT a different resident set, and a reader has to be
      // able to see that from the row.
      const overlay =
        typeof record.target?.overlay === "string" && record.target.overlay !== "none"
          ? record.target.overlay
          : null;
      const values = {
        ...(typeof record.target?.provider === "string"
          ? { provider: record.target.provider }
          : {}),
        ...(overlay === null ? {} : { overlay }),
      };
      const candidate = {
        repo: null,
        revision: null,
        path: corpus.path,
        sha256: corpus.sha256,
        recordId: record.id,
        envelopeBytes: envelope,
        values: Object.keys(values).length === 0 ? null : values,
      };
      if (best === null || candidate.envelopeBytes > best.envelopeBytes) {
        best = candidate;
        continue;
      }
      if (candidate.envelopeBytes < best.envelopeBytes) continue;
      const byPath = compareText(candidate.path, best.path);
      if (byPath < 0 || (byPath === 0 && compareText(candidate.recordId, best.recordId) < 0)) {
        best = candidate;
      }
    }
  }
  return best;
}

/** `measured: true` tier tables in the catalog manifest — a declared envelope, per tier. */
export function manifestTierEvidence(manifest, manifestPath, manifestSha256, cell) {
  const model = manifest.models?.find((entry) => entry.id === cell.modelId);
  const block = model?.[cell.backend];
  if (!block || block.measured !== true) return null;
  const values = {};
  for (const field of ["vramGbByTier", "sequentialPeakGb"]) {
    const declared = block[field]?.[cell.tier];
    if (typeof declared === "number" && Number.isFinite(declared)) {
      values[field] = String(declared);
    }
  }
  if (Object.keys(values).length === 0) return null;
  return {
    repo: null,
    revision: null,
    path: `${manifestPath}#models/${cell.modelId}/${cell.backend}`,
    sha256: manifestSha256,
    recordId: null,
    envelopeBytes: null,
    values,
  };
}

/** The pinned inference revision, read from the worker's git dependency declaration. */
export function inferencePin(cargoToml) {
  // Scoped to the inference remote on purpose: the same file pins other git dependencies (mlx-rs),
  // and a bare `rev = ` sweep would read one of those as the inference pin.
  const revisions = [
    ...cargoToml.matchAll(
      /github\.com\/SceneWorks\/inference"\s*,\s*rev\s*=\s*"([0-9a-f]{40})"/g,
    ),
  ].map((match) => match[1]);
  const unique = [...new Set(revisions)];
  if (unique.length !== 1) {
    throw new Error(
      `${PIN_PATH} declares ${unique.length} distinct inference revisions; expected exactly one`,
    );
  }
  return unique[0];
}

/**
 * Top-level `pub const <NAME>_BYTES: u64 = <n>;` measured constants in one provider source. Only
 * column-zero declarations count: the same spelling nested inside a function or a `#[cfg(test)]`
 * module is a fixture, not a provider fact.
 */
export function providerByteConstants(source) {
  const values = {};
  for (const match of source.matchAll(/^pub const ([A-Z0-9_]*BYTES): u64 = ([0-9_]+);/gm)) {
    values[match[1]] = String(Number.parseInt(match[2].replaceAll("_", ""), 10));
  }
  return values;
}

/**
 * The cargo checkout of the PINNED inference revision, if this host has one. Cargo names checkout
 * directories by the revision's short hash, so this locates the pin exactly rather than reading
 * whatever a working tree currently holds.
 */
export async function locateInferenceCheckout(revision, home = os.homedir()) {
  const base = path.join(home, ".cargo/git/checkouts");
  let entries;
  try {
    entries = await readdir(base, { withFileTypes: true });
  } catch {
    return null;
  }
  const found = [];
  for (const entry of entries) {
    if (!entry.isDirectory() || !entry.name.startsWith("inference-")) continue;
    let revisions;
    try {
      revisions = await readdir(path.join(base, entry.name), { withFileTypes: true });
    } catch {
      continue;
    }
    for (const candidate of revisions) {
      if (candidate.isDirectory() && revision.startsWith(candidate.name)) {
        found.push(path.join(base, entry.name, candidate.name));
      }
    }
  }
  return found.sort(compareText)[0] ?? null;
}

/** Normalised provider crate names under `crates/media/mlx-gen`, longest first. */
async function mlxProviderCrates(checkoutRoot) {
  const base = path.join(checkoutRoot, "crates/media/mlx-gen");
  let entries;
  try {
    entries = await readdir(base, { withFileTypes: true });
  } catch {
    return [];
  }
  return entries
    .filter((entry) => entry.isDirectory() && entry.name.startsWith("mlx-gen-"))
    .map((entry) => ({
      crate: entry.name,
      key: entry.name.slice("mlx-gen-".length).replaceAll("-", "_"),
      source: path.join(base, entry.name, "src/memory_strategy.rs"),
    }))
    .sort((left, right) => right.key.length - left.key.length || compareText(left.key, right.key));
}

/**
 * Measured byte constants published by the pinned MLX providers, keyed by catalog model id. The
 * provider crate is matched by LONGEST normalised-name prefix of the id (`flux2_dev` ->
 * `mlx-gen-flux2`, `z_image_turbo` -> `mlx-gen-z-image`), which is how the routing catalog's model
 * ids relate to the engine crates.
 */
export async function inferenceProviderConstants(checkoutRoot, revision, modelIds) {
  if (checkoutRoot === null) return new Map();
  const crates = await mlxProviderCrates(checkoutRoot);
  const byRoute = new Map();
  for (const route of [...modelIds].sort(compareText)) {
    const crate = crates.find(
      (entry) => route === entry.key || route.startsWith(`${entry.key}_`),
    );
    if (!crate) continue;
    let source;
    try {
      source = await readFile(crate.source, "utf8");
    } catch {
      continue;
    }
    const values = providerByteConstants(source);
    if (Object.keys(values).length === 0) continue;
    byRoute.set(route, {
      repo: "SceneWorks/inference",
      revision,
      path: `crates/media/mlx-gen/${crate.crate}/src/memory_strategy.rs`,
      sha256: sha256(source),
      recordId: null,
      envelopeBytes: null,
      values,
    });
  }
  return byRoute;
}

const analyticId = (cell) => `analytic:${cellKey(cell.modelId, cell.backend, cell.tier)}`;

const REASONS = {
  measured_envelope:
    "the retained corpus measures this cell's overall allocator envelope but carries no per-phase " +
    "decomposition, so it bounds admission without anchoring the derivation",
  provider_measured_constants:
    "no retained render for this cell; the pinned MLX provider publishes measured component/stage " +
    "byte constants, which price components rather than a render peak",
  manifest_tier_declaration:
    "no retained render for this cell; the catalog manifest declares a measured per-tier envelope, " +
    "which is a whole-render figure with no phase decomposition",
  no_retained_evidence:
    "no retained render, provider constant or measured manifest declaration covers this cell; its " +
    "peak is derived analytically from architecture facts alone",
};

/**
 * Build the complete store. `existingStore` supplies the two carry-forward rules: anchors this run
 * does not produce are preserved (union with concurrent work), and provider constants extracted at
 * the pin survive on a host with no cargo checkout of it.
 */
export async function buildAnchorStore({
  root = ROOT,
  matrix = null,
  existingStore = null,
  inferenceRoot = undefined,
} = {}) {
  const resolvedMatrix = matrix ?? (await buildMatrix());
  const cells = await catalogCells(resolvedMatrix);
  const corpora = await loadCorpora(root);
  const manifestBody = await readFile(path.join(root, MANIFEST_PATH), "utf8");
  const manifest = JSON.parse(stripJsoncComments(manifestBody));
  const manifestSha256 = sha256(manifestBody);
  const pin = inferencePin(await readFile(path.join(root, PIN_PATH), "utf8"));

  // The fitted video curves name the corpora the video lane's identities were measured from. They
  // are not a separate reader: the assertion is that corpus discovery already reached them, so a
  // renamed or dropped sweep fails here instead of silently narrowing the anchor population.
  const curves = JSON.parse(await readFile(path.join(root, VIDEO_CURVES_PATH), "utf8"));
  const curveSources = new Set();
  for (const curve of curves.curves ?? []) {
    for (const source of curve.evidence?.sources ?? []) {
      if (typeof source?.path === "string") curveSources.add(source.path);
    }
  }
  const discovered = new Set(corpora.map((corpus) => corpus.path));
  const missing = [...curveSources].filter((source) => !discovered.has(source)).sort(compareText);
  if (missing.length > 0) {
    throw new Error(
      `${VIDEO_CURVES_PATH} cites record corpora that corpus discovery did not reach: ${missing.join(", ")}`,
    );
  }

  const catalogByCell = new Map(
    cells.map((cell) => [cellKey(cell.modelId, cell.backend, cell.tier), cell]),
  );

  // 1. Anchors from the retained corpora, one per identity cell, catalog-scoped.
  const byIdentity = new Map();
  for (const corpus of corpora) {
    for (const record of corpus.records) {
      const candidate = anchorCandidate(record, corpus);
      if (candidate === null) continue;
      // An OVERLAY render measures a different resident set (krea's q4 MLX evidence is
      // control-branch-only, under its own `*_control` provider). Anchoring the base cell from it
      // would let one provider's measurement answer for another's render, which is exactly what
      // the anchor's provider binding exists to prevent. The cell stays analytic-only, and the
      // overlay record is still cited there as envelope evidence rather than thrown away.
      if (candidate.overlay !== null) continue;
      const cell = catalogByCell.get(
        cellKey(candidate.modelId, candidate.backend, candidate.tier),
      );
      // A record for a coordinate the routing catalog does not resolve cannot be admitted against,
      // so an anchor for it would be an unreachable row.
      if (!cell) continue;
      const key = identityKey(candidate);
      const bucket = byIdentity.get(key);
      if (bucket) bucket.push(candidate);
      else byIdentity.set(key, [candidate]);
    }
  }
  const extracted = new Map();
  for (const [key, candidates] of byIdentity) {
    const chosen = selectRepresentative(candidates);
    const cell = catalogByCell.get(cellKey(chosen.modelId, chosen.backend, chosen.tier));
    extracted.set(key, anchorRow(chosen, cell));
  }

  // 2. Union with anchors already in the store that this run did not produce (sc-22509 lands its
  //    candle proving-model anchors concurrently; a re-run must not delete them).
  for (const anchor of existingStore?.anchors ?? []) {
    const key = identityKey({
      modelId: anchor.modelId,
      backend: anchor.backend,
      tier: anchor.tier,
      transformerVariant: anchor.transformerVariant ?? null,
      decoder: anchor.decoder ?? null,
    });
    if (extracted.has(key)) continue;
    if (!catalogByCell.has(cellKey(anchor.modelId, anchor.backend, anchor.tier))) continue;
    // Authority is scoped to the corpora this run WALKED. A row citing one of them that this run
    // did not re-produce was rejected on purpose (an overlay render, a record that lost its phase
    // peaks), and carrying it forward anyway would make a withdrawn anchor immortal. A row citing
    // evidence outside those corpora is another story's, and is preserved untouched.
    if (discovered.has(anchor.source?.path)) continue;
    extracted.set(key, anchor);
  }
  const anchors = [...extracted.values()].sort((left, right) => compareText(left.id, right.id));
  const anchoredCells = new Set(
    anchors.map((anchor) => cellKey(anchor.modelId, anchor.backend, anchor.tier)),
  );

  // 3. Provider constants at the PIN, with carry-forward when this host has no checkout of it.
  const mlxModelIds = new Set(
    cells.filter((cell) => cell.backend === "mlx").map((cell) => cell.modelId),
  );
  const checkout =
    inferenceRoot === undefined ? await locateInferenceCheckout(pin) : inferenceRoot;
  const providerConstants = await inferenceProviderConstants(checkout, pin, mlxModelIds);
  const carriedProviderEvidence = new Map();
  for (const entry of existingStore?.analyticOnly ?? []) {
    if (entry.basis === "provider_measured_constants" && entry.evidence) {
      carriedProviderEvidence.set(cellKey(entry.modelId, entry.backend, entry.tier), entry.evidence);
    }
  }

  // 4. Every remaining catalog cell is classified explicitly. There is no unclassified state.
  const analyticOnly = [];
  for (const cell of cells) {
    const key = cellKey(cell.modelId, cell.backend, cell.tier);
    if (anchoredCells.has(key)) continue;
    const envelope = envelopeEvidence(corpora, cell);
    const provider =
      cell.backend === "mlx"
        ? (providerConstants.get(cell.modelId) ?? carriedProviderEvidence.get(key) ?? null)
        : null;
    const declared = manifestTierEvidence(manifest, MANIFEST_PATH, manifestSha256, cell);
    const [basis, evidence] =
      envelope !== null
        ? ["measured_envelope", envelope]
        : provider !== null
          ? ["provider_measured_constants", provider]
          : declared !== null
            ? ["manifest_tier_declaration", declared]
            : ["no_retained_evidence", null];
    // An overlay render's envelope is evidence about a DIFFERENT resident set; say that in the row
    // rather than letting it read as a missing phase decomposition.
    const reason =
      basis === "measured_envelope" && evidence?.values?.overlay
        ? `the only retained render for this cell ran under the '${evidence.values.overlay}' ` +
          `overlay on provider '${evidence.values.provider ?? "unknown"}', which measures a ` +
          "different resident set, so it bounds the envelope without anchoring this cell"
        : REASONS[basis];
    analyticOnly.push({
      id: analyticId(cell),
      modelId: cell.modelId,
      modelFamily: cell.modelFamily,
      route: cell.route,
      backend: cell.backend,
      tier: cell.tier,
      basis,
      reason,
      evidence,
    });
  }
  analyticOnly.sort((left, right) => compareText(left.id, right.id));

  return { schemaVersion: MEMORY_ANCHOR_SCHEMA_VERSION, anchors, analyticOnly };
}

export const serialiseStore = (store) => `${JSON.stringify(store, null, 2)}\n`;

async function readExistingStore(root) {
  try {
    return JSON.parse(await readFile(path.join(root, STORE_PATH), "utf8"));
  } catch {
    return null;
  }
}

async function main() {
  const args = process.argv.slice(2);
  const rootFlag = args.indexOf("--inference-root");
  const inferenceRoot = rootFlag === -1 ? undefined : (args[rootFlag + 1] ?? null);
  const store = await buildAnchorStore({
    existingStore: await readExistingStore(ROOT),
    inferenceRoot,
  });
  const serialised = serialiseStore(store);
  const target = path.join(ROOT, STORE_PATH);
  if (args.includes("--check")) {
    const existing = await readFile(target, "utf8");
    if (existing !== serialised) {
      process.stderr.write(`${STORE_PATH} is stale — re-run scripts/extract-memory-anchors.mjs\n`);
      process.exitCode = 1;
      return;
    }
    process.stdout.write(`${STORE_PATH} is current\n`);
    return;
  }
  // Never redirect a generator into its own checked-in output: write beside it and rename, so an
  // aborted run cannot leave a truncated store behind.
  const temporary = `${target}.tmp`;
  await writeFile(temporary, serialised);
  await rename(temporary, target);
  process.stdout.write(
    `wrote ${STORE_PATH}: ${store.anchors.length} anchors, ${store.analyticOnly.length} analytic-only cells\n`,
  );
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  await main();
}
