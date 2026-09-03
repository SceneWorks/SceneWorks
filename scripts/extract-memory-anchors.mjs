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
 * * `anchors` — a cell whose retained corpus contains a render with the three phase peaks (named
 *   per backend lane: see `PHASE_MEASUREMENTS`) and an overall allocator envelope, in a
 *   composition the lane's derivation law can actually price (`isDerivable`), from a corpus the
 *   Rust loader compiles in. Byte-exact source provenance is recorded and re-checked by
 *   `memory_anchor.rs` against the compiled-in evidence, so the store cannot drift from the corpus
 *   it cites.
 * * `analyticOnly` — a cell with no such render. It is NOT a gap: the row names WHY (`basis`) and
 *   carries whatever weaker evidence does exist, so "nobody measured this" is distinguishable from
 *   "this was measured and the record was dropped".
 *
 * Sources walked (all read-only, all committed to THIS repo):
 *
 *   1. every `{records: [...]}` corpus under `docs/calibration/` and `docs/generated/` — the
 *      committed evidence bundle, the per-story receipt seeds and the geometry sweeps;
 *   2. `docs/generated/video-memory-curves.json` — the fitted curves' `evidence.sources` name the
 *      record corpora the video lane's identities were measured from, which is how the LTX-2.3
 *      sweep is reached (its paths are asserted to be part of the discovered corpus set);
 *   3. `config/manifests/builtin.models.jsonc` — `measured: true` `vramGbByTier` /
 *      `sequentialPeakGb` tier tables, a per-tier declared envelope but not a phase decomposition.
 *
 * A fourth source — the pinned inference revision's per-provider
 * `mlx-gen-<family>/src/memory_strategy.rs` measured byte constants — is OPT-IN and off by default,
 * because it lives outside this repo. A default run reads no cargo checkout at all, so the output
 * is a function of the committed tree and nothing else: this Mac, a CI runner and a fresh clone all
 * emit the same bytes. `--inference-root <dir>` reads one explicitly, and `--inference-root auto`
 * locates the cargo checkout OF THE PIN (never a working tree, never a different revision). Neither
 * is used to produce the checked-in store, and a run without them withdraws any row that only an
 * opt-in run could have written, rather than carrying a host-local value forward invisibly.
 *
 * The catalog population is NOT re-invented here: it comes from `buildMatrix()` in
 * `scripts/generate-memory-matrix.mjs`, the same resolution the memory matrix publishes, so a
 * catalog change moves both artifacts or neither.
 *
 * DETERMINISM is a contract, not an accident: every collection is sorted by an explicit code-unit
 * comparison, every selection rule is total, and re-running over the same corpora reproduces the
 * file byte-identically (`scripts/extract-memory-anchors.test.mjs`).
 *
 * The store is a PURE FUNCTION of the committed evidence, with EXACTLY ONE named exception: each
 * anchor's `source.loaderClosureDigest` currency key is CARRIED FORWARD from the checked-in
 * store's own previous content rather than re-derived (see `loaderClosureDigestFor` — the key
 * records what the model's loader hashed to AT THE MEASUREMENT, which no committed input of this
 * script can reproduce, and a newly extracted anchor fails loudly rather than borrowing the
 * pin's digest). Everything else is re-derived from the committed evidence on every run.
 * There is deliberately no carry-forward union: a row a regeneration does not re-derive has no
 * evidence behind it in this tree, and preserving it would make a withdrawn anchor immortal and
 * make `--check` unable to see the difference. A concurrently landed anchor (sc-22509's candle
 * proving models) survives a re-run the only way an anchor can be trusted to: its corpus is
 * committed under a walked root AND named in `PACKAGED_MEMORY_ANCHOR_SOURCES`, so the re-run
 * derives it again. `assertPackagedSources` enforces the other half of that contract — every
 * emitted anchor cites a file compiled into that list, so a store this script writes is always one
 * the Rust loader accepts.
 *
 * EVERY RETAINED CORPUS IS PACKAGED (sc-22666, epic 22657 E5). Anchor candidacy is still
 * restricted to corpora named in `PACKAGED_MEMORY_ANCHOR_SOURCES` — the Rust loader hard-rejects
 * an anchor citing a file it does not compile in, so a row from an unpackaged corpus would be
 * unloadable — but that list is no longer an opt-in a story may defer. Packaging used to be
 * per-model because the image lanes priced a cell with per-pixel slopes fitted on ONE model's
 * empirics (Krea Turbo), so anchoring a new model's cell the day its corpus landed would have
 * repriced its admission with borrowed slopes. Since sc-22663 there is one image law and it fits
 * nothing: it decomposes the anchor's OWN measured peaks against the contract's component bytes
 * and rescales the residues by architecture facts, so no slope exists to borrow. A retained
 * corpus that could anchor a catalog cell but is missing from the list is therefore a DEFECT, and
 * `assertEveryDerivableCorpusIsPackaged` fails the run rather than skipping it silently.
 *
 * Usage: node scripts/extract-memory-anchors.mjs [--check] [--inference-root <dir>|auto]
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

/**
 * The Rust loader's compiled-in evidence list. An anchor may only cite a file named here — the
 * loader hard-rejects any other path — so the generator reads the SAME list rather than trusting
 * the two halves to stay aligned by convention.
 */
export const PACKAGED_SOURCES_PATH =
  "crates/sceneworks-core/src/memory_anchor.rs";

/**
 * The per-model loader closures (sc-22511). Not read by this generator — an anchor's currency key
 * is carried forward from the store's own previous content (see `loaderClosureDigestFor`) — but
 * named here because it is the other half of the pair and the file the key is compared against.
 */
export const ANCHOR_LOADER_CLOSURES_PATH = "config/anchor-loader-closures.json";

/**
 * The three phase peaks an anchor is made of, keyed by BACKEND LANE. Absent any one of them, the
 * record cannot anchor. The two lanes name the same three peaks differently — the MLX adapter
 * reports unified ACTIVE peaks, the candle adapter reports discrete-device peak DELTAS — and the
 * names are chosen by the record's own `backend` rather than by probing for whichever is present,
 * exactly as `memory_anchor::validate_anchor` chooses them. A record captured on one lane therefore
 * cannot satisfy an anchor declared on the other, in the generator and in the loader alike.
 */
const PHASE_MEASUREMENTS = {
  mlx: {
    conditioning: "conditioningActivePeak",
    denoise: "denoiseActivePeak",
    decode: "decodeActivePeak",
  },
  candle: {
    conditioning: "conditioningDevicePeakDelta",
    denoise: "denoiseDevicePeakDelta",
    decode: "decodeDevicePeakDelta",
  },
};

/**
 * Analytic-only bases, strongest evidence first. The order IS the precedence: a cell is classified
 * by the best evidence that exists for it, and `no_retained_evidence` is the honest floor rather
 * than a silent gap.
 */
export const ANALYTIC_BASES = [
  "measured_envelope",
  "provider_measured_constants",
  "contract_estimate",
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
      const absolute = path.join(
        entry.parentPath ?? entry.path ?? base,
        entry.name,
      );
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
    if (!parsed || typeof parsed !== "object" || !Array.isArray(parsed.records))
      continue;
    corpora.push({
      path: relative,
      sha256: sha256(body),
      records: parsed.records,
    });
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

export const cellKey = (modelId, backend, tier) =>
  `${modelId}:${backend}:${tier}`;

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
  for (const [phase, name] of Object.entries(PHASE_MEASUREMENTS[backend])) {
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
  if (
    !Number.isInteger(width) ||
    !Number.isInteger(height) ||
    !Number.isInteger(frames)
  ) {
    return null;
  }
  if (width <= 0 || height <= 0 || frames <= 0) return null;
  const loadShape = record.loadShape;
  if (
    loadShape !== "eager_materialization" &&
    loadShape !== "deferred_materialization"
  ) {
    return null;
  }
  const engaged = Array.isArray(record.strategy?.engagedRungs)
    ? record.strategy.engagedRungs
    : [];
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
      typeof target.transformerVariant === "string"
        ? target.transformerVariant
        : null,
    decoder: typeof target.decoder === "string" ? target.decoder : null,
    overlay:
      target.overlay === "none" || target.overlay === undefined
        ? null
        : target.overlay,
    referenceCount: Number.isInteger(target.referenceCount)
      ? target.referenceCount
      : 0,
    loadShape,
    // All FOUR declared rungs, not just the two the video law reads: `staged` decides whether the
    // text encoder was still co-resident through conditioning and `attentionChunked` decides the
    // denoise intercept, both of which the candle image law (sc-22509) reads.
    // `AnchorMeasuredRegime` is `deny_unknown_fields` with four non-optional members, so a store
    // emitting fewer would not load at all.
    measuredRegime: {
      decodeTiled: engaged.includes("bounded_decode"),
      transformerWindowed: engaged.includes("bounded_transformer_residency"),
      staged: engaged.includes("staged_residency"),
      attentionChunked: engaged.includes("bounded_attention"),
    },
    calibrationFingerprint:
      typeof record.calibrationFingerprint === "string"
        ? record.calibrationFingerprint
        : null,
    geometry: {
      width,
      height,
      frames,
      fps: Number.isInteger(fps) && fps > 0 ? fps : null,
    },
    phaseActivePeakBytes: phases,
    // Per-phase ALLOCATOR levels (epic 22505 feature-end fix round): the quantity the MLX image
    // law derives. Bound in both directions by `memory_anchor::validate_anchor`, so it is emitted
    // exactly when the record reports all three.
    phaseAllocatorEnvelopeBytes: phaseAllocatorEnvelopes(record),
    overallAllocatorEnvelopeBytes: envelope,
    sourcePath: corpus.path,
    sourceSha256: corpus.sha256,
    recordId,
  };
}

/** The record's per-phase allocator levels, or `null` when any phase omits one. */
export function phaseAllocatorEnvelopes(record) {
  const observed = record?.observedMemory ?? {};
  const at = (phase) => {
    const bytes = observed?.[phase]?.allocatorBytes;
    return Number.isInteger(bytes) && bytes > 0 ? bytes : null;
  };
  const conditioning = at("conditioning");
  const denoise = at("denoise");
  const decode = at("decode");
  if (conditioning === null || denoise === null || decode === null) return null;
  return { conditioning, denoise, decode };
}

/**
 * Whether the lane's derivation law can price ANY request from a record in this composition.
 *
 * This mirrors the anchor-local guards in `crates/sceneworks-core/src/memory_anchor.rs`, and it is
 * deliberately only the anchor-local half. Guards that compare the anchor to the REQUEST being
 * graded (the video law's "an anchor measured in a bounded regime refuses the unbounded request")
 * are not derivability questions: such an anchor still prices every correspondingly-bounded
 * request, so it is a usable row. What is filtered here is a record no request at all could be
 * priced from — an anchor the law rejects outright, which would be an unreachable store row.
 *
 * * `mlx` mirrors `MemoryAnchor::derive_video_phase_peaks`, which rejects no composition
 *   outright: its regime guards are all anchor-vs-request, so a record in ANY composition prices
 *   the correspondingly-bounded requests. Its one anchor-local guard — pipeline axes must be
 *   stated — is deliberately NOT a filter here: since sc-22510 an axis-free row is how a cell
 *   measured without pipeline axes is CLASSIFIED, and withdrawing those rows would narrow the
 *   catalog coverage that story established rather than fix anything.
 * * `candle` mirrors `MemoryAnchor::derive_image_phase_peaks`: that law prices exactly the SHALLOW
 *   optimized composition — `staged_residency` engaged and nothing deeper — on a still, with no
 *   pipeline axes. Every deeper rung exists to make a phase smaller, so the shallow anchor upper
 *   bounds them; a resident composition holds the text encoder through denoise and decode and is
 *   strictly LARGER, the direction the anchor cannot cover, so the law refuses it. Anchoring the
 *   cell from a resident render would therefore emit a row the law rejects on every lookup.
 *
 * An unknown lane has no law to mirror and gets no opinion.
 */
export function isDerivable(candidate) {
  const regime = candidate.measuredRegime;
  if (candidate.backend === "mlx") {
    return true;
  }
  if (candidate.backend === "candle") {
    return (
      candidate.transformerVariant === null &&
      candidate.decoder === null &&
      candidate.geometry?.frames === 1 &&
      regime?.staged === true &&
      regime?.decodeTiled === false &&
      regime?.attentionChunked === false &&
      regime?.transformerWindowed === false
    );
  }
  return true;
}

/**
 * The representative render for one identity cell, chosen mechanically: DERIVABILITY first (a
 * record whose composition the lane's law can actually price outranks one it cannot, however much
 * larger the latter's envelope is — see [`isDerivable`]), then the LARGEST measured allocator
 * envelope, tie-broken by (source path, record id). The largest envelope is the most binding
 * retained observation of that cell, and every tie-break term is a stable string, so the choice
 * cannot move with corpus iteration order.
 *
 * Ordering the two terms the other way would let a cell be anchored by a render the law rejects,
 * which reads as coverage and admits nothing: krea's resident-only capture is the largest envelope
 * its corpus retains AND the one composition `derive_image_phase_peaks` refuses.
 */
export function selectRepresentative(candidates) {
  return candidates.reduce((best, candidate) => {
    if (best === null) return candidate;
    const candidateDerivable = isDerivable(candidate);
    if (candidateDerivable !== isDerivable(best))
      return candidateDerivable ? candidate : best;
    if (
      candidate.overallAllocatorEnvelopeBytes !==
      best.overallAllocatorEnvelopeBytes
    ) {
      return candidate.overallAllocatorEnvelopeBytes >
        best.overallAllocatorEnvelopeBytes
        ? candidate
        : best;
    }
    const byPath = compareText(candidate.sourcePath, best.sourcePath);
    if (byPath !== 0) return byPath < 0 ? candidate : best;
    return compareText(candidate.recordId, best.recordId) < 0
      ? candidate
      : best;
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

/**
 * The anchor's CURRENCY KEY — CARRIED FORWARD from the checked-in store, deliberately.
 *
 * This is the one field the generator does not re-derive, and the reason is the whole point of the
 * key (sc-22511, E9). `source.loaderClosureDigest` records what the model's loader looked like AT
 * THE MEASUREMENT. Currency is that recorded value compared against the digest at the CURRENT pin,
 * so the recorded half has to stay frozen at the measurement's own revision. Both other ways of
 * filling it are wrong, in opposite directions:
 *
 *   * re-deriving it at the PIN would make every anchor eternally current — a value compared with
 *     itself, reporting "fine" straight through any loader change, and a pin bump would silently
 *     rewrite the evidence's own provenance;
 *   * re-deriving it at the MEASUREMENT revision here would need a clone carrying every historical
 *     revision the store cites, making this generator's output depend on the host — the exact
 *     host-dependence `buildAnchorStore` exists to avoid (see `inferenceRoot`).
 *
 * So it is derived ONCE, out of band, by `node scripts/anchor-loader-closure.mjs --repo <clone>
 * --stamp-anchors`, which digests each anchor at its own record's revision; from then on this
 * generator carries it. A pin bump therefore does NOT make this file stale — the anchors simply
 * stop being current, which is the designed outcome and not an error.
 *
 * A NEW anchor (evidence this store has never carried) has nothing to carry forward and FAILS
 * loudly. Borrowing the pin's digest there would mark a freshly extracted anchor current against a
 * loader it was never measured against — the false green this key exists to prevent.
 */
export function loaderClosureDigestFor(previousStore, anchorId) {
  const recorded = (previousStore?.anchors ?? []).find((anchor) => anchor.id === anchorId)?.source
    ?.loaderClosureDigest;
  if (typeof recorded !== "string" || !/^[0-9a-f]{64}$/.test(recorded)) {
    throw new Error(
      `anchor ${anchorId} has no recorded loader-closure digest in ${STORE_PATH}. A newly ` +
        "extracted anchor must have its currency key derived at ITS OWN measurement revision: " +
        "node scripts/anchor-loader-closure.mjs --repo <inference clone> --stamp-anchors",
    );
  }
  return recorded;
}

/** Serialise one candidate into the store's anchor shape (field order is the file's field order). */
function anchorRow(candidate, catalogCell, previousStore, underivedReason) {
  const id = anchorId(candidate);
  const loaderClosureDigest = loaderClosureDigestFor(previousStore, id);
  return {
    id,
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
      staged: candidate.measuredRegime.staged,
      attentionChunked: candidate.measuredRegime.attentionChunked,
    },
    source: {
      path: candidate.sourcePath,
      sha256: candidate.sourceSha256,
      recordId: candidate.recordId,
      calibrationFingerprint: candidate.calibrationFingerprint,
      loaderClosureDigest,
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
    // Emitted exactly when the record reports all three phase allocator levels — the Rust loader
    // binds this field in both directions (epic 22505 feature-end fix round).
    ...(candidate.phaseAllocatorEnvelopeBytes === null
      ? {}
      : {
          phaseAllocatorEnvelopeBytes: {
            conditioning: candidate.phaseAllocatorEnvelopeBytes.conditioning,
            denoise: candidate.phaseAllocatorEnvelopeBytes.denoise,
            decode: candidate.phaseAllocatorEnvelopeBytes.decode,
          },
        }),
    overallAllocatorEnvelopeBytes: candidate.overallAllocatorEnvelopeBytes,
    // `Some` when the anchor validates its measured point but no lane law may derive from it,
    // with the stated per-model reason (epic 22505 feature-end fix round). Absent otherwise.
    ...(underivedReason === null ? {} : { underivedReason }),
  };
}

/**
 * Why THIS anchor validates its measured point but may not price anything (epic 22505 feature-end
 * fix round), or `null` when the lane's law derives from it. Per-MODEL and computed from the
 * model's own retained evidence — never a blanket switch:
 *
 * * an MLX still-image anchor derives only when it is the eager unbounded resident composition
 *   (the widest, which is what lets one law upper-bound the whole ladder);
 * * an MLX VIDEO anchor that states no pipeline axes cannot be priced by the video law, whose
 *   per-token coefficients are keyed on `(transformer variant, decoder)`.
 *
 * THE BORROWED-SLOPE REFUSAL IS GONE (sc-22666, epic 22657 E5). Until sc-22663 an MLX image anchor
 * also had to come from a model whose own retained records varied geometry WITHIN a cell, because
 * the image lane priced a cell with per-pixel slopes fitted on one model's spread and a model with
 * no spread of its own would have been priced with another's. The image law fits nothing now — it
 * decomposes the anchor's own measured peaks against the contract's component bytes and rescales
 * the residues by architecture facts and geometry — so there is no slope to borrow and no reason
 * to withhold derivation from a single-geometry anchor. The regime guard below is unaffected: it
 * is about WHICH composition was measured, not about fitting anything.
 *
 * Candle anchors take no reason here: `isDerivable` already refuses to ANCHOR a candle cell from
 * a composition the candle law rejects, so every candle anchor that exists is derivable.
 */
export function underivedReasonFor(candidate) {
  if (candidate.backend !== "mlx") return null;
  if (candidate.geometry.frames === 1) {
    const regime = candidate.measuredRegime;
    const unboundedEager =
      candidate.loadShape === "eager_materialization" &&
      !regime.staged &&
      !regime.decodeTiled &&
      !regime.attentionChunked &&
      !regime.transformerWindowed;
    if (!unboundedEager) {
      return (
        "the anchor render engaged bounded rungs (or deferred materialization), so it does not " +
        "measure the widest composition the MLX image law upper-bounds the ladder from; this " +
        "anchor validates its measured point and prices nothing beyond it"
      );
    }
    return null;
  }
  if (candidate.transformerVariant === null || candidate.decoder === null) {
    return (
      "the source record states no (transformer variant, decoder) pipeline axes and the video " +
      "law's per-token coefficients are keyed on them; this anchor validates its measured point " +
      "and prices nothing beyond it"
    );
  }
  return null;
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
      // Both fields are non-optional on the Rust side, under `deny_unknown_fields`. An undefined
      // value is dropped by `JSON.stringify` rather than written as null, so a catalog entry that
      // stops publishing one would fail on the CONSUMER as a serde "missing field" — name it here,
      // against the model it came from, instead.
      const modelFamily = model.family ?? model.familyGroup;
      const route = model.resolvedRoutes?.[backend] ?? model.resolvedRoute;
      for (const [field, value] of [
        ["family/familyGroup", modelFamily],
        [`resolvedRoutes.${backend}/resolvedRoute`, route],
      ]) {
        if (typeof value !== "string" || value.length === 0) {
          throw new Error(
            `${model.id} (${backend}): the routing catalog publishes ${JSON.stringify(value)} for ` +
              `${field}, but every anchor-store row is required to carry it as a non-empty string`,
          );
        }
      }
      for (const tier of axes.tiers) {
        cells.push({
          modelId: model.id,
          modelFamily,
          route,
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
  // Two independent races, not one: a CLEAN render of this cell always outranks an overlay render
  // of it, however much larger the overlay's envelope is, because the overlay measures a different
  // resident set. Only when every retained render for the cell carries an overlay does the overlay
  // race decide — which is also what makes the emitted `overlay` value mean "all of them", and lets
  // the row's reason say so truthfully.
  let best = null;
  let bestOverlay = null;
  for (const corpus of corpora) {
    for (const record of corpus.records) {
      if (record?.backend !== cell.backend) continue;
      if (
        record?.target?.modelId !== cell.modelId ||
        record?.target?.tier !== cell.tier
      )
        continue;
      const envelope = envelopeOf(record);
      if (envelope === null || typeof record.id !== "string") continue;
      // The overlay and provider of the cited render are carried, not elided: an envelope measured
      // under a control overlay is evidence ABOUT a different resident set, and a reader has to be
      // able to see that from the row.
      const overlay =
        typeof record.target?.overlay === "string" &&
        record.target.overlay !== "none"
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
      if (overlay === null) best = preferEnvelope(best, candidate);
      else bestOverlay = preferEnvelope(bestOverlay, candidate);
    }
  }
  return best ?? bestOverlay;
}

/** Larger envelope wins; ties broken by (source path, record id), both stable strings. */
function preferEnvelope(best, candidate) {
  if (best === null || candidate.envelopeBytes > best.envelopeBytes)
    return candidate;
  if (candidate.envelopeBytes < best.envelopeBytes) return best;
  const byPath = compareText(candidate.path, best.path);
  if (
    byPath < 0 ||
    (byPath === 0 && compareText(candidate.recordId, best.recordId) < 0)
  ) {
    return candidate;
  }
  return best;
}

/**
 * The published `memoryStrategyContract` for this cell's backend block (sc-22666, epic 22657 E5).
 *
 * This is the evidence behind the `contract_estimate` basis: a cell whose contract declares the
 * ladder's rungs is priced by the worker as a CONTRACT-ONLY per-rung estimate (the manifest row
 * rescaled by the image law's per-rung ratios, as sc-22664 wired), not as a bare manifest scalar
 * repeated across the ladder, so classifying it as `manifest_tier_declaration` would understate
 * where its evidence genuinely is.
 *
 * KEYED ON THE LADDER'S INPUTS (sc-22667, feature-end round): the worker builds that pseudo-anchor
 * (`candle_memory_strategy.rs`, `floor_anchor`) only when BOTH hold —
 *
 *   * the manifest declares the RAW staged row the law decomposes,
 *     `candle.sequentialPeakGb[tier]` (`vram_gate::measured_sequential_peak_gb`, with its `q8`
 *     fallback for an unmeasured `nvfp4` tier), and
 *   * the route is not RECEIPT-PRICED (`is_receipt_priced`): a receipt-priced family's floor is a
 *     structural weights-plus-headroom sum sealed from the provider receipt and is never rescaled.
 *
 * A cell missing either is NOT priced by the mechanism this reason names, whatever its contract
 * publishes, and falls through to `manifest_tier_declaration` / `no_retained_evidence`, which is
 * true of it. `RECEIPT_PRICED_ROUTES` mirrors the worker's list; the extractor test reads the
 * worker source so the two cannot drift.
 *
 * SCOPE, stated because it bounds the claim: at this pin the generator cannot resolve a contract's
 * own asset facts (they live behind the provider surface at the pinned inference revision), so the
 * remaining key is the PRESENCE of the model's `<backend>.memoryStrategyContract` block plus the
 * rungs it declares. The row carries the declared rung names verbatim so a reader can see exactly
 * what was published, and the digest is the manifest's.
 *
 * LANE RESTRICTION (sc-22666, fix round): the per-rung ladder this reason asserts is a CANDLE
 * mechanism. `candle_memory_strategy.rs`'s `floor_pseudo_anchor` is the code that rescales the
 * manifest row by the image law's per-rung ratios; `mlx_fit_gate.rs` has no pseudo-anchor and no
 * manifest-row-rescale path at all, so an mlx cell is not priced that way whatever its contract
 * publishes. Those cells fall through to `manifest_tier_declaration` / `no_retained_evidence`,
 * which is true of them. `CONTRACT_LADDER_BACKENDS` names the lanes that implement the ladder; add
 * a backend here only when that lane grows the mechanism.
 *
 * The row also carries the manifest figures the reason says the ladder RESCALES
 * (`manifestTierEvidence`'s `vramGbByTier` / `sequentialPeakGb`) whenever the manifest declares
 * them, so the row states the base it rescales rather than only the rungs it rescales it onto.
 */
export const CONTRACT_LADDER_BACKENDS = Object.freeze(["candle"]);

/**
 * Every candle route whose admitted peak is priced from an exact provider receipt rather than a
 * manifest estimate — the mirror of `candle_memory_strategy::is_receipt_priced` (and the `is_*`
 * helpers it calls), pinned here because no config file states the list. The extractor test
 * `RECEIPT_PRICED_ROUTES mirrors the worker's is_receipt_priced` parses the worker source and
 * fails when the two sets differ, in either direction.
 */
export const RECEIPT_PRICED_ROUTES = Object.freeze(
  [
    // is_chroma
    "chroma1_hd",
    "chroma1_base",
    "chroma1_flash",
    // is_ideogram
    "ideogram_4",
    "ideogram_4_turbo",
    // is_sana
    "sana_1600m",
    "sana_sprint_1600m",
    // is_sd35
    "sd3_5_large",
    "sd3_5_large_turbo",
    "sd3_5_medium",
    // engine_id == "kolors"
    "kolors",
    // is_sealed_kolors_bespoke
    "candle_kolors_ipadapter",
    "candle_kolors_control",
  ].sort(compareText),
);

export function isReceiptPricedRoute(route) {
  return RECEIPT_PRICED_ROUTES.includes(route);
}

/**
 * The RAW staged row the worker's contract-only ladder decomposes,
 * `candle.sequentialPeakGb[tier]`, read exactly as `vram_gate::measured_sequential_peak_gb` reads
 * it: the tier's own row, or the `q8` row for an `nvfp4` tier that has none. `null` when the
 * manifest declares no such row — then no pseudo-anchor is built for the cell.
 */
export function manifestSequentialRow(manifest, cell) {
  const model = manifest.models?.find((entry) => entry.id === cell.modelId);
  const rows = model?.[cell.backend]?.sequentialPeakGb;
  if (!rows || typeof rows !== "object") return null;
  const read = (tier) => {
    const declared = rows[tier];
    return typeof declared === "number" && Number.isFinite(declared)
      ? declared
      : null;
  };
  const own = read(cell.tier);
  if (own !== null) return own;
  return cell.tier === "nvfp4" ? read("q8") : null;
}

export function contractEstimateEvidence(
  manifest,
  manifestPath,
  manifestSha256,
  cell,
) {
  if (!CONTRACT_LADDER_BACKENDS.includes(cell.backend)) return null;
  const model = manifest.models?.find((entry) => entry.id === cell.modelId);
  const contract = model?.[cell.backend]?.memoryStrategyContract;
  if (!contract || typeof contract !== "object") return null;
  const rungs = [
    ...new Set(
      (Array.isArray(contract.implementations) ? contract.implementations : [])
        .map((item) => item?.rung)
        .filter((rung) => typeof rung === "string"),
    ),
  ].sort(compareText);
  if (rungs.length === 0) return null;
  // The ladder's inputs (sc-22667): the raw staged row it decomposes, on a route whose floor is
  // not a sealed receipt. Absent either, the worker never builds the pseudo-anchor.
  if (isReceiptPricedRoute(cell.route)) return null;
  const sequentialRow = manifestSequentialRow(manifest, cell);
  if (sequentialRow === null) return null;
  const values = {
    declaredRungs: rungs.join(","),
    sequentialPeakGb: String(sequentialRow),
  };
  if (typeof contract.provider === "string") values.provider = contract.provider;
  if (typeof contract.abi === "number") values.abi = String(contract.abi);
  // The manifest row the ladder rescales, when the manifest declares one. Without this the row
  // asserts a rescale of figures it does not carry.
  const declared = manifestTierEvidence(
    manifest,
    manifestPath,
    manifestSha256,
    cell,
  );
  if (declared !== null) Object.assign(values, declared.values);
  return {
    repo: null,
    revision: null,
    path: `${manifestPath}#models/${cell.modelId}/${cell.backend}/memoryStrategyContract`,
    sha256: manifestSha256,
    recordId: null,
    envelopeBytes: null,
    values,
  };
}

/** `measured: true` tier tables in the catalog manifest — a declared envelope, per tier. */
export function manifestTierEvidence(
  manifest,
  manifestPath,
  manifestSha256,
  cell,
) {
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
 * The repo-relative paths in `PACKAGED_MEMORY_ANCHOR_SOURCES`, parsed out of the Rust module that
 * declares it. This is the loader's whole domain for `anchor.source.path`: a store citing anything
 * else is rejected by `validate_anchor` as "not a compiled retained-evidence file", so producing
 * one would be writing an artifact the consumer cannot load.
 */
export function packagedAnchorSources(source) {
  const start = source.indexOf(
    "PACKAGED_MEMORY_ANCHOR_SOURCES: &[(&str, &str)] = &[",
  );
  const end = start === -1 ? -1 : source.indexOf("\n];", start);
  if (start === -1 || end === -1) {
    throw new Error(
      `${PACKAGED_SOURCES_PATH} no longer declares PACKAGED_MEMORY_ANCHOR_SOURCES in the shape this ` +
        "generator reads; the anchor store's source domain cannot be checked",
    );
  }
  const paths = [
    ...source.slice(start, end).matchAll(/\(\s*"([^"]+)",\s*include_str!/g),
  ].map((match) => match[1]);
  if (paths.length === 0) {
    throw new Error(
      `${PACKAGED_SOURCES_PATH} declares an empty PACKAGED_MEMORY_ANCHOR_SOURCES`,
    );
  }
  return new Set(paths);
}

/**
 * Every anchor must cite a compiled-in corpus. The failure names the offending row and what to do,
 * rather than surfacing two lanes later as a Rust load error on a committed artifact.
 */
export function assertPackagedSources(anchors, packaged) {
  const foreign = anchors
    .filter((anchor) => !packaged.has(anchor.source?.path))
    .map((anchor) => `${anchor.id} -> ${anchor.source?.path}`)
    .sort(compareText);
  if (foreign.length > 0) {
    throw new Error(
      `anchors cite evidence that is not compiled into PACKAGED_MEMORY_ANCHOR_SOURCES ` +
        `(${PACKAGED_SOURCES_PATH}), so the store would not load: ${foreign.join(", ")}`,
    );
  }
}

/**
 * THE PACKAGING-LAPSE GUARD (sc-22666, epic 22657 E5), the converse of `assertPackagedSources`.
 *
 * That guard asks "does every emitted anchor cite a compiled-in corpus"; this one asks the
 * question the old opt-in let go unasked: "does every retained corpus that COULD anchor a catalog
 * cell get compiled in". A corpus that clears every candidacy rule the anchor pass applies — it
 * parses, its records carry the lane's three phase measurements, they are overlay-free, they
 * resolve to a routing-catalog cell and their composition is one the lane's law derives from — but
 * is absent from `PACKAGED_MEMORY_ANCHOR_SOURCES` is now a defect: the store would classify a cell
 * as analytic-only while the evidence to anchor it sits committed in the tree.
 *
 * The candidacy rules are applied through the same helpers the anchor pass uses, so this cannot
 * drift into a second, laxer opinion about what "derivable" means.
 */
export function assertEveryDerivableCorpusIsPackaged(
  corpora,
  packaged,
  catalogByCell,
) {
  const lapsed = [];
  for (const corpus of corpora) {
    if (packaged.has(corpus.path)) continue;
    const cells = new Set();
    for (const record of corpus.records) {
      const candidate = anchorCandidate(record, corpus);
      if (candidate === null || candidate.overlay !== null) continue;
      if (!isDerivable(candidate)) continue;
      const key = cellKey(
        candidate.modelId,
        candidate.backend,
        candidate.tier,
      );
      if (catalogByCell.has(key)) cells.add(key);
    }
    if (cells.size > 0) {
      lapsed.push(`${corpus.path} -> ${[...cells].sort(compareText).join(", ")}`);
    }
  }
  if (lapsed.length > 0) {
    throw new Error(
      "retained corpora carry derivable anchors for catalog cells but are not compiled into " +
        `PACKAGED_MEMORY_ANCHOR_SOURCES (${PACKAGED_SOURCES_PATH}), so those cells would be ` +
        "classified analytic-only while the evidence to anchor them is committed. Packaging is " +
        `no longer an opt-in (epic 22657 E5) — add them: ${lapsed.sort(compareText).join("; ")}`,
    );
  }
}

/**
 * Top-level `pub const <NAME>_BYTES: u64 = <n>;` measured constants in one provider source. Only
 * column-zero declarations count: the same spelling nested inside a function or a `#[cfg(test)]`
 * module is a fixture, not a provider fact.
 */
export function providerByteConstants(source) {
  const values = {};
  for (const match of source.matchAll(
    /^pub const ([A-Z0-9_]*BYTES): u64 = ([0-9_]+);/gm,
  )) {
    values[match[1]] = String(
      Number.parseInt(match[2].replaceAll("_", ""), 10),
    );
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
      revisions = await readdir(path.join(base, entry.name), {
        withFileTypes: true,
      });
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
    .sort(
      (left, right) =>
        right.key.length - left.key.length || compareText(left.key, right.key),
    );
}

/**
 * Measured byte constants published by the pinned MLX providers, keyed by catalog model id. The
 * provider crate is matched by LONGEST normalised-name prefix of the id (`flux2_dev` ->
 * `mlx-gen-flux2`, `z_image_turbo` -> `mlx-gen-z-image`), which is how the routing catalog's model
 * ids relate to the engine crates.
 */
export async function inferenceProviderConstants(
  checkoutRoot,
  revision,
  modelIds,
) {
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

const analyticId = (cell) =>
  `analytic:${cellKey(cell.modelId, cell.backend, cell.tier)}`;

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Variant/decoder component deltas (epic 22505 E2, feature-end fix round).
//
// An unmeasured (variant, decoder) cell of an anchored (model, tier, lane) derives from a sibling
// anchor plus deltas computed from SHIPPED FILE INVENTORIES: the variant's adapter/refiner file
// sizes and the decoder's weight file sizes. The sizes come from the committed weights file
// inventory (per-file sizes of the pinned artifact revision); the Rust loader recomputes every
// row's bytes from that inventory by digest, so nothing here is a trusted literal.
//
// Direction is conservative by construction: a row prices only the component the TARGET cell
// materializes and a sibling might lack (the DEV variant's distillation LoRA, the target decoder's
// weight files, at the widest variant where the file exists per variant). Nothing is ever
// subtracted for sibling-only components, and a variant that crosses nothing gets an explicit ZERO
// row rather than no row — see the matching section comment in
// `crates/sceneworks-core/src/memory_anchor.rs`.
// ─────────────────────────────────────────────────────────────────────────────────────────────

export const LTX25_WEIGHTS_INVENTORY_PATH = "config/ltx25-weights-file-inventory.json";

const LTX25_VARIANTS = ["dev", "distilled"];

/**
 * What each LTX-2.5 transformer variant MATERIALIZES beyond the components its sibling also loads,
 * mirrored from the engine rather than from the variant's name:
 *
 *   * `crates/sceneworks-memory-adapter/src/bin/mlx_ltx25.rs::configured_spec` pushes `DEV_ADAPTER`
 *     (the distillation LoRA) onto `spec.adapters` under `variant == TransformerVariant::Dev`, and
 *     inserts `enhancer` into `spec.components` for BOTH variants.
 *   * `crates/sceneworks-worker/src/video_jobs/ltx.rs::resolve_ltx_distill_adapter` returns `None`
 *     for `Ltx25TransformerVariant::Distilled` — the distilled checkpoint already contains the
 *     refinement — and resolves the manifest's `distilledLora` co-requisite for dev.
 *
 * So dev crosses one component and distilled crosses none. Component names are engine-side keys,
 * not inventory paths; `ltx25ComponentDeltas` resolves each to the shipped file. Held as data so
 * the Rust cross-check has one thing to compare against instead of a shape hidden in a loop.
 */
export const LTX25_VARIANT_COMPONENTS = [
  { to: "dev", components: ["distilled_lora"] },
  { to: "distilled", components: [] },
];

/** The single inventory file under `prefix` matching `suffix`, or a loud failure. */
function inventoryFile(files, prefix, suffix) {
  const matches = Object.keys(files)
    .filter((file) => file.startsWith(prefix) && file.endsWith(suffix))
    .sort(compareText);
  if (matches.length !== 1) {
    throw new Error(
      `weights inventory names ${matches.length} files under ${prefix}*${suffix}; expected exactly one`,
    );
  }
  return matches[0];
}

/** The largest-summing variant's file list for one decoder's per-tier weight files. */
function widestDecoderFiles(files, tier, decoderFiles) {
  let best = null;
  for (const variant of LTX25_VARIANTS) {
    const candidate = decoderFiles.map((name) => `${variant}/${tier}/${name}`);
    const bytes = candidate.reduce((total, file) => {
      const size = files[file];
      if (!Number.isInteger(size) || size <= 0) {
        throw new Error(`weights inventory does not size ${file}`);
      }
      return total + size;
    }, 0);
    if (best === null || bytes > best.bytes) best = { files: candidate, bytes };
  }
  return best;
}

/**
 * The LTX-2.5 MLX component-delta rows for the catalog's tiers, priced from the committed weights
 * inventory. Per tier and per axis:
 *
 *   * `transformer_variant -> dev`: the distillation LoRA. The ENGINE is the authority here, not
 *     the recipe's name: `mlx_ltx25.rs` pushes `DEV_ADAPTER`
 *     (`distilled_lora/ltx-2.5-22b-distilled-lora-450-bf16.safetensors`) onto the load spec only
 *     when `variant == Dev`, and `video_jobs/ltx.rs::resolve_ltx_distill_adapter` returns `None`
 *     for LTX-2.5 Distilled because the distilled checkpoint already contains the refinement. So
 *     DEV is the variant that materializes the extra file, and dev is the row that prices it.
 *   * `transformer_variant -> distilled`: NOTHING is crossed. The distilled recipe materializes no
 *     component the dev sibling lacks, so the row is a legitimate ZERO — it exists so the
 *     fall-through can cross the variant axis toward distilled at all (a missing row means
 *     "unpriced axis, refuse", which is a different claim from "priced at zero"). The stock
 *     `enhancer/` directory is deliberately NOT here: `configured_spec` inserts it into
 *     `spec.components` unconditionally for BOTH variants, so it is resident in either measured
 *     envelope and cancels out of every variant delta. Pricing it into the dev row (as the first
 *     cut did) both mis-attributed it and, mirrored, under-estimated dev-from-distilled by the
 *     8.9 GB LoRA — the OOM direction.
 *   * `decoder -> conv` / `decoder -> diffvae`: the target decoder's per-tier weight files, taken
 *     at the WIDER of the two variant subdirectories so one row upper-bounds both variants.
 *
 * `LTX25_VARIANT_COMPONENTS` is the mirror of the engine's materialization the rows are keyed on;
 * `ltx25_variant_delta_matches_the_engine_materialization` in `memory_anchor.rs` cross-checks it
 * against the engine constants, and re-inverting the mapping reds that test.
 */
export function ltx25ComponentDeltas(inventoryBody, tiers) {
  const inventory = JSON.parse(inventoryBody);
  const files = inventory.files ?? {};
  const inventorySha256 = sha256(inventoryBody);
  const source = { path: LTX25_WEIGHTS_INVENTORY_PATH, sha256: inventorySha256 };
  const rows = [];
  // A component name IS its top-level directory in the bundle, which is what the Rust cross-check
  // compares against; resolving it here (rather than hard-coding the file) keeps the mapping
  // single-sourced in `LTX25_VARIANT_COMPONENTS` and fails loudly if a named component stops
  // shipping exactly one file.
  const variantRows = LTX25_VARIANT_COMPONENTS.map(({ to, components }) => ({
    to,
    files: components.map((component) =>
      inventoryFile(files, `${component}/`, ".safetensors"),
    ),
  }));
  for (const tier of [...tiers].sort(compareText)) {
    for (const { to, files: components } of variantRows) {
      const bytes = components.reduce((total, file) => total + files[file], 0);
      rows.push({
        id: `ltx_2_5:mlx:${tier}:transformer_variant:${to}`,
        modelId: "ltx_2_5",
        backend: "mlx",
        tier,
        axis: "transformer_variant",
        to,
        bytes,
        files: components,
        source,
      });
    }
    const decoderRows = [
      { to: "conv", names: ["vae_decoder.safetensors"] },
      {
        to: "diffvae",
        names: ["diffusion_vae_encoder.safetensors", "vae_diffusion_decoder.safetensors"],
      },
    ];
    for (const { to, names } of decoderRows) {
      const widest = widestDecoderFiles(files, tier, names);
      rows.push({
        id: `ltx_2_5:mlx:${tier}:decoder:${to}`,
        modelId: "ltx_2_5",
        backend: "mlx",
        tier,
        axis: "decoder",
        to,
        bytes: widest.bytes,
        files: widest.files,
        source,
      });
    }
  }
  return rows.sort((left, right) => compareText(left.id, right.id));
}

const REASONS = {
  measured_envelope:
    "the retained corpus measures this cell's overall allocator envelope but carries no per-phase " +
    "decomposition, so it bounds admission without anchoring the derivation",
  provider_measured_constants:
    "no retained render for this cell; the pinned MLX provider publishes measured component/stage " +
    "byte constants, which price components rather than a render peak",
  contract_estimate:
    "no retained render for this cell; the model's backend block publishes a memoryStrategyContract " +
    "and declares the staged row (candle.sequentialPeakGb) on a route that is not receipt-priced, " +
    "so the cell's estimate is the CONTRACT-ONLY per-rung ladder (that row, anchored at the " +
    "manifest's measured geometry and rescaled by the image law's per-rung ratios) rather than one " +
    "manifest scalar repeated across every rung — this generator reads the contract's presence and " +
    "the row in the manifest, not the contract's own asset facts, which only the worker resolves at " +
    "admission",
  manifest_tier_declaration:
    "no retained render for this cell; the catalog manifest declares a measured per-tier envelope, " +
    "which is a whole-render figure with no phase decomposition",
  no_retained_evidence:
    "no retained render, provider constant or measured manifest declaration covers this cell; its " +
    "peak is derived analytically from architecture facts alone",
};

/**
 * Build the complete store from the committed evidence. The previous output is NOT an input: there
 * is no carry-forward, so `--check` and a regeneration ask the same question.
 *
 * `inferenceRoot` is the one optional input and it defaults to OFF (`null`): `"auto"` locates the
 * cargo checkout of the pin on this host, a path reads that directory, and `null` reads no checkout
 * at all, which is what the checked-in store is generated with.
 */
export async function buildAnchorStore({
  root = ROOT,
  matrix = null,
  inferenceRoot = null,
} = {}) {
  const resolvedMatrix = matrix ?? (await buildMatrix());
  const cells = await catalogCells(resolvedMatrix);
  const corpora = await loadCorpora(root);
  const manifestBody = await readFile(path.join(root, MANIFEST_PATH), "utf8");
  const manifest = JSON.parse(stripJsoncComments(manifestBody));
  const manifestSha256 = sha256(manifestBody);
  const pin = inferencePin(await readFile(path.join(root, PIN_PATH), "utf8"));
  // The store's own previous content, read for ONE field: each anchor's frozen currency key. See
  // `loaderClosureDigestFor` for why that field is carried rather than re-derived.
  const previousStore = JSON.parse(await readFile(path.join(root, STORE_PATH), "utf8"));

  // The fitted video curves name the corpora the video lane's identities were measured from. They
  // are not a separate reader: the assertion is that corpus discovery already reached them, so a
  // renamed or dropped sweep fails here instead of silently narrowing the anchor population.
  const curves = JSON.parse(
    await readFile(path.join(root, VIDEO_CURVES_PATH), "utf8"),
  );
  const curveSources = new Set();
  for (const curve of curves.curves ?? []) {
    for (const source of curve.evidence?.sources ?? []) {
      if (typeof source?.path === "string") curveSources.add(source.path);
    }
  }
  const discovered = new Set(corpora.map((corpus) => corpus.path));
  const missing = [...curveSources]
    .filter((source) => !discovered.has(source))
    .sort(compareText);
  if (missing.length > 0) {
    throw new Error(
      `${VIDEO_CURVES_PATH} cites record corpora that corpus discovery did not reach: ${missing.join(", ")}`,
    );
  }

  const catalogByCell = new Map(
    cells.map((cell) => [cellKey(cell.modelId, cell.backend, cell.tier), cell]),
  );

  // 1. Anchors from the retained corpora, one per identity cell, catalog-scoped.
  //
  // ANCHOR candidacy is restricted to corpora the Rust loader compiles in
  // (`PACKAGED_MEMORY_ANCHOR_SOURCES`): `validate_anchor` hard-rejects an anchor whose source is
  // not compiled in, so a row derived from an unpackaged corpus would make the whole store
  // unloadable. That mechanical restriction stays. What is GONE (sc-22666, epic 22657 E5) is the
  // discretion: packaging was an opt-in while the lane's law carried per-pixel slopes fitted on
  // Krea Turbo, because anchoring another model from a newly committed corpus would have repriced
  // it with borrowed slopes. The image law fits nothing since sc-22663, so skipping a retained
  // corpus is a defect, and `assertEveryDerivableCorpusIsPackaged` below fails the run when a
  // walked corpus could anchor a catalog cell but is not compiled in.
  const packagedSources = packagedAnchorSources(
    await readFile(path.join(root, PACKAGED_SOURCES_PATH), "utf8"),
  );
  assertEveryDerivableCorpusIsPackaged(corpora, packagedSources, catalogByCell);
  const byIdentity = new Map();
  for (const corpus of corpora) {
    if (!packagedSources.has(corpus.path)) continue;
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
    // A cell whose every retained render is in a composition the lane's law refuses is NOT
    // anchored: the row would be rejected on every lookup, so it would read as coverage while
    // admitting nothing. It falls through to the analytic-only pass below, where its largest
    // envelope is cited as `measured_envelope` — the honest classification for it.
    if (!isDerivable(chosen)) continue;
    const cell = catalogByCell.get(
      cellKey(chosen.modelId, chosen.backend, chosen.tier),
    );
    extracted.set(
      key,
      anchorRow(
        chosen,
        cell,
        previousStore,
        underivedReasonFor(chosen),
      ),
    );
  }

  // 2. Every emitted anchor must cite a corpus the Rust loader compiles in. A sibling story's
  //    anchors reach this store the same way these did — its corpus lands under a walked root and
  //    in `PACKAGED_MEMORY_ANCHOR_SOURCES`, and the regeneration derives them — so the two halves
  //    are checked against each other here rather than assumed to agree.
  const anchors = [...extracted.values()].sort((left, right) =>
    compareText(left.id, right.id),
  );
  assertPackagedSources(anchors, packagedSources);
  const anchoredCells = new Set(
    anchors.map((anchor) =>
      cellKey(anchor.modelId, anchor.backend, anchor.tier),
    ),
  );

  // 3. Provider constants at the PIN — OPT-IN. A default run reads no checkout, so this limb
  //    contributes nothing and the output cannot differ between a host that has the pin's cargo
  //    checkout and one that does not. Opting in is a deliberate, host-dependent act.
  const mlxModelIds = new Set(
    cells.filter((cell) => cell.backend === "mlx").map((cell) => cell.modelId),
  );
  const checkout =
    inferenceRoot === "auto"
      ? await locateInferenceCheckout(pin)
      : inferenceRoot;
  const providerConstants = await inferenceProviderConstants(
    checkout,
    pin,
    mlxModelIds,
  );

  // 4. Every remaining catalog cell is classified explicitly. There is no unclassified state.
  const analyticOnly = [];
  for (const cell of cells) {
    const key = cellKey(cell.modelId, cell.backend, cell.tier);
    if (anchoredCells.has(key)) continue;
    const envelope = envelopeEvidence(corpora, cell);
    const provider =
      cell.backend === "mlx"
        ? (providerConstants.get(cell.modelId) ?? null)
        : null;
    const contract = contractEstimateEvidence(
      manifest,
      MANIFEST_PATH,
      manifestSha256,
      cell,
    );
    const declared = manifestTierEvidence(
      manifest,
      MANIFEST_PATH,
      manifestSha256,
      cell,
    );
    const [basis, evidence] =
      envelope !== null
        ? ["measured_envelope", envelope]
        : provider !== null
          ? ["provider_measured_constants", provider]
          : contract !== null
            ? ["contract_estimate", contract]
            : declared !== null
              ? ["manifest_tier_declaration", declared]
              : ["no_retained_evidence", null];
    // An overlay render's envelope is evidence about a DIFFERENT resident set; say that in the row
    // rather than letting it read as a missing phase decomposition. `envelopeEvidence` only ever
    // returns an overlay record when NO clean render of the cell was retained, so the wording is
    // entailed by the selection rule rather than asserted about one chosen record.
    const reason =
      basis === "measured_envelope" && evidence?.values?.overlay
        ? `every retained render for this cell ran under the '${evidence.values.overlay}' ` +
          `overlay (the largest is on provider '${evidence.values.provider ?? "unknown"}'), which ` +
          "measures a different resident set, so it bounds the envelope without anchoring this cell"
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

  // 5. Component deltas (epic 22505 E2, feature-end fix round): emitted exactly when the store
  //    carries variant-keyed anchors the fall-through can price from. The rows' byte values come
  //    from the committed weights inventory and are recomputed by the Rust loader.
  const componentDeltas = anchors.some(
    (anchor) =>
      anchor.modelId === "ltx_2_5" &&
      anchor.backend === "mlx" &&
      anchor.transformerVariant !== null,
  )
    ? ltx25ComponentDeltas(
        await readFile(path.join(root, LTX25_WEIGHTS_INVENTORY_PATH), "utf8"),
        sortedUniqueTiers(cells, "ltx_2_5", "mlx"),
      )
    : [];

  return {
    schemaVersion: MEMORY_ANCHOR_SCHEMA_VERSION,
    anchors,
    analyticOnly,
    componentDeltas,
  };
}

function sortedUniqueTiers(cells, modelId, backend) {
  return [
    ...new Set(
      cells
        .filter((cell) => cell.modelId === modelId && cell.backend === backend)
        .map((cell) => cell.tier),
    ),
  ].sort(compareText);
}

export const serialiseStore = (store) => `${JSON.stringify(store, null, 2)}\n`;

async function main() {
  const args = process.argv.slice(2);
  const rootFlag = args.indexOf("--inference-root");
  // Off unless asked for, and `--check` gets the same resolution a generation gets: the check is
  // "regenerate from the committed evidence and compare", with no seeding from the file under test.
  const inferenceRoot = rootFlag === -1 ? null : (args[rootFlag + 1] ?? null);
  const store = await buildAnchorStore({ inferenceRoot });
  const serialised = serialiseStore(store);
  const target = path.join(ROOT, STORE_PATH);
  if (args.includes("--check")) {
    const existing = await readFile(target, "utf8");
    if (existing !== serialised) {
      process.stderr.write(
        `${STORE_PATH} is stale — re-run scripts/extract-memory-anchors.mjs\n`,
      );
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

if (
  process.argv[1] &&
  path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)
) {
  await main();
}
