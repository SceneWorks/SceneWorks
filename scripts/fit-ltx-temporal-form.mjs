#!/usr/bin/env node
/**
 * sc-18810 — fit and cross-validate the temporal coefficient form for the video phase curve.
 *
 * The shipped phase curve is two coefficients per phase per tier,
 * `fixedGb + perMpxGb * megapixels` (`crates/sceneworks-worker/src/vram_gate.rs:513`, schema at
 * `packages/schemas/model-manifest.schema.json:175`), and the admission peak is the MAX of three
 * such curves — text, denoise, decode (`vram_gate.rs:321-331`, `KreaTurboPhasePeaks::peak_gb`).
 * Video needs a temporal term. Epic 18803 left the FORM of that term open on purpose, to be closed
 * by measurement rather than argument; this script is the measurement half.
 *
 * Five candidate forms are fitted, per tier, per series, on the plan's `fit` points ONLY, and then
 * scored on the `held_out` points that no fit ever saw:
 *
 *   area_only      fixed + perMpx*mpx                    — the SHIPPED image form, as a baseline
 *   additive       fixed + perMpx*mpx + perFrame*frames
 *   cross          fixed + perMpx*mpx + perMpxFrame*(mpx*frames)
 *   latent_tokens  fixed + perToken*(T_lat*(W/32)*(H/32))
 *   output_voxels  fixed + perMpxFrame*(mpx*frames)      — the constrained cross term
 *
 * `latent_tokens` and `output_voxels` are CONSTRAINED special cases of `cross`: latent tokens are
 * `mpx * (frames + 7) * 976.5625/8` and output voxels are `mpx * frames * 1e6`, so both are
 * reachable by `cross` with coefficients tied. That is exactly why held-out residuals — not
 * in-sample residuals — decide: `cross` can never fit the training points worse, and the question is
 * whether its extra freedom buys generalisation or fits noise. The two constrained forms differ from
 * each other only by the causal `+7` frame offset (~4% across the declared frame range), so the
 * report has to say honestly whether this sweep can separate them.
 *
 * Selection is mechanical, not by eye: lowest held-out MAXIMUM absolute residual, with the in-sample
 * fit reported beside it. The measured replicate spread is reported as the noise floor, so "fits
 * within the noise floor" is a comparison against a number this dataset produced.
 *
 * Usage: node scripts/fit-ltx-temporal-form.mjs [--dataset <bundle.json>]... [--plan <plan.json>]
 *                                               [--driver-log <sweep-run.log>]...
 *                                               [--write <report.json>]
 *                                               [--curve-write <curves.json>] [--check]
 */
import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { isDeepStrictEqual } from "node:util";

import { stripJsoncComments } from "./lib/jsonc.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const GIB = 1024 ** 3;
// Artifact ordering must not depend on the host's ICU locale. All persisted identifiers and paths
// are UTF-8 protocol strings, so compare their JavaScript code-unit order directly.
const compareText = (left, right) => (left < right ? -1 : left > right ? 1 : 0);
const repoRelativePath = (file) => path.relative(ROOT, file).split(path.sep).join("/");
const isCanonicalRepoPath = (value) =>
  value !== "." &&
  !value.startsWith("/") &&
  !value.startsWith("../") &&
  !value.includes("\\") &&
  path.posix.normalize(value) === value;
// Kept in lockstep with `sceneworks_core::memory_calibration::MEMORY_CALIBRATION_ABI`, whose
// worker-side parity test in turn pins it to `gen_core::MEMORY_CALIBRATION_ABI`. A pin bump cannot
// make an old curve look current: the runtime query carries the provider contract's live ABI.
export const VIDEO_MEMORY_CURVE_CALIBRATION_ABI = 3;

/**
 * Dependency-free validator for the JSON-Schema keywords used by
 * `packages/schemas/video-memory-curves.schema.json`. This repo intentionally has no npm
 * dependencies; applying the checked-in schema here keeps it an executable producer contract
 * instead of editor-only documentation.
 */
export function videoCurveSchemaErrors(schema, value, root = schema, at = "$") {
  const errors = [];
  if (!schema || typeof schema !== "object") return errors;
  if (typeof schema.$ref === "string") {
    const target = schema.$ref
      .replace(/^#\//, "")
      .split("/")
      .reduce((node, key) => node?.[key], root);
    return videoCurveSchemaErrors(target, value, root, at);
  }
  if (Object.hasOwn(schema, "const") && value !== schema.const) {
    errors.push(`${at}: expected constant ${JSON.stringify(schema.const)}`);
  }
  if (Array.isArray(schema.enum) && !schema.enum.includes(value)) {
    errors.push(`${at}: ${JSON.stringify(value)} is outside ${JSON.stringify(schema.enum)}`);
  }
  const actual = Array.isArray(value) ? "array" : value === null ? "null" : typeof value;
  const allowedTypes = schema.type === undefined
    ? []
    : Array.isArray(schema.type)
      ? schema.type
      : [schema.type];
  const typeMatches =
    allowedTypes.length === 0 ||
    allowedTypes.includes(actual) ||
    (allowedTypes.includes("object") && actual === "object") ||
    (allowedTypes.includes("integer") && typeof value === "number" && Number.isInteger(value));
  if (!typeMatches) {
    errors.push(`${at}: expected ${schema.type}, got ${actual}`);
    return errors;
  }
  if (typeof value === "string") {
    if (typeof schema.minLength === "number" && value.length < schema.minLength) {
      errors.push(`${at}: shorter than minLength ${schema.minLength}`);
    }
    if (typeof schema.pattern === "string" && !new RegExp(schema.pattern).test(value)) {
      errors.push(`${at}: does not match ${schema.pattern}`);
    }
  }
  if (typeof value === "number" && typeof schema.minimum === "number" && value < schema.minimum) {
    errors.push(`${at}: ${value} is below minimum ${schema.minimum}`);
  }
  if (Array.isArray(value)) {
    if (typeof schema.minItems === "number" && value.length < schema.minItems) {
      errors.push(`${at}: fewer than minItems ${schema.minItems}`);
    }
    if (
      schema.uniqueItems === true &&
      new Set(value.map((item) => JSON.stringify(item))).size !== value.length
    ) {
      errors.push(`${at}: array items are not unique`);
    }
    value.forEach((item, index) => {
      errors.push(...videoCurveSchemaErrors(schema.items, item, root, `${at}[${index}]`));
    });
  } else if (actual === "object") {
    for (const key of schema.required ?? []) {
      if (!(key in value)) errors.push(`${at}: missing required property ${JSON.stringify(key)}`);
    }
    const properties = schema.properties ?? {};
    if (schema.additionalProperties === false) {
      for (const key of Object.keys(value)) {
        if (!(key in properties)) errors.push(`${at}: unknown property ${JSON.stringify(key)}`);
      }
    }
    for (const [key, child] of Object.entries(properties)) {
      if (key in value) {
        errors.push(...videoCurveSchemaErrors(child, value[key], root, `${at}.${key}`));
      }
    }
  }
  return errors;
}

/** LTX's video VAE is x32 spatial and x8 causal temporal: `out_f = 1 + (T_lat - 1) * 8`. */
export function latentTemporalDepth(frames) {
  return 1 + (frames - 1) / 8;
}

/** Stage-2 latent token count — the physically motivated regressor named by the story. */
export function latentTokens({ width, height, frames }) {
  return latentTemporalDepth(frames) * (width / 32) * (height / 32);
}

/**
 * The candidate design matrices. Each returns the row of regressors for one geometry; the first
 * entry is always the intercept, which is the form's `fixedGb`.
 */
export const FORMS = Object.freeze({
  area_only: {
    coefficients: ["fixedGb", "perMpxGb"],
    row: (g) => [1, g.mpx],
  },
  additive: {
    coefficients: ["fixedGb", "perMpxGb", "perFrameGb"],
    row: (g) => [1, g.mpx, g.frames],
  },
  cross: {
    coefficients: ["fixedGb", "perMpxGb", "perMpxFrameGb"],
    row: (g) => [1, g.mpx, g.mpx * g.frames],
  },
  latent_tokens: {
    coefficients: ["fixedGb", "perLatentTokenGb"],
    row: (g) => [1, g.tokens],
  },
  // The constrained cross term on the regressor the ENGINE's own decode cost model uses: output
  // voxels `frames * H * W` (`mlx-gen-ltx/src/pipeline.rs:218-256` charges 3.3 GB + 340 B/voxel for
  // a single-pass decode).
  output_voxels: {
    coefficients: ["fixedGb", "perMpxFrameGb"],
    row: (g) => [1, g.mpx * g.frames],
  },
});

/**
 * Ordinary least squares by Gauss-Jordan on the normal equations. The design matrices here are at
 * most 3x3 and well-conditioned by construction (the sweep crosses two area levels with three frame
 * levels precisely so the columns are not collinear), so a dedicated decomposition would be
 * ceremony. A singular system returns `null` rather than NaN coefficients, so a degenerate sweep
 * fails loudly instead of publishing a curve fitted on nothing.
 */
export function leastSquares(rows, targets) {
  const width = rows[0].length;
  if (rows.length < width) return null;
  const normal = Array.from({ length: width }, (_, i) => {
    const row = Array.from({ length: width }, (_, j) =>
      rows.reduce((sum, r) => sum + r[i] * r[j], 0),
    );
    row.push(rows.reduce((sum, r, k) => sum + r[i] * targets[k], 0));
    return row;
  });
  for (let column = 0; column < width; column += 1) {
    let pivot = column;
    for (let candidate = column + 1; candidate < width; candidate += 1) {
      if (Math.abs(normal[candidate][column]) > Math.abs(normal[pivot][column])) pivot = candidate;
    }
    if (Math.abs(normal[pivot][column]) < 1e-12) return null;
    [normal[column], normal[pivot]] = [normal[pivot], normal[column]];
    const scale = normal[column][column];
    for (let j = column; j <= width; j += 1) normal[column][j] /= scale;
    for (let row = 0; row < width; row += 1) {
      if (row === column) continue;
      const factor = normal[row][column];
      if (factor === 0) continue;
      for (let j = column; j <= width; j += 1) normal[row][j] -= factor * normal[column][j];
    }
  }
  return normal.map((row) => row[width]);
}

function residuals(points, form, coefficients) {
  return points.map((point) => {
    const predicted = FORMS[form]
      .row(point.geometry)
      .reduce((sum, value, index) => sum + value * coefficients[index], 0);
    return {
      fixture: point.fixture,
      role: point.role,
      measuredGib: point.value,
      predictedGib: predicted,
      residualGib: predicted - point.value,
    };
  });
}

function summarise(entries) {
  if (entries.length === 0) return { count: 0, rmseGib: null, maxAbsGib: null };
  const squares = entries.reduce((sum, entry) => sum + entry.residualGib ** 2, 0);
  return {
    count: entries.length,
    rmseGib: Math.sqrt(squares / entries.length),
    maxAbsGib: Math.max(...entries.map((entry) => Math.abs(entry.residualGib))),
  };
}

/** Fit every candidate for one (tier, series) slice and score it on the held-out points. */
export function fitSlice(fitPoints, heldOutPoints) {
  const candidates = {};
  for (const [name, form] of Object.entries(FORMS)) {
    const coefficients = leastSquares(
      fitPoints.map((point) => form.row(point.geometry)),
      fitPoints.map((point) => point.value),
    );
    if (!coefficients) {
      candidates[name] = { singular: true };
      continue;
    }
    const inSample = residuals(fitPoints, name, coefficients);
    const heldOut = residuals(heldOutPoints, name, coefficients);
    candidates[name] = {
      coefficients: Object.fromEntries(
        form.coefficients.map((label, index) => [label, coefficients[index]]),
      ),
      fit: summarise(inSample),
      heldOut: summarise(heldOut),
      points: [...inSample, ...heldOut],
    };
  }
  // Mechanical selection: the smallest held-out maximum absolute residual wins. Ties fall back to
  // the in-sample maximum so the rule is total.
  const ranked = Object.entries(candidates)
    .filter(([, candidate]) => !candidate.singular && candidate.heldOut.count > 0)
    .sort(
      (left, right) =>
        left[1].heldOut.maxAbsGib - right[1].heldOut.maxAbsGib ||
        left[1].fit.maxAbsGib - right[1].fit.maxAbsGib,
    );
  return { candidates, chosen: ranked[0]?.[0] ?? null };
}

/**
 * The measured noise floor: the spread between records that share a logical geometry (tier, width,
 * height, frames, fps) but were captured in separate provider invocations. Without replicates this
 * returns `null` and every "within the noise floor" claim downstream has to say so.
 *
 * 🔴 Each spread carries the SceneWorks revisions it spans, and `crossRevision` says whether any
 * group spans more than one. In the committed dataset every one of the four replicate groups does,
 * because the capture ran across two driver sessions and four revisions — so this floor bounds
 * repeat-plus-revision variation, not repeat variation alone, and every "×noise" statement judged
 * against it inherits that. It is reported rather than corrected because the alternative is
 * re-rendering, and a floor that is too WIDE is the conservative direction for a residual to be
 * compared against.
 */
export function noiseFloor(points) {
  const groups = new Map();
  for (const point of points) {
    if (!groups.has(point.replicateKey)) groups.set(point.replicateKey, []);
    groups.get(point.replicateKey).push(point);
  }
  const spreads = [...groups.entries()]
    .filter(([, members]) => members.length > 1)
    .map(([key, members]) => {
      const values = members.map((member) => member.value);
      return {
        key,
        replicates: members.length,
        meanGib: values.reduce((sum, value) => sum + value, 0) / values.length,
        spreadGib: Math.max(...values) - Math.min(...values),
        sceneWorksRevisions: [
          ...new Set(members.map((member) => member.sceneWorksRevision).filter(Boolean)),
        ].sort(),
      };
    });
  if (spreads.length === 0) {
    return {
      replicatedGeometries: 0,
      maxSpreadGib: null,
      maxSpreadFraction: null,
      crossRevision: false,
      spreads: [],
    };
  }
  return {
    replicatedGeometries: spreads.length,
    maxSpreadGib: Math.max(...spreads.map((entry) => entry.spreadGib)),
    maxSpreadFraction: Math.max(...spreads.map((entry) => entry.spreadGib / entry.meanGib)),
    crossRevision: spreads.some((entry) => entry.sceneWorksRevisions.length > 1),
    spreads,
  };
}

/**
 * The measured series. The three phases are what the shipped curve structure actually stores — one
 * two-coefficient curve each — and `overallActive` is their max, which is the admission quantity.
 *
 * CAVEAT on `text`, stated because it bounds what the series means: the arm's `conditioning` window
 * runs from generate start to `Progress::Step { current: 1 }`, so it spans the staged Gemma text
 * encoder AND the AvDiT build AND the first denoise step. It is a clean text-phase measurement only
 * while the text phase dominates. Where denoise overtakes it, this series tracks the first denoise
 * step instead, which is a phase-composition artifact rather than geometry sensitivity in the text
 * encoder. `overallActive` is unaffected — the union of the three windows still covers the run.
 */
const SERIES = Object.freeze({
  text: (record) => record.observedMemory.conditioning.activeBytes,
  denoise: (record) => record.observedMemory.denoise.activeBytes,
  decode: (record) => record.observedMemory.decode.activeBytes,
  overallActive: (record) => record.observedMemory.overall.activeBytes,
  // The tightest active+cache quantity this apparatus can produce: per-phase peak ACTIVE plus
  // end-of-phase CACHE, maximised over the phases. It is an UPPER BOUND on co-existence, not a
  // simultaneous maximum — `PhaseMemory::capture()` (`mlx.rs`, `get_peak_memory` + `get_cache_memory`)
  // pairs a phase-WINDOW peak with an INSTANTANEOUS end-of-phase cache reading, and MLX exposes no
  // "cache at the active peak". During an LTX decode the cache enters at ~0 and grows monotonically,
  // so the two instants are furthest apart exactly where the number is largest.
  // It is still strictly tighter than `observedMemory.overall.allocatorBytes`, which is
  // `max(active) + max(cache)` across DIFFERENT phases (active peaks in the text phase at q4/q8,
  // cache in the decode) and over-bounds by tens of GiB.
  maxPhaseActivePlusCache: (record) =>
    Math.max(
      ...["conditioning", "denoise", "decode"].map(
        (phase) =>
          record.observedMemory[phase].activeBytes + record.observedMemory[phase].reclaimableBytes,
      ),
    ),
});

export function pointsFrom(records, roleByFixture, manifest = null) {
  return records.map((record) => {
    const { width, height, frames } = record.target.geometry;
    const measurements = Object.fromEntries(
      record.diagnostics.measurements.map((entry) => [entry.name, entry.value]),
    );
    const role = roleByFixture.get(record.fixture);
    if (!role) throw new Error(`record fixture ${record.fixture} has no role in the sweep plan`);
    const fps = measurements.outputFps;
    const referenceCount = record.target.referenceCount ?? (record.target.mode === "text_to_video" ? 0 : null);
    const referenceShape = record.target.referenceShape ?? (referenceCount === 0 ? "none" : null);
    if (!Number.isInteger(referenceCount) || referenceCount < 0 || typeof referenceShape !== "string" || referenceShape.length === 0 || (referenceShape === "none") !== (referenceCount === 0) || !Number.isInteger(fps) || fps < 1) {
      throw new Error(`record ${record.id} has an incomplete reference/FPS evidence identity`);
    }
    const modelFamily = manifest?.models?.find((model) => model.id === record.target.modelId)?.family;
    if (manifest && typeof modelFamily !== "string") {
      throw new Error(`model ${record.target.modelId} is absent from builtin.models.jsonc`);
    }
    return {
      recordId: record.id,
      fixture: record.fixture,
      capturedAt: record.capturedAt,
      modelId: record.target.modelId,
      modelFamily,
      route: record.target.route ?? record.target.provider,
      provider: record.target.provider,
      backend: record.backend,
      tier: record.target.tier,
      mode: record.target.mode,
      referenceShape,
      referenceCount,
      overlay: record.target.overlay === "none" ? null : record.target.overlay,
      role,
      rung: record.strategy.rung,
      loadShape: record.loadShape,
      batch: record.target.geometry.batch,
      closureDigest: record.repositories.inference.closureDigest,
      calibrationFingerprint: record.calibrationFingerprint,
      calibrationAbi: VIDEO_MEMORY_CURVE_CALIBRATION_ABI,
      decodeTilingEngaged: measurements.decodeTilingEngaged === 1,
      sceneWorksRevision: record.repositories.sceneWorks.revision,
      // A repeat is same-CELL evidence, not merely the same tier/geometry. Keeping every selector
      // axis here prevents a multi-curve campaign from calling a rung/provider/closure change
      // capture noise.
      replicateKey: JSON.stringify([
        record.target.modelId,
        record.target.route ?? record.target.provider,
        record.target.provider,
        record.backend,
        record.target.tier,
        record.target.mode,
        record.strategy.rung,
        record.loadShape,
        record.target.geometry.batch,
        record.repositories.inference.closureDigest,
        record.calibrationFingerprint,
        measurements.decodeTilingEngaged,
        width,
        height,
        frames,
        fps,
      ]),
      geometry: {
        width,
        height,
        frames,
        fps,
        mpx: (width * height) / 1e6,
        tLat: latentTemporalDepth(frames),
        tokens: latentTokens({ width, height, frames }),
      },
      series: Object.fromEntries(
        Object.entries(SERIES).map(([name, read]) => [name, read(record) / GIB]),
      ),
      cacheGib: {
        text: record.observedMemory.conditioning.reclaimableBytes / GIB,
        denoise: record.observedMemory.denoise.reclaimableBytes / GIB,
        decode: record.observedMemory.decode.reclaimableBytes / GIB,
      },
    };
  });
}

function persistedObservation(point) {
  return {
    recordId: point.recordId,
    fixture: point.fixture,
    modelId: point.modelId,
    modelFamily: point.modelFamily,
    route: point.route,
    provider: point.provider,
    backend: point.backend,
    tier: point.tier,
    mode: point.mode,
    role: point.role,
    rung: point.rung,
    loadShape: point.loadShape,
    batch: point.batch,
    closureDigest: point.closureDigest,
    calibrationFingerprint: point.calibrationFingerprint,
    calibrationAbi: point.calibrationAbi,
    decodeTilingEngaged: point.decodeTilingEngaged,
    sceneWorksRevision: point.sceneWorksRevision,
    geometry: point.geometry,
    activeGib: point.series,
    cacheGib: point.cacheGib,
  };
}

/**
 * The driver's own terminal state per plan entry, parsed from the committed sweep logs
 * (`docs/calibration/sc-18810/*.log`, written verbatim by the sweep driver — see its
 * `BEGIN`/`OK`/`FAIL`/`STOP` writes).
 *
 * This exists because the previous version of this script carried a HARDCODED two-element list of
 * "attempted and killed" fixtures, and it was wrong: `1280x704 f241` was reported
 * `not_attempted_host_limit` while the log shows `BEGIN ... free=16GiB 11:40:16` and no terminal
 * line at all. A coverage state that is typed by hand is a claim about the run that nothing checks.
 *
 * The capture spans TWO driver sessions and therefore two logs — the first one crashed the host
 * mid-run — so this takes an ARRAY of log texts and sums the per-name counters across them. A
 * single string is accepted as a one-element array. Counts, not just booleans, are summed because
 * the coverage guard compares a fixture's `OK` count against how many records it produced; a
 * session whose log is missing shows up there as records with no terminal line.
 *
 * Terminal states, all four of which the logs distinguish:
 *   `completed`           — the driver wrote `OK`.
 *   `failed`              — the driver wrote `FAIL`; the child exited non-zero or was killed.
 *   `arithmetic_unmeasurable` — no child was started; the driver wrote an explicit
 *                           `ARITHMETIC_UNMEASURABLE name :: reason` proof.
 *   `no_terminal_record`  — a `BEGIN` with no `OK`/`FAIL`. The driver ITSELF did not survive to
 *                           write one. Attempted, and it did not survive; not "never run".
 *   `not_begun`           — no `BEGIN` line. `stoppedBefore` marks the entry the driver named when
 *                           it halted on the free-disk floor.
 * An entry re-run after a failure (`704x1280 f177` was refused by the staged-residency guard, then
 * re-run and captured) counts as `completed`: one `OK` is enough.
 *
 * A terminal line is believed only while an unmatched `BEGIN` for that name is open. The driver
 * pipes each child's stderr into the same stream it writes these lines to, so an own-line
 * `OK <planname> 1s` inside captured child output would otherwise be indistinguishable from the
 * driver's own verdict and could flip an entry to `completed` — the one bucket that silences the
 * record-versus-terminal guard below. The committed logs are unaffected either way; this closes the
 * channel rather than trusting the transcripts to stay clean.
 */
export function driverStatesFrom(logs) {
  const texts = Array.isArray(logs) ? logs : [logs];
  const states = new Map();
  const of = (name) => {
    if (!states.has(name)) {
      states.set(name, {
        begins: 0,
        oks: 0,
        fails: 0,
        failReasons: [],
        arithmeticReasons: [],
        stoppedBefore: false,
      });
    }
    return states.get(name);
  };
  for (const text of texts) {
    for (const line of text.split("\n")) {
      let match;
      if ((match = /^BEGIN (\S+) /.exec(line))) {
        of(match[1]).begins += 1;
      } else if ((match = /^(OK|FAIL) (\S+) /.exec(line))) {
        const state = of(match[2]);
        // No open BEGIN ⇒ this line cannot be the driver's verdict on this entry.
        if (state.begins <= state.oks + state.fails) continue;
        if (match[1] === "OK") state.oks += 1;
        else {
          state.fails += 1;
          state.failReasons.push(line.split(" :: ").slice(1).join(" :: "));
        }
      } else if ((match = /^ARITHMETIC_UNMEASURABLE (\S+) :: (.+)$/.exec(line))) {
        of(match[1]).arithmeticReasons.push(match[2]);
      } else if ((match = /^STOP .* before (\S+)\s*$/.exec(line))) {
        of(match[1]).stoppedBefore = true;
      }
    }
  }
  for (const state of states.values()) {
    state.terminal =
      state.oks > 0
        ? "completed"
        : state.begins === 0 && state.arithmeticReasons.length > 0
          ? "arithmetic_unmeasurable"
        : state.begins === 0
          ? "not_begun"
          : state.fails > 0
            ? "failed"
            : "no_terminal_record";
  }
  return states;
}

/**
 * Per-session provenance, derived from the same logs. It exists because the replicates that produce
 * this dataset's noise floor are NOT within-session: every one of the four replicated geometries has
 * one record from the crashed first session and one from the second, and the two sessions ran
 * different SceneWorks revisions. A "measured noise floor" that silently spans revisions is a
 * different — weaker — claim than one measured back to back, so the report has to carry which
 * session and which revision each record came from.
 *
 * Records are attributed WITHOUT clock arithmetic: within a fixture, the nth record in `capturedAt`
 * order is the nth `OK` for that name in session order. The driver runs one child at a time and
 * appends, so both sequences are chronological by construction, and the counts are already forced
 * to agree by `coverageOf`'s record-versus-terminal guard. Two of the thirteen `capturedAt` values
 * differ by one second from `BEGIN + duration` (the driver rounds the duration), which is exactly
 * why timestamps are not used to pair them.
 */
export function sessionsFrom(logs, points, fixtureByName) {
  // `okOrders[name]` is the session index of each successive OK for that driver name.
  const okOrders = new Map();
  const sessions = logs.map(({ path: logPath, text }, order) => {
    let begins = 0;
    let completed = 0;
    let failed = 0;
    let firstBeginAt = null;
    const open = new Map();
    for (const line of text.split("\n")) {
      let match;
      if ((match = /^BEGIN (\S+) .* (\d\d:\d\d:\d\d)\s*$/.exec(line))) {
        begins += 1;
        firstBeginAt ??= match[2];
        open.set(match[1], (open.get(match[1]) ?? 0) + 1);
      } else if ((match = /^(OK|FAIL) (\S+) /.exec(line))) {
        if ((open.get(match[2]) ?? 0) === 0) continue;
        open.set(match[2], open.get(match[2]) - 1);
        if (match[1] === "FAIL") {
          failed += 1;
          continue;
        }
        completed += 1;
        if (!okOrders.has(match[2])) okOrders.set(match[2], []);
        okOrders.get(match[2]).push(order);
      }
    }
    return {
      log: logPath,
      firstBeginAt,
      begins,
      completed,
      failed,
      begunWithoutTerminalLine: begins - completed - failed,
      records: 0,
      sceneWorksRevisions: [],
    };
  });
  const revisions = sessions.map(() => new Set());
  const byFixture = new Map();
  for (const point of points) {
    if (!byFixture.has(point.fixture)) byFixture.set(point.fixture, []);
    byFixture.get(point.fixture).push(point);
  }
  for (const [name, orders] of okOrders) {
    const records = byFixture.get(fixtureByName.get(name)) ?? [];
    records
      .slice()
      .sort((left, right) => compareText(left.capturedAt, right.capturedAt))
      .forEach((record, index) => {
        const order = orders[index];
        if (order === undefined) return;
        sessions[order].records += 1;
        revisions[order].add(record.sceneWorksRevision);
      });
  }
  sessions.forEach((session, order) => {
    session.sceneWorksRevisions = [...revisions[order]].sort();
  });
  return sessions;
}

/**
 * Per-planned-entry coverage. A sweep that silently drops points is indistinguishable from one that
 * never planned them, so every planned entry lands in exactly one bucket and the report carries the
 * count.
 *
 * The bucket is DERIVED — from the dataset (was a record captured?) and from the driver log (was the
 * geometry ever begun?). `_role` is consulted for exactly one thing: telling a geometry the sweep
 * deliberately declined to attempt (`not_attempted_host_limit`) apart from one it simply never got
 * to (`not_reached`), which the log cannot distinguish because neither has a `BEGIN`.
 *
 * `_role` is a PRE-REGISTRATION label and the driver log is a RUN RECORD, so the two are checked
 * against each other rather than one being derived from the other. A `fit` row can perfectly well
 * have been attempted and killed — `q8 768x512 f361` is — and that does not make it any less a `fit`
 * row. What is NOT allowed is a `*_host_limit` role that contradicts the log; that is the exact
 * defect this function first threw on.
 *
 * Two further disagreements throw, both found in re-review:
 *
 * 1. **A record with no `OK` terminal in any committed log.** The link from a plan entry to the log
 *    runs through `provider.name`, while records key on `provider.fixture`, so before this check
 *    nothing tied the two together for any role but `*_host_limit`: renaming all eight captured
 *    providers threw nothing and left `byState` byte-identical, and four of the thirteen records
 *    came from a session whose log was not committed at all. Requiring `captures <= oks` per entry
 *    forces every record to be accounted for by a terminal line in a log that ships with it.
 * 2. **A driver name that matches no plan entry.** The mirror of the same break: a log naming an
 *    entry the plan does not declare is a plan/log divergence, and silently ignoring it is how a
 *    renamed provider stops being checked.
 */
export function coverageOf(plan, points, driverStates = new Map()) {
  const captured = new Map();
  for (const point of points) {
    captured.set(point.fixture, (captured.get(point.fixture) ?? 0) + 1);
  }
  const declaredNames = new Set(plan.providers.map((provider) => provider.name));
  for (const name of driverStates.keys()) {
    if (!declaredNames.has(name)) {
      throw new Error(
        `the driver log names ${name}, which is not a provider in the sweep plan — the plan and ` +
          `the log disagree about which entries exist`,
      );
    }
  }
  const entries = plan.providers.map((provider) => {
    const state = driverStates.get(provider.name);
    const terminal = state?.terminal ?? "not_begun";
    const attempted = (state?.begins ?? 0) > 0;
    if (provider._role === "not_attempted_host_limit" && attempted) {
      throw new Error(
        `${provider.fixture} is declared not_attempted_host_limit but the driver log records it as ` +
          `${terminal} — the plan and the log disagree about whether it was run`,
      );
    }
    if (provider._role === "attempted_failed_host_limit" && !attempted) {
      throw new Error(
        `${provider.fixture} is declared attempted_failed_host_limit but the driver log has no ` +
          `BEGIN line for it`,
      );
    }
    const captures = captured.get(provider.fixture) ?? 0;
    if (captures > (state?.oks ?? 0)) {
      throw new Error(
        `${provider.fixture} has ${captures} record(s) but the committed driver logs record ` +
          `${state?.oks ?? 0} OK terminal(s) for ${provider.name} — a captured record with no ` +
          `terminal line means its session's log is missing, so its provenance is unstated`,
      );
    }
    return {
      fixture: provider.fixture,
      role: provider._role,
      tier: provider.target.tier,
      outputVoxels: provider._outputVoxels,
      latentTokens: provider._latentTokens,
      driverTerminalState: terminal,
      state:
        captures > 0
          ? "captured"
          : terminal === "arithmetic_unmeasurable"
            ? "arithmetic_unmeasurable"
            : terminal === "failed" &&
              state.failReasons.length > 0 &&
              state.failReasons.every((reason) => reason.startsWith("arithmetic_unmeasurable:"))
            ? "arithmetic_unmeasurable"
            : terminal === "failed" || terminal === "no_terminal_record"
              ? "attempted_failed_host_limit"
            : // The driver wrote OK and the dataset has no record for it — a retention hole, not a
              // host limit. Given its own bucket rather than folded into either neighbour, because
              // silently calling it "failed" is the same class of mislabel this function exists to
              // stop. Zero of these in the committed run.
              terminal === "completed"
              ? "completed_without_record"
              : provider._role === "not_attempted_host_limit"
                ? "not_attempted_host_limit"
                : // The entry the driver NAMED on its STOP line: the run halted on the free-disk
                  // floor immediately before it. Its own bucket rather than folded into
                  // `not_reached`, because it is the one never-run entry the driver made a
                  // statement about — and in the committed run it is a `fit` row, so burying it
                  // among 25 unreached rows hid that the realized fit design is smaller than the
                  // declared one.
                  state?.stoppedBefore
                  ? "stopped_before"
                  : "not_reached",
      captures,
    };
  });
  // Derived, including the per-tier split: the runbook used to state that split in prose and got it
  // wrong (it called 26 unreached rows "all bf16 and q4" while three were q8, one of them a `fit`
  // row). A count that is typed rather than computed is the defect class this whole function exists
  // to remove, so the split is computed here and quoted from here.
  const byState = {};
  for (const entry of entries) {
    byState[entry.state] ??= { total: 0, byTier: {} };
    byState[entry.state].total += 1;
    byState[entry.state].byTier[entry.tier] = (byState[entry.state].byTier[entry.tier] ?? 0) + 1;
  }
  return {
    plannedEntries: entries.length,
    // Fixtures and GEOMETRIES are different counts and the runbook conflated them: the fps probe
    // repeats {768x512, f241} at 24 fps, so eight captured fixtures cover seven distinct
    // {w,h,frames} geometries. §3 of the runbook argues those two are the same geometry, so the
    // report may not simultaneously count them as two.
    capturedFixtures: new Set(points.map((point) => point.fixture)).size,
    capturedGeometries: new Set(
      points.map((point) => {
        const { width, height, frames } = point.geometry;
        return `${width}x${height}:f${frames}`;
      }),
    ).size,
    byState,
    entries,
  };
}

function completeSelectorOfPoint(point) {
  const selector = {
    modelId: point.modelId,
    modelFamily: point.modelFamily,
    route: point.route,
    provider: point.provider,
    backend: point.backend,
    tier: point.tier,
    mode: point.mode,
    referenceShape: point.referenceShape,
    referenceCount: point.referenceCount,
    overlay: point.overlay,
    rung: point.rung,
    loadShape: point.loadShape,
    batch: point.batch,
    closureDigest: point.closureDigest,
    calibrationAbi: point.calibrationAbi,
    calibrationFingerprint: point.calibrationFingerprint,
    decodePass: point.decodeTilingEngaged ? "tiled" : "single_pass",
  };
  if (
    Object.entries(selector).some(
      ([key, value]) => value === undefined || value === "" || (value === null && key !== "overlay"),
    )
  ) {
    throw new Error(`record ${point.recordId} has an incomplete video-curve selector`);
  }
  return selector;
}

function selectorFitReport(scopedPoints) {
  const fits = {};
  for (const series of Object.keys(SERIES)) {
    const valued = scopedPoints.map((point) => ({ ...point, value: point.series[series] }));
    // `role` decides membership and lives in the committed PLAN, not in the records, so the
    // fit/held-out split cannot be redrawn after seeing residuals. A replicate of a fit geometry
    // is training data; a replicate of a held-out geometry stays held out.
    const fitPoints = valued.filter((point) => point.role === "fit");
    const heldOutPoints = valued.filter((point) => point.role.startsWith("held_out"));
    // Everything else is carried but NOT scored. A tiled/rung-boundary point belongs to another
    // complete selector and never contaminates this fit merely because its tier is the same.
    const unscoredPoints = valued.filter(
      (point) => point.role !== "fit" && !point.role.startsWith("held_out"),
    );
    if (fitPoints.length === 0) continue;
    const slice = fitSlice(fitPoints, heldOutPoints);
    fits[series] = {
      fitPoints: fitPoints.length,
      heldOutPoints: heldOutPoints.length,
      unscoredPoints: unscoredPoints.length,
      ...slice,
      unscored: Object.fromEntries(
        Object.entries(slice.candidates)
          .filter(([, candidate]) => !candidate.singular)
          .map(([name, candidate]) => [
            name,
            residuals(unscoredPoints, name, Object.values(candidate.coefficients)),
          ]),
      ),
    };
  }
  return fits;
}

function tierAnalysisFits(selectorFits) {
  const eligible = selectorFits.filter(
    (entry) =>
      entry.selector.rung === "staged_residency" &&
      entry.selector.decodePass === "single_pass",
  );
  const grouped = new Map();
  for (const entry of eligible) {
    if (!grouped.has(entry.selector.tier)) grouped.set(entry.selector.tier, []);
    grouped.get(entry.selector.tier).push(entry);
  }
  return grouped;
}

export function heldOutCoefficientTransfer(selectorFits) {
  const grouped = tierAnalysisFits(selectorFits);
  const byTier = new Map(
    [...grouped].filter(([, entries]) => entries.length === 1).map(([tier, entries]) => [tier, entries[0]]),
  );
  const q8 = byTier.get("q8");
  const requiredTargets = ["q4", "bf16"];
  const targets = requiredTargets.filter((tier) => byTier.has(tier));
  const missingTiers = ["q8", ...requiredTargets].filter((tier) => !byTier.has(tier));
  if (missingTiers.length > 0) {
    return {
      status: "insufficient_data",
      referenceTier: "q8",
      targetTiers: requiredTargets,
      missingTiers,
      verdict: "open",
      reason: "one unambiguous staged-residency single-pass selector is required for q8, q4 and bf16",
      phases: {},
    };
  }
  const phases = {};
  let allFullTransfersPass = true;
  let allSlopeTransfersPass = true;
  let allFullTransfersMeasured = true;
  let allSlopeTransfersMeasured = true;
  for (const series of ["text", "denoise", "decode"]) {
    const q8Cross = q8.fits?.[series]?.candidates?.cross;
    phases[series] = {};
    if (!q8Cross || q8Cross.singular) {
      phases[series].status = "insufficient_data";
      phases[series].reason = "q8 cross fit is absent or singular";
      allFullTransfersMeasured = false;
      allSlopeTransfersMeasured = false;
      continue;
    }
    const q8Coefficients = q8Cross.coefficients;
    for (const tier of targets) {
      const target = byTier.get(tier);
      const targetFit = target.fits?.[series];
      const targetCross = targetFit?.candidates?.cross;
      const points = target.points.map((point) => ({ ...point, value: point.series[series] }));
      const fitPoints = points.filter((point) => point.role === "fit");
      const heldOutPoints = points.filter((point) => point.role.startsWith("held_out"));
      if (!targetCross || targetCross.singular || fitPoints.length === 0 || heldOutPoints.length === 0) {
        phases[series][tier] = { status: "insufficient_data" };
        allFullTransfersMeasured = false;
        allSlopeTransfersMeasured = false;
        continue;
      }
      const direct = residuals(heldOutPoints, "cross", Object.values(q8Coefficients));
      const temporalSlope = q8Coefficients.perMpxFrameGb;
      const interceptArea = leastSquares(
        fitPoints.map((point) => [1, point.geometry.mpx]),
        fitPoints.map((point) => point.value - temporalSlope * point.geometry.mpx * point.geometry.frames),
      );
      if (!interceptArea) {
        phases[series][tier] = { status: "insufficient_data", reason: "target intercept/area refit is singular" };
        allFullTransfersMeasured = false;
        allSlopeTransfersMeasured = false;
        continue;
      }
      const slopeTransfer = heldOutPoints.map((point) => {
        const predicted = interceptArea[0] + interceptArea[1] * point.geometry.mpx +
          temporalSlope * point.geometry.mpx * point.geometry.frames;
        return {
          fixture: point.fixture,
          measuredGib: point.value,
          predictedGib: predicted,
          residualGib: predicted - point.value,
        };
      });
      const targetNoise = noiseFloor(points);
      const toleranceGib = targetCross.heldOut.maxAbsGib + (targetNoise.maxSpreadGib ?? 0);
      const directSummary = summarise(direct);
      const directPassed = directSummary.maxAbsGib <= toleranceGib + 1e-12;
      const slopeSummary = summarise(slopeTransfer);
      const passed = slopeSummary.maxAbsGib <= toleranceGib + 1e-12;
      allFullTransfersPass &&= directPassed;
      allSlopeTransfersPass &&= passed;
      phases[series][tier] = {
        status: "measured",
        directQ8Coefficients: {
          coefficients: q8Coefficients,
          ...directSummary,
          residuals: direct,
          toleranceGib,
          verdict: directPassed ? "transfers" : "does_not_transfer",
        },
        q8TemporalSlopeWithTargetInterceptArea: {
          coefficients: {
            fixedGb: interceptArea[0],
            perMpxGb: interceptArea[1],
            perMpxFrameGb: temporalSlope,
          },
          ...slopeSummary,
          residuals: slopeTransfer,
          toleranceGib,
          targetOwnHeldOutEnvelopeGib: targetCross.heldOut.maxAbsGib,
          sameSelectorReplicateNoiseGib: targetNoise.maxSpreadGib,
          verdict: passed ? "transfers" : "does_not_transfer",
        },
      };
    }
  }
  return {
    status: allFullTransfersMeasured ? "measured" : "insufficient_data",
    referenceTier: "q8",
    targetTiers: targets,
    verdict: !allFullTransfersMeasured
      ? "open"
      : allFullTransfersPass
        ? "q8_coefficients_transfer"
        : "per_tier_coefficients_required",
    temporalSlopeStatus: allSlopeTransfersMeasured ? "measured" : "insufficient_data",
    temporalSlopeVerdict: !allSlopeTransfersMeasured
      ? "open"
      : allSlopeTransfersPass
        ? "q8_temporal_slopes_transfer_after_target_intercept_area_refit"
        : "per_tier_temporal_slopes_required",
    threshold: "target tier own cross held-out max absolute residual plus same-selector replicate spread",
    phases,
  };
}

export function phaseFlipVerdict(selectorFits) {
  const tiers = {};
  const grouped = tierAnalysisFits(selectorFits);
  for (const [tier, entries] of grouped) {
    if (entries.length !== 1) {
      tiers[tier] = {
        status: "insufficient_data",
        reason: "tier has multiple staged-residency single-pass selectors",
        selectors: entries.map((entry) => entry.selector),
      };
      continue;
    }
    const entry = entries[0];
    const bindings = entry.points
      .filter((point) => point.role === "fit" || point.role.startsWith("held_out"))
      .map((point) => {
        const values = Object.fromEntries(["text", "denoise", "decode"].map((phase) => [phase, point.series[phase]]));
        const maximum = Math.max(...Object.values(values));
        const bindingPhases = Object.entries(values)
          .filter(([, value]) => value === maximum)
          .map(([phase]) => phase);
        const bindingPhase = bindingPhases.length === 1 ? bindingPhases[0] : null;
        return {
          fixture: point.fixture,
          geometry: point.geometry,
          activeGib: values,
          bindingPhase,
          ...(bindingPhase === null ? { bindingPhases } : {}),
        };
      });
    const surfaces = [];
    for (const [left, right] of [["text", "denoise"], ["text", "decode"], ["denoise", "decode"]]) {
      const a = entry.fits?.[left]?.candidates?.cross?.coefficients;
      const b = entry.fits?.[right]?.candidates?.cross?.coefficients;
      if (!a || !b) continue;
      const byArea = [...new Set(bindings.map((binding) => binding.geometry.mpx))].sort((x, y) => x - y)
        .map((mpx) => {
          const denominator = (a.perMpxFrameGb - b.perMpxFrameGb) * mpx;
          const frames = Math.abs(denominator) < 1e-12
            ? null
            : -((a.fixedGb - b.fixedGb) + (a.perMpxGb - b.perMpxGb) * mpx) / denominator;
          return { mpx, frames: Number.isFinite(frames) && frames > 0 ? frames : null };
        });
      surfaces.push({ phases: [left, right], byArea });
    }
    const bindingCounts = {
      ...Object.fromEntries(["text", "denoise", "decode"].map((phase) => [
        phase,
        bindings.filter((binding) => binding.bindingPhase === phase).length,
      ])),
      ambiguous: bindings.filter((binding) => binding.bindingPhase === null).length,
    };
    const observedBindingPhases = ["text", "denoise", "decode"]
      .filter((phase) => bindingCounts[phase] > 0);
    tiers[tier] = {
      status: "measured",
      selector: entry.selector,
      bindingCounts,
      verdict: bindingCounts.ambiguous > 0
        ? "ambiguous_binding_at_captured_geometry"
        : observedBindingPhases.length > 1
          ? "phase_binding_flips_with_geometry"
          : "one_phase_binds_across_captured_geometry",
      bindings,
      curvePhaseFlipSurfaces: surfaces,
    };
  }
  const requiredTiers = ["q4", "bf16"];
  const tierEntries = requiredTiers.map((tier) => tiers[tier]);
  let tierPhaseFlip;
  if (tierEntries.some((entry) => entry?.status !== "measured")) {
    tierPhaseFlip = {
      status: "insufficient_data",
      tiers: requiredTiers,
      verdict: "open",
      reason: "measured q4 and bf16 staged-residency single-pass selectors are required",
      matchedGeometries: [],
    };
  } else {
    const geometryKey = (binding) => JSON.stringify([
      binding.geometry.width,
      binding.geometry.height,
      binding.geometry.frames,
      binding.geometry.fps,
    ]);
    const q4Bindings = new Map(tiers.q4.bindings.map((binding) => [geometryKey(binding), binding]));
    const bf16Bindings = new Map(tiers.bf16.bindings.map((binding) => [geometryKey(binding), binding]));
    const matchedGeometries = [...q4Bindings]
      .filter(([key]) => bf16Bindings.has(key))
      .map(([key, q4]) => {
        const bf16 = bf16Bindings.get(key);
        const ambiguous = q4.bindingPhase === null || bf16.bindingPhase === null;
        return {
          geometry: q4.geometry,
          q4BindingPhase: q4.bindingPhase,
          bf16BindingPhase: bf16.bindingPhase,
          ...(ambiguous ? { ambiguous: true } : { differs: q4.bindingPhase !== bf16.bindingPhase }),
        };
      })
      .sort((left, right) => compareText(JSON.stringify(left.geometry), JSON.stringify(right.geometry)));
    const ambiguous = matchedGeometries.filter((entry) => entry.ambiguous).length;
    const differences = matchedGeometries.filter((entry) => entry.differs).length;
    tierPhaseFlip = matchedGeometries.length === 0 || ambiguous > 0
      ? {
          status: "insufficient_data",
          tiers: requiredTiers,
          verdict: "open",
          reason: matchedGeometries.length === 0
            ? "q4 and bf16 have no matched scored geometries"
            : "one or more matched geometries has an ambiguous binding phase",
          matchedGeometries,
        }
      : {
          status: "measured",
          tiers: requiredTiers,
          verdict: differences > 0
            ? "tier_phase_flip_observed_at_matched_geometry"
            : "no_tier_phase_flip_observed_at_matched_geometry",
          differingGeometries: differences,
          matchedGeometries,
        };
  }
  return {
    status: tierPhaseFlip.status,
    distinction: "cross-tier binding differences and within-tier geometry flips are reported separately",
    tierPhaseFlip,
    tiers,
  };
}

export function buildReport(
  points,
  plan = null,
  driverStates = new Map(),
  sourceSessions = [],
  story = "sc-18810",
) {
  const orderedPoints = points
    .slice()
    .sort((left, right) => compareText(left.recordId ?? "", right.recordId ?? ""));
  const tiers = [...new Set(orderedPoints.map((point) => point.tier))].sort();
  const grouped = new Map();
  for (const point of orderedPoints) {
    const selector = completeSelectorOfPoint(point);
    const key = JSON.stringify(Object.values(selector));
    if (!grouped.has(key)) grouped.set(key, { key, selector, points: [] });
    grouped.get(key).points.push(point);
  }
  const completeSelectorFits = [...grouped.values()]
    .map(({ key, selector, points: scopedPoints }) => ({
      key,
      selector,
      recordIds: scopedPoints.map((point) => point.recordId).sort(compareText),
      points: scopedPoints,
      fits: selectorFitReport(scopedPoints),
    }))
    .sort((left, right) => compareText(left.key, right.key));

  // Preserve the v1 tier-indexed view only where a tier maps to exactly one complete selector.
  // In a mixed-rung/provider campaign, omitting that tier is more truthful than pooling records
  // across selectors. `selectorFits` is the canonical source for every promoted runtime curve.
  const selectorFitsByTier = new Map();
  for (const entry of completeSelectorFits) {
    if (!selectorFitsByTier.has(entry.selector.tier)) selectorFitsByTier.set(entry.selector.tier, []);
    selectorFitsByTier.get(entry.selector.tier).push(entry);
  }
  const bySeries = Object.fromEntries(Object.keys(SERIES).map((series) => [series, {}]));
  const legacyFitsOmittedForTiers = [];
  for (const tier of tiers) {
    const entries = selectorFitsByTier.get(tier) ?? [];
    if (entries.length !== 1) {
      legacyFitsOmittedForTiers.push(tier);
      continue;
    }
    for (const series of Object.keys(SERIES)) {
      if (entries[0].fits[series]) bySeries[series][tier] = entries[0].fits[series];
    }
  }
  // Avoid duplicating the large residual tables on the ordinary one-selector-per-tier artifact:
  // that selector references the exact legacy slice just populated above. Only ambiguous tiers
  // carry their fit inline, because no truthful tier-only slice exists for them.
  const coefficientTransfer = heldOutCoefficientTransfer(completeSelectorFits);
  const phaseFlip = phaseFlipVerdict(completeSelectorFits);
  const selectorFits = completeSelectorFits.map((entry) => {
    const { points: _points, ...persisted } = entry;
    if ((selectorFitsByTier.get(entry.selector.tier) ?? []).length !== 1) return persisted;
    const { fits: _fits, ...identity } = persisted;
    return { ...identity, legacyFitTier: entry.selector.tier };
  });
  const noiseFloors = Object.fromEntries(
    Object.keys(SERIES).map((series) => [
      series,
      noiseFloor(orderedPoints.map((point) => ({ ...point, value: point.series[series] }))),
    ]),
  );
  return {
    schemaVersion: 1,
    story,
    generatedBy: "scripts/fit-ltx-temporal-form.mjs",
    capturedRecords: orderedPoints.length,
    tiers,
    // Which driver session produced which record, and under which SceneWorks revision. The evidence
    // BUNDLE's own `sourceSessions` is `[]` for this lane (as it is for sc-18808): the harness only
    // populates it on its capture path, and these records were ingested without it. This block is
    // the provenance that does exist, derived from the committed logs rather than typed.
    sourceSessions,
    ...(plan ? { coverage: coverageOf(plan, orderedPoints, driverStates) } : {}),
    noiseFloors,
    selectorFits,
    ...(story === "sc-18810" ? {} : { coefficientTransfer, phaseFlip }),
    ...(legacyFitsOmittedForTiers.length > 0 ? { legacyFitsOmittedForTiers } : {}),
    observations: orderedPoints
      .map(persistedObservation)
      .sort(
        (left, right) =>
          compareText(left.fixture, right.fixture) ||
          compareText(left.sceneWorksRevision, right.sceneWorksRevision) ||
          compareText(left.recordId, right.recordId),
      ),
    fits: bySeries,
  };
}

function hullCross(origin, left, right) {
  return (
    (left.pixels - origin.pixels) * (right.voxels - origin.voxels) -
    (left.voxels - origin.voxels) * (right.pixels - origin.pixels)
  );
}

/**
 * Convex hull in the exact regressor plane the adopted curve evaluates in:
 * `(pixels, pixels * frames)`. Keeping integer coordinates avoids a floating-point boundary gap
 * between generation and consumption. Collinear interior points are discarded; boundary vertices
 * remain ordered counter-clockwise and the evaluator accepts their edges.
 */
export function geometryHull(geometries) {
  const points = [
    ...new Map(
      geometries.map(({ width, height, frames }) => {
        if (
          !Number.isSafeInteger(width) || width <= 0 ||
          !Number.isSafeInteger(height) || height <= 0 ||
          !Number.isSafeInteger(frames) || frames <= 0
        ) {
          throw new Error("fitted video curve geometry requires positive safe integers");
        }
        const pixels = width * height;
        const voxels = pixels * frames;
        if (!Number.isSafeInteger(pixels) || !Number.isSafeInteger(voxels)) {
          throw new Error("fitted video curve geometry exceeds exact JSON integer arithmetic");
        }
        return [`${pixels}:${voxels}`, { pixels, voxels }];
      }),
    ).values(),
  ].sort((left, right) => left.pixels - right.pixels || left.voxels - right.voxels);
  if (points.length < 3) {
    throw new Error(`a fitted video curve needs at least three distinct geometry points, got ${points.length}`);
  }
  const half = (ordered) => {
    const result = [];
    for (const point of ordered) {
      while (
        result.length >= 2 &&
        hullCross(result[result.length - 2], result[result.length - 1], point) <= 0
      ) {
        result.pop();
      }
      result.push(point);
    }
    return result;
  };
  const lower = half(points);
  const upper = half(points.slice().reverse());
  const hull = [...lower.slice(0, -1), ...upper.slice(0, -1)];
  if (hull.length < 3) throw new Error("the measured geometry hull is collinear");
  return hull;
}

/**
 * Promote the ratified per-phase `cross` form into the backend-neutral runtime container owned by
 * sc-19020. Records are partitioned by the COMPLETE runtime selector before fitting: a campaign may
 * capture several tiers/rungs/providers/sources, but no regression is ever allowed to cross one of
 * those boundaries. Each curve names the exact immutable record subset it consumed, partitioned by
 * source path and bound to the digest of that source's exact committed bytes.
 */
export function buildVideoMemoryCurveBundle(
  report,
  records,
  manifest,
  sourceEvidenceInput,
  sourceFit = "docs/generated/ltx-temporal-form-fit-sc-18810.json",
) {
  if (!Array.isArray(records) || records.length === 0) {
    throw new Error("the video curve source dataset has no records");
  }
  const recordIds = records.map((record) => record.id);
  if (recordIds.some((id) => !/^imc-[0-9a-f]{20}$/.test(id)) || new Set(recordIds).size !== records.length) {
    throw new Error("video curve source records need unique immutable imc ids");
  }
  const sourceInputs = sourceEvidenceInput;
  if (!Array.isArray(sourceInputs) || sourceInputs.length === 0) {
    throw new Error("video curve generation requires one or more exact source evidence inputs");
  }
  const sourceByRecord = new Map();
  const sourceCatalog = sourceInputs
    .map((source) => {
      if (
        typeof source?.path !== "string" ||
        !isCanonicalRepoPath(source.path) ||
        typeof source.raw !== "string"
      ) {
        throw new Error(
          "each video curve source requires a canonical repository-relative path and exact raw bytes",
        );
      }
      const parsed = JSON.parse(source.raw);
      if (!Array.isArray(parsed.records) || parsed.records.length === 0) {
        throw new Error(`${source.path} has no non-empty records array`);
      }
      const sha256 = createHash("sha256").update(source.raw).digest("hex");
      for (const record of parsed.records) {
        if (sourceByRecord.has(record.id)) throw new Error(`record ${record.id} appears in multiple evidence sources`);
        sourceByRecord.set(record.id, { path: source.path, sha256, record });
      }
      return { path: source.path, sha256 };
    })
    .sort((left, right) => compareText(left.path, right.path));
  if (!sourceCatalog.every((source, index) => index === 0 || sourceCatalog[index - 1].path < source.path)) {
    throw new Error("video curve evidence source paths must be unique");
  }
  for (const record of records) {
    const source = sourceByRecord.get(record.id);
    if (!source || !isDeepStrictEqual(source.record, record)) {
      throw new Error(`source evidence bytes do not contain promoted record ${record.id} exactly`);
    }
  }
  if (sourceByRecord.size !== records.length) {
    throw new Error("source evidence contains records outside the promoted input set");
  }

  if (
    report?.generatedBy !== "scripts/fit-ltx-temporal-form.mjs" ||
    report.capturedRecords !== records.length ||
    !Array.isArray(report.observations) ||
    report.observations.length !== records.length
  ) {
    throw new Error("fit report does not describe the exact promoted record set");
  }
  const observationById = new Map(
    report.observations.map((observation) => [observation.recordId, observation]),
  );
  if (observationById.size !== records.length || records.some((record) => !observationById.has(record.id))) {
    throw new Error("fit report observations do not match the immutable promoted record ids");
  }
  for (const record of records) {
    const observation = observationById.get(record.id);
    if (typeof observation.role !== "string" || observation.role.length === 0) {
      throw new Error(`fit report observation ${record.id} has no declared plan role`);
    }
    const [expectedPoint] = pointsFrom(
      [record],
      new Map([[record.fixture, observation.role]]),
      manifest,
    );
    if (!isDeepStrictEqual(observation, persistedObservation(expectedPoint))) {
      throw new Error(
        `fit report observation ${record.id} does not match its immutable source record`,
      );
    }
  }
  const selectorOf = (record) => {
    const measurements = Object.fromEntries(record.diagnostics.measurements.map((entry) => [entry.name, entry.value]));
    const decodeTilingEngaged = measurements.decodeTilingEngaged;
    if (decodeTilingEngaged !== 0 && decodeTilingEngaged !== 1) {
      throw new Error(
        `record ${record.id} needs an exact decodeTilingEngaged measurement of 0 or 1`,
      );
    }
    const catalogModel = manifest.models.find((model) => model.id === record.target.modelId);
    if (!catalogModel) throw new Error(`model ${record.target.modelId} is absent from builtin.models.jsonc`);
    const referenceCount = record.target.referenceCount ?? (record.target.mode === "text_to_video" ? 0 : null);
    const referenceShape = record.target.referenceShape ?? (referenceCount === 0 ? "none" : null);
    const outputFps = record.diagnostics.measurements.find((entry) => entry.name === "outputFps")?.value;
    if (!Number.isInteger(referenceCount) || referenceCount < 0 || typeof referenceShape !== "string" || referenceShape.length === 0 || (referenceShape === "none") !== (referenceCount === 0) || !Number.isInteger(outputFps) || outputFps < 1) {
      throw new Error(`record ${record.id} has an incomplete reference/FPS evidence identity`);
    }
    return {
      modelId: record.target.modelId,
      modelFamily: catalogModel.family,
      route: record.target.route ?? record.target.provider,
      provider: record.target.provider,
      backend: record.backend,
      tier: record.target.tier,
      mode: record.target.mode,
      referenceShape,
      referenceCount,
      overlay: record.target.overlay === "none" ? null : record.target.overlay,
      rung: record.strategy.rung,
      loadShape: record.loadShape,
      batch: record.target.geometry.batch,
      closureDigest: record.repositories.inference.closureDigest,
      calibrationAbi: VIDEO_MEMORY_CURVE_CALIBRATION_ABI,
      calibrationFingerprint: record.calibrationFingerprint,
      decodePass: decodeTilingEngaged === 1 ? "tiled" : "single_pass",
    };
  };
  const groups = new Map();
  for (const record of records) {
    const selector = selectorOf(record);
    const key = JSON.stringify(Object.values(selector));
    if (!groups.has(key)) groups.set(key, { selector, records: [] });
    groups.get(key).records.push(record);
  }
  if (!Array.isArray(report.selectorFits) || report.selectorFits.length !== groups.size) {
    throw new Error("fit report does not contain one fit for every complete selector");
  }
  const reportedFits = new Map();
  for (const entry of report.selectorFits) {
    const key = JSON.stringify(Object.values(entry?.selector ?? {}));
    const hasInlineFits = entry?.fits && typeof entry.fits === "object";
    const hasLegacyFit = typeof entry?.legacyFitTier === "string";
    if (
      entry?.key !== key ||
      reportedFits.has(key) ||
      hasInlineFits === hasLegacyFit ||
      (hasLegacyFit && entry.legacyFitTier !== entry.selector?.tier)
    ) {
      throw new Error("fit report contains a malformed or duplicate complete selector");
    }
    reportedFits.set(key, entry);
  }

  const curves = [...groups.entries()].map(([selectorKey, { selector, records: scopedRecords }]) => {
    scopedRecords.sort((left, right) => compareText(left.id, right.id));
    const {
      modelId,
      modelFamily,
      route,
      provider,
      backend,
      tier,
      mode,
      referenceShape,
      referenceCount,
      overlay,
      rung,
      loadShape,
      batch,
      closureDigest,
      calibrationAbi,
      calibrationFingerprint,
      decodePass,
    } = selector;
    if (!/^[0-9a-f]{64}$/.test(closureDigest)) {
      throw new Error(`inference closure digest is not sha256: ${closureDigest}`);
    }
    const observations = scopedRecords.map((record) => observationById.get(record.id));
    const reported = reportedFits.get(selectorKey);
    if (
      !reported ||
      !isDeepStrictEqual(reported.selector, selector) ||
      !isDeepStrictEqual(reported.recordIds, scopedRecords.map((record) => record.id))
    ) {
      throw new Error(`${modelId}/${tier}/${rung} fit report record subset is detached`);
    }
    const reportedSelectorFits = reported.fits ?? (
      reported.legacyFitTier === tier
        ? Object.fromEntries(
          Object.keys(SERIES)
            .filter((series) => report.fits?.[series]?.[tier])
            .map((series) => [series, report.fits[series][tier]]),
        )
        : null
    );
    if (!reportedSelectorFits) {
      throw new Error(`${modelId}/${tier}/${rung} fit report has no selector-scoped fit`);
    }
    const phase = (series) => {
      const candidate = reportedSelectorFits[series]?.candidates?.cross;
      if (!candidate || candidate.singular) {
        throw new Error(`missing non-singular ${series}/${tier} cross fit`);
      }
      const coefficients = candidate.coefficients;
      const keys = Object.keys(coefficients);
      if (JSON.stringify(keys) !== JSON.stringify(["fixedGb", "perMpxGb", "perMpxFrameGb"])) {
        throw new Error(
          `${series}/${tier} cross coefficients have unexpected shape ${keys.join(",")}`,
        );
      }
      for (const [name, value] of Object.entries(coefficients)) {
        if (!Number.isFinite(value) || value < 0) {
          throw new Error(`${series}/${tier} ${name} must be finite and non-negative`);
        }
      }
      if (
        !Number.isFinite(candidate.fit.maxAbsGib) ||
        !Number.isFinite(candidate.heldOut.maxAbsGib)
      ) {
        throw new Error(`${series}/${tier} cross fit needs fit and held-out residual bounds`);
      }
      return {
        ...coefficients,
        maxResidualGb: Math.max(candidate.fit.maxAbsGib, candidate.heldOut.maxAbsGib),
      };
    };
    const sourceGroups = new Map();
    for (const record of scopedRecords) {
      const source = sourceByRecord.get(record.id);
      const key = `${source.path}\0${source.sha256}`;
      if (!sourceGroups.has(key)) {
        sourceGroups.set(key, { path: source.path, sha256: source.sha256, recordIds: [] });
      }
      sourceGroups.get(key).recordIds.push(record.id);
    }
    const evidenceSources = [...sourceGroups.values()]
      .map((source) => ({ ...source, recordIds: source.recordIds.sort() }))
      .sort((left, right) => compareText(left.path, right.path));
    const fitPoints = observations.filter((observation) => observation.role === "fit").length;
    const heldOutPoints = observations.filter(
      (observation) => observation.role.startsWith("held_out"),
    ).length;
    if (fitPoints + heldOutPoints !== scopedRecords.length) {
      const unscored = observations
        .filter(
          (observation) =>
            observation.role !== "fit" && !observation.role.startsWith("held_out"),
        )
        .map((observation) => `${observation.recordId}:${observation.role}`)
        .sort();
      throw new Error(
        `${modelId}/${tier}/${rung} contains records outside the fitted or held-out subsets: ${unscored.join(", ")}`,
      );
    }
    const framesPerSecond = [
      ...new Set(
        scopedRecords.map((record) =>
          record.diagnostics.measurements.find((entry) => entry.name === "outputFps")?.value,
        ),
      ),
    ].sort((a, b) => a - b);
    return {
      // Keep the human-readable id bijective with the complete selector. Runtime also validates
      // every field independently; the id is not an authorization shortcut.
      id: `${modelId}:${modelFamily}:${route}:${provider}:${backend}:${tier}:${mode}:ref${referenceShape}-${referenceCount}:fps${framesPerSecond.join("+")}:${overlay ?? "none"}:${rung}:${loadShape}:b${batch}:abi${calibrationAbi}:${decodePass}:${closureDigest.slice(0, 12)}:${calibrationFingerprint}`,
      modelId,
      modelFamily,
      route,
      provider,
      backend,
      tier,
      mode,
      referenceShape,
      referenceCount,
      framesPerSecond,
      overlay,
      rung,
      loadShape,
      batch,
      closureDigest,
      calibrationAbi,
      calibrationFingerprint,
      decodePass,
      measuredGeometryHull: geometryHull(observations.map((observation) => observation.geometry)),
      phases: { conditioning: phase("text"), denoise: phase("denoise"), decode: phase("decode") },
      evidence: {
        records: scopedRecords.length,
        fitPoints,
        heldOutPoints,
        sources: evidenceSources,
      },
    };
  });
  curves.sort((left, right) => compareText(left.id, right.id));

  return {
    schemaVersion: 3,
    generatedBy: "scripts/fit-ltx-temporal-form.mjs",
    sourceFit,
    sourceCatalog,
    curves,
  };
}

async function readJson(file) {
  return JSON.parse(await readFile(file, "utf8"));
}

/** Roles live in the sweep PLAN, not in the records. */
export function rolesFromPlan(plan) {
  return new Map(plan.providers.map((provider) => [provider.fixture, provider._role]));
}

async function main() {
  const args = process.argv.slice(2);
  const value = (flag, fallback) => {
    const index = args.indexOf(flag);
    return index < 0 ? fallback : args[index + 1];
  };
  const repeated = (flag, fallback) => {
    const found = args.flatMap((arg, index) => (arg === flag ? [args[index + 1]] : []));
    return found.length > 0 ? found : fallback;
  };
  const story = value("--story", "sc-18810");
  if (!/^sc-[0-9]+$/.test(story)) throw new Error(`--story must be sc-<digits>, got ${story}`);
  const datasetPaths = repeated("--dataset", [
    story === "sc-18810"
      ? "docs/generated/ltx-mlx-geometry-sweep-sc-18810.json"
      : `docs/generated/ltx-mlx-single-pass-${story}.json`,
  ]).map((relative) => path.resolve(ROOT, relative));
  const planPath = path.resolve(
    ROOT,
    value(
      "--plan",
      story === "sc-18810"
        ? "docs/calibration/sc-18810/ltx-mlx-geometry-sweep.json"
        : `docs/calibration/${story}/ltx-mlx-single-pass-sweep.json`,
    ),
  );
  const reportPath = path.resolve(
    ROOT,
    value("--write", `docs/generated/ltx-temporal-form-fit-${story}.json`),
  );
  const curvePath = path.resolve(
    ROOT,
    value("--curve-write", "docs/generated/video-memory-curves.json"),
  );
  const sourceFit = value("--source-fit", `docs/generated/ltx-temporal-form-fit-${story}.json`);
  if (!isCanonicalRepoPath(sourceFit)) {
    throw new Error(`--source-fit must be a canonical repository-relative path, got ${sourceFit}`);
  }
  const manifestPath = path.resolve(ROOT, "config/manifests/builtin.models.jsonc");
  const curveSchemaPath = path.resolve(ROOT, "packages/schemas/video-memory-curves.schema.json");
  // BOTH driver sessions, in chronological order. The first crashed the host after four captures
  // and its log went uncommitted in the original PR, which is what let four of the thirteen records
  // ship with no terminal line anywhere. `--driver-log` may be repeated.
  const driverLogPaths = repeated(
    "--driver-log",
    story === "sc-18810"
      ? [
          "docs/calibration/sc-18810/precrash-q8-run.log",
          "docs/calibration/sc-18810/sweep-run.log",
        ]
      : [],
  ).map((relative) => path.resolve(ROOT, relative));
  const datasets = await Promise.all(
    datasetPaths.map(async (file) => {
      const raw = await readFile(file, "utf8");
      return { path: repoRelativePath(file), raw, parsed: JSON.parse(raw) };
    }),
  );
  const records = datasets.flatMap(({ parsed }) => parsed.records);
  const plan = await readJson(planPath);
  const manifest = JSON.parse(stripJsoncComments(await readFile(manifestPath, "utf8")));
  const curveSchema = await readJson(curveSchemaPath);
  const logs = await Promise.all(
    driverLogPaths.map(async (file) => ({
      path: repoRelativePath(file),
      text: await readFile(file, "utf8"),
    })),
  );
  // Which geometries were ATTEMPTED comes from the drivers' own logs, not from a hand-typed list.
  const driverStates = driverStatesFrom(logs.map((log) => log.text));
  const points = pointsFrom(records, rolesFromPlan(plan), manifest);
  const fixtureByName = new Map(
    plan.providers.map((provider) => [provider.name, provider.fixture]),
  );
  const report = buildReport(
    points,
    plan,
    driverStates,
    sessionsFrom(logs, points, fixtureByName),
    story,
  );
  const serialised = `${JSON.stringify(report, null, 2)}\n`;
  const curveBundle = buildVideoMemoryCurveBundle(
    report,
    records,
    manifest,
    datasets,
    sourceFit,
  );
  const curveSchemaProblems = videoCurveSchemaErrors(curveSchema, curveBundle);
  if (curveSchemaProblems.length > 0) {
    throw new Error(
      `${repoRelativePath(curvePath)} violates ${repoRelativePath(curveSchemaPath)}:\n${curveSchemaProblems.join("\n")}`,
    );
  }
  const serialisedCurves = `${JSON.stringify(curveBundle, null, 2)}\n`;
  if (args.includes("--check")) {
    const [existing, existingCurves] = await Promise.all([
      readFile(reportPath, "utf8"),
      readFile(curvePath, "utf8"),
    ]);
    if (existing !== serialised || existingCurves !== serialisedCurves) {
      const stale = [
        ...(existing !== serialised ? [repoRelativePath(reportPath)] : []),
        ...(existingCurves !== serialisedCurves ? [repoRelativePath(curvePath)] : []),
      ];
      process.stderr.write(
        `${stale.join(", ")} is stale — re-run scripts/fit-ltx-temporal-form.mjs\n`,
      );
      process.exitCode = 1;
      return;
    }
    process.stdout.write(
      `${repoRelativePath(reportPath)} and ${repoRelativePath(curvePath)} are current\n`,
    );
    return;
  }
  await Promise.all([writeFile(reportPath, serialised), writeFile(curvePath, serialisedCurves)]);
  process.stdout.write(
    `wrote ${repoRelativePath(reportPath)} and ${repoRelativePath(curvePath)}\n`,
  );
  for (const [series, tiers] of Object.entries(report.fits)) {
    for (const [tier, slice] of Object.entries(tiers)) {
      const rows = Object.entries(slice.candidates)
        .map(([name, candidate]) =>
          candidate.singular
            ? `${name}=singular`
            : `${name} fit=${candidate.fit.maxAbsGib.toFixed(3)} held=${
                candidate.heldOut.maxAbsGib === null
                  ? "n/a"
                  : candidate.heldOut.maxAbsGib.toFixed(3)
              }`,
        )
        .join("  ");
      process.stdout.write(`${series} ${tier}: chosen=${slice.chosen}  ${rows}\n`);
    }
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  await main();
}
