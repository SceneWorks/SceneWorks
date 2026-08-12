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
 * Usage: node scripts/fit-ltx-temporal-form.mjs [--dataset <bundle.json>] [--plan <plan.json>]
 *                                               [--driver-log <sweep-run.log>]
 *                                               [--write <report.json>] [--check]
 */
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const GIB = 1024 ** 3;

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
 */
export function noiseFloor(points) {
  const groups = new Map();
  for (const point of points) {
    if (!groups.has(point.replicateKey)) groups.set(point.replicateKey, []);
    groups.get(point.replicateKey).push(point.value);
  }
  const spreads = [...groups.entries()]
    .filter(([, values]) => values.length > 1)
    .map(([key, values]) => ({
      key,
      replicates: values.length,
      meanGib: values.reduce((sum, value) => sum + value, 0) / values.length,
      spreadGib: Math.max(...values) - Math.min(...values),
    }));
  if (spreads.length === 0) {
    return { replicatedGeometries: 0, maxSpreadGib: null, maxSpreadFraction: null, spreads: [] };
  }
  return {
    replicatedGeometries: spreads.length,
    maxSpreadGib: Math.max(...spreads.map((entry) => entry.spreadGib)),
    maxSpreadFraction: Math.max(...spreads.map((entry) => entry.spreadGib / entry.meanGib)),
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

export function pointsFrom(records, roleByFixture) {
  return records.map((record) => {
    const { width, height, frames } = record.target.geometry;
    const measurements = Object.fromEntries(
      record.diagnostics.measurements.map((entry) => [entry.name, entry.value]),
    );
    const fps = measurements.outputFps;
    const role = roleByFixture.get(record.fixture);
    if (!role) throw new Error(`record fixture ${record.fixture} has no role in the sweep plan`);
    return {
      fixture: record.fixture,
      tier: record.target.tier,
      role,
      rung: record.strategy.rung,
      decodeTilingEngaged: measurements.decodeTilingEngaged === 1,
      sceneWorksRevision: record.repositories.sceneWorks.revision,
      replicateKey: `${record.target.tier}:${width}x${height}:f${frames}:fps${fps}`,
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

/**
 * The driver's own terminal state per plan entry, parsed from the committed sweep log
 * (`docs/calibration/sc-18810/sweep-run.log`, written verbatim by the sweep driver — see its
 * `BEGIN`/`OK`/`FAIL`/`STOP` writes).
 *
 * This exists because the previous version of this script carried a HARDCODED two-element list of
 * "attempted and killed" fixtures, and it was wrong: `1280x704 f241` was reported
 * `not_attempted_host_limit` while the log shows `BEGIN ... free=16GiB 11:40:16` and no terminal
 * line at all. A coverage state that is typed by hand is a claim about the run that nothing checks.
 *
 * Terminal states, all four of which the log distinguishes:
 *   `completed`           — the driver wrote `OK`.
 *   `failed`              — the driver wrote `FAIL`; the child exited non-zero or was killed.
 *   `no_terminal_record`  — a `BEGIN` with no `OK`/`FAIL`. The driver ITSELF did not survive to
 *                           write one. Attempted, and it did not survive; not "never run".
 *   `not_begun`           — no `BEGIN` line. `stoppedBefore` marks the entry the driver named when
 *                           it halted on the free-disk floor.
 * An entry re-run after a failure (`704x1280 f177` was refused by the staged-residency guard, then
 * re-run and captured) counts as `completed`: one `OK` is enough.
 */
export function driverStatesFrom(logText) {
  const states = new Map();
  const of = (name) => {
    if (!states.has(name)) {
      states.set(name, { begins: 0, oks: 0, fails: 0, stoppedBefore: false });
    }
    return states.get(name);
  };
  for (const line of logText.split("\n")) {
    let match;
    if ((match = /^BEGIN (\S+) /.exec(line))) of(match[1]).begins += 1;
    else if ((match = /^OK (\S+) /.exec(line))) of(match[1]).oks += 1;
    else if ((match = /^FAIL (\S+) /.exec(line))) of(match[1]).fails += 1;
    else if ((match = /^STOP .* before (\S+)\s*$/.exec(line))) of(match[1]).stoppedBefore = true;
  }
  for (const state of states.values()) {
    state.terminal =
      state.oks > 0
        ? "completed"
        : state.begins === 0
          ? "not_begun"
          : state.fails > 0
            ? "failed"
            : "no_terminal_record";
  }
  return states;
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
 * defect this function now throws on.
 */
export function coverageOf(plan, points, driverStates = new Map()) {
  const captured = new Map();
  for (const point of points) {
    captured.set(point.fixture, (captured.get(point.fixture) ?? 0) + 1);
  }
  const entries = plan.providers.map((provider) => {
    const terminal = driverStates.get(provider.name)?.terminal ?? "not_begun";
    const attempted = terminal !== "not_begun";
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
    return {
      fixture: provider.fixture,
      role: provider._role,
      tier: provider.target.tier,
      outputVoxels: provider._outputVoxels,
      latentTokens: provider._latentTokens,
      driverTerminalState: terminal,
      state: captured.has(provider.fixture)
        ? "captured"
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
              : "not_reached",
      captures: captured.get(provider.fixture) ?? 0,
    };
  });
  const byState = {};
  for (const entry of entries) byState[entry.state] = (byState[entry.state] ?? 0) + 1;
  return {
    plannedEntries: entries.length,
    capturedGeometries: new Set(
      points.map((point) => {
        const { width, height, frames, fps } = point.geometry;
        return `${width}x${height}:f${frames}:fps${fps}`;
      }),
    ).size,
    byState,
    entries,
  };
}

export function buildReport(points, plan = null, driverStates = new Map()) {
  const tiers = [...new Set(points.map((point) => point.tier))].sort();
  const bySeries = {};
  for (const series of Object.keys(SERIES)) {
    bySeries[series] = {};
    for (const tier of tiers) {
      const scoped = points
        .filter((point) => point.tier === tier)
        .map((point) => ({ ...point, value: point.series[series] }));
      // `role` decides membership and lives in the committed PLAN, not in the records, so the
      // fit/held-out split cannot be redrawn after seeing residuals. A replicate of a fit geometry
      // is training data; a replicate of a held-out geometry stays held out.
      const fitPoints = scoped.filter((point) => point.role === "fit");
      const heldOutPoints = scoped.filter((point) => point.role.startsWith("held_out"));
      // Everything else is carried but NOT scored. `rung2_boundary` points decode through a TILED
      // VAE pass — a different rung and a different memory regime, so scoring them against a rung-1
      // curve would report a capability change as fit error. `reproduction_probe` exists to compare
      // against a withdrawn external number, not to test the form.
      const unscoredPoints = scoped.filter(
        (point) => point.role !== "fit" && !point.role.startsWith("held_out"),
      );
      if (fitPoints.length === 0) continue;
      const slice = fitSlice(fitPoints, heldOutPoints);
      bySeries[series][tier] = {
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
  }
  const noiseFloors = Object.fromEntries(
    Object.keys(SERIES).map((series) => [
      series,
      noiseFloor(points.map((point) => ({ ...point, value: point.series[series] }))),
    ]),
  );
  return {
    schemaVersion: 1,
    story: "sc-18810",
    generatedBy: "scripts/fit-ltx-temporal-form.mjs",
    capturedRecords: points.length,
    tiers,
    ...(plan ? { coverage: coverageOf(plan, points, driverStates) } : {}),
    noiseFloors,
    observations: points
      .map((point) => ({
        fixture: point.fixture,
        tier: point.tier,
        role: point.role,
        rung: point.rung,
        decodeTilingEngaged: point.decodeTilingEngaged,
        sceneWorksRevision: point.sceneWorksRevision,
        geometry: point.geometry,
        activeGib: point.series,
        cacheGib: point.cacheGib,
      }))
      .sort(
        (left, right) =>
          left.fixture.localeCompare(right.fixture) ||
          left.sceneWorksRevision.localeCompare(right.sceneWorksRevision),
      ),
    fits: bySeries,
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
  const datasetPath = path.resolve(
    ROOT,
    value("--dataset", "docs/generated/ltx-mlx-geometry-sweep-sc-18810.json"),
  );
  const planPath = path.resolve(
    ROOT,
    value("--plan", "docs/calibration/sc-18810/ltx-mlx-geometry-sweep.json"),
  );
  const reportPath = path.resolve(
    ROOT,
    value("--write", "docs/generated/ltx-temporal-form-fit-sc-18810.json"),
  );
  const driverLogPath = path.resolve(
    ROOT,
    value("--driver-log", "docs/calibration/sc-18810/sweep-run.log"),
  );
  const dataset = await readJson(datasetPath);
  const plan = await readJson(planPath);
  // Which geometries were ATTEMPTED comes from the driver's own log, not from a hand-typed list.
  const driverStates = driverStatesFrom(await readFile(driverLogPath, "utf8"));
  const report = buildReport(pointsFrom(dataset.records, rolesFromPlan(plan)), plan, driverStates);
  const serialised = `${JSON.stringify(report, null, 2)}\n`;
  if (args.includes("--check")) {
    const existing = await readFile(reportPath, "utf8");
    if (existing !== serialised) {
      process.stderr.write(
        `${path.relative(ROOT, reportPath)} is stale — re-run scripts/fit-ltx-temporal-form.mjs\n`,
      );
      process.exitCode = 1;
      return;
    }
    process.stdout.write(`${path.relative(ROOT, reportPath)} is current\n`);
    return;
  }
  await writeFile(reportPath, serialised);
  process.stdout.write(`wrote ${path.relative(ROOT, reportPath)}\n`);
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
