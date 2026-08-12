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
      capturedAt: record.capturedAt,
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
      states.set(name, { begins: 0, oks: 0, fails: 0, stoppedBefore: false });
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
        else state.fails += 1;
      } else if ((match = /^STOP .* before (\S+)\s*$/.exec(line))) {
        of(match[1]).stoppedBefore = true;
      }
    }
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
      .sort((left, right) => left.capturedAt.localeCompare(right.capturedAt))
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

export function buildReport(points, plan = null, driverStates = new Map(), sourceSessions = []) {
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
    // Which driver session produced which record, and under which SceneWorks revision. The evidence
    // BUNDLE's own `sourceSessions` is `[]` for this lane (as it is for sc-18808): the harness only
    // populates it on its capture path, and these records were ingested without it. This block is
    // the provenance that does exist, derived from the committed logs rather than typed.
    sourceSessions,
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
  const repeated = (flag, fallback) => {
    const found = args.flatMap((arg, index) => (arg === flag ? [args[index + 1]] : []));
    return found.length > 0 ? found : fallback;
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
  // BOTH driver sessions, in chronological order. The first crashed the host after four captures
  // and its log went uncommitted in the original PR, which is what let four of the thirteen records
  // ship with no terminal line anywhere. `--driver-log` may be repeated.
  const driverLogPaths = repeated("--driver-log", [
    "docs/calibration/sc-18810/precrash-q8-run.log",
    "docs/calibration/sc-18810/sweep-run.log",
  ]).map((relative) => path.resolve(ROOT, relative));
  const dataset = await readJson(datasetPath);
  const plan = await readJson(planPath);
  const logs = await Promise.all(
    driverLogPaths.map(async (file) => ({
      path: path.relative(ROOT, file),
      text: await readFile(file, "utf8"),
    })),
  );
  // Which geometries were ATTEMPTED comes from the drivers' own logs, not from a hand-typed list.
  const driverStates = driverStatesFrom(logs.map((log) => log.text));
  const points = pointsFrom(dataset.records, rolesFromPlan(plan));
  const fixtureByName = new Map(
    plan.providers.map((provider) => [provider.name, provider.fixture]),
  );
  const report = buildReport(
    points,
    plan,
    driverStates,
    sessionsFrom(logs, points, fixtureByName),
  );
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
