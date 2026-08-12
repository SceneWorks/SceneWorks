import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  FORMS,
  coverageOf,
  driverStatesFrom,
  fitSlice,
  latentTemporalDepth,
  latentTokens,
  leastSquares,
  noiseFloor,
  pointsFrom,
  rolesFromPlan,
  sessionsFrom,
} from "./fit-ltx-temporal-form.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const PLAN = JSON.parse(
  readFileSync(path.join(ROOT, "docs/calibration/sc-18810/ltx-mlx-geometry-sweep.json"), "utf8"),
);
/** BOTH committed driver sessions, chronological — the capture crashed the host and resumed. */
const LOG_PATHS = [
  "docs/calibration/sc-18810/precrash-q8-run.log",
  "docs/calibration/sc-18810/sweep-run.log",
];
const LOGS = LOG_PATHS.map((relative) => ({
  path: relative,
  text: readFileSync(path.join(ROOT, relative), "utf8"),
}));
const DRIVER_LOGS = LOGS.map((log) => log.text);
const [PRECRASH_LOG, DRIVER_LOG] = DRIVER_LOGS;
const FIXTURE_BY_NAME = new Map(
  PLAN.providers.map((provider) => [provider.name, provider.fixture]),
);
const DATASET = JSON.parse(
  readFileSync(
    path.join(ROOT, "docs/generated/ltx-mlx-geometry-sweep-sc-18810.json"),
    "utf8",
  ),
);

const geometry = (width, height, frames, fps = 30) => ({
  width,
  height,
  frames,
  fps,
  mpx: (width * height) / 1e6,
  tLat: latentTemporalDepth(frames),
  tokens: latentTokens({ width, height, frames }),
});

/** The COMMITTED design, read from the plan rather than restated, so a plan edit that destroys
 * identifiability reds here instead of silently publishing an unsupported coefficient. */
const planned = (tier, role) =>
  PLAN.providers
    .filter((provider) => provider.target.tier === tier && provider._role === role)
    .map((provider) =>
      geometry(
        provider.target.geometry.width,
        provider.target.geometry.height,
        provider.target.geometry.frames,
      ),
    );
const FIT = planned("q8", "fit");
const HELD = planned("q8", "held_out");

const points = (geometries, value, role = "fit") =>
  geometries.map((g, index) => ({
    fixture: `synthetic-${index}-${g.width}x${g.height}-f${g.frames}`,
    role,
    geometry: g,
    value: value(g),
  }));

test("the LTX temporal handles match the x8 causal VAE", () => {
  assert.equal(latentTemporalDepth(97), 13);
  assert.equal(latentTemporalDepth(121), 16);
  assert.equal(latentTemporalDepth(449), 57);
  // 1280x704 -> 40 x 22 latent cells.
  assert.equal(latentTokens({ width: 1280, height: 704, frames: 449 }), 57 * 40 * 22);
});

test("least squares recovers an exactly-linear generator on every candidate form", () => {
  const truth = {
    area_only: (g) => 4 + 11 * g.mpx,
    additive: (g) => 4 + 11 * g.mpx + 0.02 * g.frames,
    cross: (g) => 4 + 11 * g.mpx + 0.03 * g.mpx * g.frames,
    latent_tokens: (g) => 4 + 0.0004 * g.tokens,
    output_voxels: (g) => 4 + 0.03 * g.mpx * g.frames,
  };
  for (const [name, form] of Object.entries(FORMS)) {
    const fitPoints = points(FIT, truth[name]);
    const coefficients = leastSquares(
      fitPoints.map((point) => form.row(point.geometry)),
      fitPoints.map((point) => point.value),
    );
    assert.ok(coefficients, `${name} must not be singular on the sweep design`);
    // Prediction is the contract, not the coefficient labelling: check it on the HELD-OUT lattice.
    for (const g of HELD) {
      const predicted = form.row(g).reduce((sum, x, i) => sum + x * coefficients[i], 0);
      assert.ok(
        Math.abs(predicted - truth[name](g)) < 1e-9,
        `${name} must reproduce its own generator at ${g.width}x${g.height}xf${g.frames}`,
      );
    }
  }
});

test("the committed sweep design is not collinear — every candidate is solvable on it", () => {
  assert.equal(FIT.length, 6, "the fit set is six geometries");
  assert.equal(new Set(FIT.map((g) => g.mpx)).size, 2, "the fit set must cross two area levels");
  assert.ok(HELD.length >= 3, "the held-out set carries a third area and both transpositions");
  for (const [name, form] of Object.entries(FORMS)) {
    assert.ok(
      leastSquares(
        FIT.map((g) => form.row(g)),
        FIT.map(() => 1),
      ),
      `${name} must be solvable on the six fit geometries`,
    );
  }
  // A one-area sweep cannot fit EITHER three-parameter form: with `mpx` constant its column is
  // proportional to the intercept, so both designs are singular and neither an additive nor a cross
  // coefficient is identifiable at all. That is the whole reason the design crosses two spatial
  // levels; pin it so a later narrowing reds instead of silently publishing an unsupported
  // coefficient.
  const oneArea = FIT.filter((g) => g.mpx === FIT[0].mpx);
  for (const name of ["area_only", "additive", "cross"]) {
    assert.equal(
      leastSquares(
        oneArea.map((g) => FORMS[name].row(g)),
        oneArea.map(() => 1),
      ),
      null,
      `${name} must be unidentifiable on a single-area sweep`,
    );
  }
  // The two constrained forms stay identifiable there, because their single regressor still varies
  // with frames — which is why a one-area sweep looks like it "works" right up until someone tries
  // to read a per-area coefficient off it.
  for (const name of ["latent_tokens", "output_voxels"]) {
    assert.equal(
      leastSquares(
        oneArea.map((g) => FORMS[name].row(g)),
        oneArea.map(() => 1),
      ).length,
      2,
      `${name} stays identifiable on a single-area sweep`,
    );
  }
});

test("selection is decided on held-out residuals, not on in-sample fit", () => {
  // Truth is the CONSTRAINED latent-token form. `cross` has a superset of its freedom, so it can
  // never fit the training points worse — yet the rule must still pick the true generator.
  const truth = (g) => 30 + 0.00035 * g.tokens;
  const { candidates, chosen } = fitSlice(points(FIT, truth), points(HELD, truth, "held_out"));
  assert.equal(chosen, "latent_tokens");
  assert.ok(candidates.cross.fit.maxAbsGib <= candidates.latent_tokens.fit.maxAbsGib + 1e-9);
  assert.ok(candidates.area_only.heldOut.maxAbsGib > candidates.latent_tokens.heldOut.maxAbsGib);
});

test("an additive generator is recovered as additive, so the rule is not latent-token biased", () => {
  const truth = (g) => 30 + 5 * g.mpx + 0.01 * g.frames;
  const { chosen } = fitSlice(points(FIT, truth), points(HELD, truth, "held_out"));
  assert.equal(chosen, "additive");
});

test("the noise floor is the replicate spread and is null without replicates", () => {
  assert.equal(
    noiseFloor([
      { replicateKey: "a", value: 1 },
      { replicateKey: "b", value: 2 },
    ]).maxSpreadGib,
    null,
  );
  const floor = noiseFloor([
    { replicateKey: "a", value: 10 },
    { replicateKey: "a", value: 10.25 },
    { replicateKey: "b", value: 3 },
  ]);
  assert.equal(floor.replicatedGeometries, 1);
  assert.ok(Math.abs(floor.maxSpreadGib - 0.25) < 1e-12);
});

test("every fit and held-out point in the plan is a single-pass rung-1 decode", () => {
  // The LTX write bound `i32::MAX / (8*h*w)` is 682 / 682 / 655 / 297 / 297 over the declared
  // resolutions. A fit or held-out point above its own bucket's cap would put a TILED decode into a
  // curve fitted for single-pass decodes, which is fitting through a capability change.
  const cap = (width, height) => Math.floor((2 ** 31 - 1) / (8 * width * height));
  assert.equal(cap(1280, 704), 297);
  assert.equal(cap(768, 512), 682);
  for (const provider of PLAN.providers) {
    const { width, height, frames } = provider.target.geometry;
    const tiled = frames > cap(width, height);
    assert.equal(
      provider.rung,
      tiled ? "bounded_decode" : "staged_residency",
      `${provider.name} must declare the rung its geometry engages`,
    );
    if (provider._role === "fit" || provider._role.startsWith("held_out")) {
      assert.equal(tiled, false, `${provider.name} is scored, so it must be a single-pass decode`);
    }
  }
  assert.ok(
    PLAN.providers.some((provider) => provider._role === "rung2_boundary"),
    "the plan must still bracket the tiling boundary",
  );
});

test("a record with no declared role is refused rather than scored", () => {
  // NOT evidence of pre-registration — that lives in the commit timeline (plan 301fb80e 04:07, fit
  // e8c8353f 08:23, no captured fixture's `_role` touched by the 07:51 amendment) and no unit test
  // can stand in for it. What this pins is narrower and still worth pinning: every scored point's
  // role comes from the PLAN, so a record the plan never declared cannot slip into a fit at all.
  const roles = rolesFromPlan(PLAN);
  assert.equal(roles.size, PLAN.providers.length);
  assert.throws(
    () =>
      pointsFrom(
        [
          {
            fixture: "not-in-the-plan",
            target: { tier: "q8", geometry: { width: 768, height: 512, frames: 121 } },
            strategy: { rung: "staged_residency" },
            repositories: { sceneWorks: { revision: "0" } },
            diagnostics: { measurements: [{ name: "outputFps", value: 30 }] },
            observedMemory: {},
          },
        ],
        roles,
      ),
    /has no role in the sweep plan/,
  );
});

test("the scored roles are exactly the declared vocabulary, and host-outcome labels never score", () => {
  // Binding roles to their declaration: the committed plan may only use these seven labels, the two
  // `*_host_limit` ones are FEASIBILITY labels rather than fit membership, and neither may ever be
  // read as a scored point. A new label — or a `fit` silently renamed — reds here.
  const SCORED = new Set(["fit", "held_out", "held_out_fps_probe"]);
  const HOST_OUTCOME = new Set(["not_attempted_host_limit", "attempted_failed_host_limit"]);
  const DECLARED = new Set([...SCORED, ...HOST_OUTCOME, "reproduction_probe", "rung2_boundary"]);
  for (const provider of PLAN.providers) {
    assert.ok(DECLARED.has(provider._role), `${provider.name} uses undeclared role ${provider._role}`);
    if (HOST_OUTCOME.has(provider._role)) {
      assert.ok(
        !SCORED.has(provider._role) && !provider._role.startsWith("held_out"),
        `${provider.name} is a host-outcome label and must not be scored`,
      );
    }
  }
  // Six fit + three held-out in every tier, plus q8's one fps probe: 28 scorable rows in all.
  for (const tier of ["q8", "bf16", "q4"]) {
    const rows = PLAN.providers.filter((provider) => provider.target.tier === tier);
    assert.equal(rows.filter((provider) => provider._role === "fit").length, 6, `${tier} fit`);
    assert.equal(
      rows.filter((provider) => provider._role === "held_out").length,
      3,
      `${tier} held_out`,
    );
  }
  assert.equal(PLAN.providers.filter((provider) => SCORED.has(provider._role)).length, 28);
});

test("the driver log, not a hardcoded list, says what was attempted", () => {
  const states = driverStatesFrom(DRIVER_LOGS);
  // Every terminal state the log actually distinguishes, on a real row of the committed log.
  assert.equal(states.get("mlx-ltx-2-3-q8-768x512-f121-fps30").terminal, "completed");
  assert.equal(states.get("mlx-ltx-2-3-q8-768x512-f361-fps30").terminal, "failed");
  // The row the previous hardcoded list got wrong: BEGIN at 11:40:16 with free=16GiB and no
  // terminal line — the driver itself did not survive to write one.
  assert.equal(states.get("mlx-ltx-2-3-q8-1280x704-f241-fps30").terminal, "no_terminal_record");
  // Refused by the staged-residency guard at 252 s, then re-run and captured. One OK is enough.
  const rerun = states.get("mlx-ltx-2-3-q8-704x1280-f177-fps30");
  assert.equal(rerun.fails, 1);
  assert.equal(rerun.terminal, "completed");
  // Named by the free-disk STOP, never begun.
  const stopped = states.get("mlx-ltx-2-3-q8-1280x704-f177-fps30");
  assert.equal(stopped.stoppedBefore, true);
  assert.equal(stopped.terminal, "not_begun");
  assert.equal(states.has("mlx-ltx-2-3-q8-1280x704-f297-fps30"), false);
});

test("coverage buckets are derived from the log and the dataset, not from the role", () => {
  const points = [{ fixture: "ltx-2-3-mlx-q8-768x512-f121-fps30-seed18808", geometry: {} }];
  const coverage = coverageOf(PLAN, points, driverStatesFrom(DRIVER_LOGS));
  const state = (fixture) =>
    coverage.entries.find((entry) => entry.fixture === fixture).state;
  assert.equal(state("ltx-2-3-mlx-q8-768x512-f121-fps30-seed18808"), "captured");
  // A `fit` row can be attempted-and-killed; the role does not decide the bucket.
  assert.equal(
    state("ltx-2-3-mlx-q8-768x512-f361-fps30-seed18808"),
    "attempted_failed_host_limit",
  );
  assert.equal(
    state("ltx-2-3-mlx-q8-1280x704-f241-fps30-seed18808"),
    "attempted_failed_host_limit",
  );
  assert.equal(
    state("ltx-2-3-mlx-q8-1280x704-f297-fps30-seed18808"),
    "not_attempted_host_limit",
  );
  assert.equal(state("ltx-2-3-mlx-bf16-768x512-f121-fps30-seed18808"), "not_reached");
  // One captured (the single point supplied) and three the log shows were begun and never OK'd —
  // f361, f449 and f241. The old hardcoded list named two, and one of those two was wrong.
  assert.equal(coverage.byState.captured.total, 1);
  assert.equal(coverage.byState.attempted_failed_host_limit.total, 3);
  // Withholding the other seven captured records puts them in the retention-hole bucket rather than
  // silently in "failed" — the seven the log says completed and this call was not given.
  assert.equal(coverage.byState.completed_without_record.total, 7);
  assert.equal(coverage.plannedEntries, PLAN.providers.length);
});

test("the tier split of every coverage bucket is derived, not typed", () => {
  // The runbook stated this split in prose and got it wrong: it called the unreached rows "all bf16
  // and q4" while three were q8 — one of them a `fit` row. The split now comes from here.
  const coverage = coverageOf(PLAN, [], driverStatesFrom(DRIVER_LOGS));
  for (const [name, bucket] of Object.entries(coverage.byState)) {
    const summed = Object.values(bucket.byTier).reduce((sum, count) => sum + count, 0);
    assert.equal(summed, bucket.total, `${name} tier split must sum to its total`);
  }
  // The unreached rows are NOT all bf16 and q4 — the two q8 rung-2 boundary rows are there too.
  assert.deepEqual(coverage.byState.not_reached.byTier, { q8: 2, bf16: 12, q4: 11 });
  assert.equal(coverage.byState.not_reached.total, 25);
  // And the row the driver stopped before is its own bucket, in q8, and it is a `fit` row.
  assert.deepEqual(coverage.byState.stopped_before.byTier, { q8: 1 });
  const stopped = coverage.entries.filter((entry) => entry.state === "stopped_before");
  assert.equal(stopped.length, 1);
  assert.equal(stopped[0].fixture, "ltx-2-3-mlx-q8-1280x704-f177-fps30-seed18808");
  assert.equal(stopped[0].role, "fit");
});

test("the REALIZED fit design is smaller than the declared one, against the real dataset", () => {
  // Six q8 `fit` rows were declared; the committed dataset has records for four. One was attempted
  // and killed (768x512 f361) and one was never begun at all — 1280x704 f177, the geometry the
  // driver named on its STOP line. Pinned against the shipped bundle so no later reading of "the
  // fit" can quietly assume all six, and so re-capturing either row reds here until the prose that
  // describes the design is updated with it.
  const coverage = coverageOf(
    PLAN,
    pointsFrom(DATASET.records, rolesFromPlan(PLAN)),
    driverStatesFrom(DRIVER_LOGS),
  );
  const q8Fit = coverage.entries.filter(
    (entry) => entry.tier === "q8" && entry.role === "fit",
  );
  assert.equal(q8Fit.length, 6, "six q8 fit rows are DECLARED");
  const byState = (state) =>
    q8Fit.filter((entry) => entry.state === state).map((entry) => entry.fixture);
  assert.equal(byState("captured").length, 4, "four q8 fit rows were REALIZED");
  assert.deepEqual(byState("attempted_failed_host_limit"), [
    "ltx-2-3-mlx-q8-768x512-f361-fps30-seed18808",
  ]);
  assert.deepEqual(byState("stopped_before"), [
    "ltx-2-3-mlx-q8-1280x704-f177-fps30-seed18808",
  ]);
  // Eight captured FIXTURES over seven distinct {w,h,frames} geometries — the fps probe repeats
  // {768x512, f241}, which §3 of the runbook argues is the same geometry.
  assert.equal(coverage.capturedFixtures, 8);
  assert.equal(coverage.capturedGeometries, 7);
  assert.equal(coverage.byState.captured.total, 8);
});

test("the STOP line is consumed — deleting it moves the row the driver halted at", () => {
  // Before this, `stoppedBefore` was parsed and asserted but never read by `coverageOf`, so
  // deleting the STOP line changed no bucket and the assertion on it proved nothing downstream.
  const withoutStop = DRIVER_LOG.split("\n")
    .filter((line) => !line.startsWith("STOP "))
    .join("\n");
  const bucketOf = (logs) =>
    coverageOf(PLAN, [], driverStatesFrom(logs)).entries.find(
      (entry) => entry.fixture === "ltx-2-3-mlx-q8-1280x704-f177-fps30-seed18808",
    ).state;
  assert.equal(bucketOf(DRIVER_LOGS), "stopped_before");
  assert.equal(bucketOf([PRECRASH_LOG, withoutStop]), "not_reached");
});

test("a captured record with no OK terminal in a committed log is refused", () => {
  // The defect this closes: four of the thirteen records came from a driver session whose log was
  // not committed, so nothing tied them to any run. `provider.name` links to the log while records
  // key on `provider.fixture`, and for non-`*_host_limit` roles nothing checked the two agreed.
  const record = (fixture) => ({ fixture, geometry: {} });
  const threeF121 = [
    record("ltx-2-3-mlx-q8-768x512-f121-fps30-seed18808"),
    record("ltx-2-3-mlx-q8-768x512-f121-fps30-seed18808"),
    record("ltx-2-3-mlx-q8-768x512-f121-fps30-seed18808"),
  ];
  // Both sessions committed: three records, three OK terminals. Accounted for.
  assert.equal(
    coverageOf(PLAN, threeF121, driverStatesFrom(DRIVER_LOGS)).byState.captured.total,
    1,
  );
  // Drop the crashed session's log — exactly the shipped state this re-review found — and the
  // third record has no terminal line anywhere.
  assert.throws(
    () => coverageOf(PLAN, threeF121, driverStatesFrom([DRIVER_LOG])),
    /has 3 record\(s\) but the committed driver logs record 2 OK terminal\(s\)/,
  );
});

test("renaming a captured provider is refused rather than silently unlinking it", () => {
  // Renaming all eight captured providers used to throw nothing and leave `byState` byte-identical,
  // because only `*_host_limit` roles were cross-checked against the log at all.
  const renamed = {
    providers: PLAN.providers.map((provider) =>
      provider.name === "mlx-ltx-2-3-q8-768x512-f121-fps30"
        ? { ...provider, name: "mlx-ltx-2-3-q8-768x512-f121-fps30-renamed" }
        : provider,
    ),
  };
  assert.throws(
    () => coverageOf(renamed, [], driverStatesFrom(DRIVER_LOGS)),
    /the driver log names mlx-ltx-2-3-q8-768x512-f121-fps30, which is not a provider/,
  );
});

test("a terminal line with no open BEGIN is not the driver's verdict", () => {
  // The driver pipes each child's stderr into the stream it writes its own lines to, so an own-line
  // `OK <planname> 1s` inside captured child output is byte-indistinguishable from a real terminal.
  // Believing it would flip an entry to `completed` — the one bucket that silences the guard above.
  const forged = [
    "OK mlx-ltx-2-3-bf16-768x512-f121-fps30 1s",
    "FAIL mlx-ltx-2-3-bf16-768x512-f145-fps30 1s :: boom",
    "",
  ].join("\n");
  const states = driverStatesFrom([forged]);
  assert.equal(states.get("mlx-ltx-2-3-bf16-768x512-f121-fps30").oks, 0);
  assert.equal(states.get("mlx-ltx-2-3-bf16-768x512-f121-fps30").terminal, "not_begun");
  assert.equal(states.get("mlx-ltx-2-3-bf16-768x512-f145-fps30").terminal, "not_begun");
  // With a real BEGIN in front of it the same line IS believed — the guard is about provenance,
  // not about the spelling of the line.
  const genuine = driverStatesFrom([
    ["BEGIN mlx-ltx-2-3-bf16-768x512-f121-fps30 tier=bf16 free=95GiB 10:00:00", forged].join("\n"),
  ]);
  assert.equal(genuine.get("mlx-ltx-2-3-bf16-768x512-f121-fps30").terminal, "completed");
});

test("source sessions attribute every record to the session and revision that produced it", () => {
  const points = [
    { fixture: "ltx-2-3-mlx-q8-768x512-f121-fps30-seed18808", capturedAt: "2026-08-12T08:28:21Z", sceneWorksRevision: "aaa" },
    { fixture: "ltx-2-3-mlx-q8-768x512-f121-fps30-seed18808", capturedAt: "2026-08-12T10:34:17Z", sceneWorksRevision: "bbb" },
    { fixture: "ltx-2-3-mlx-q8-768x512-f121-fps30-seed18808", capturedAt: "2026-08-12T11:09:33Z", sceneWorksRevision: "ccc" },
  ];
  const sessions = sessionsFrom(LOGS, points, FIXTURE_BY_NAME);
  assert.equal(sessions.length, 2);
  assert.equal(sessions[0].log, LOG_PATHS[0]);
  assert.equal(sessions[0].firstBeginAt, "08:26:28");
  // The crashed session: five begun, four OK'd, one left without a terminal line.
  assert.equal(sessions[0].begins, 5);
  assert.equal(sessions[0].completed, 4);
  assert.equal(sessions[0].begunWithoutTerminalLine, 1);
  // The earliest f121 record belongs to the crashed session; the later two to the second.
  assert.equal(sessions[0].records, 1);
  assert.deepEqual(sessions[0].sceneWorksRevisions, ["aaa"]);
  assert.equal(sessions[1].records, 2);
  assert.deepEqual(sessions[1].sceneWorksRevisions, ["bbb", "ccc"]);
});

test("the noise floor reports whether its replicates span revisions", () => {
  const spanning = noiseFloor([
    { replicateKey: "a", value: 10, sceneWorksRevision: "aaa" },
    { replicateKey: "a", value: 10.25, sceneWorksRevision: "bbb" },
  ]);
  assert.equal(spanning.crossRevision, true);
  assert.deepEqual(spanning.spreads[0].sceneWorksRevisions, ["aaa", "bbb"]);
  const withinOne = noiseFloor([
    { replicateKey: "a", value: 10, sceneWorksRevision: "aaa" },
    { replicateKey: "a", value: 10.25, sceneWorksRevision: "aaa" },
  ]);
  assert.equal(withinOne.crossRevision, false);
  // The committed dataset is the first kind: no replicate group was captured under one revision.
  assert.equal(noiseFloor([]).crossRevision, false);
});

test("a plan that contradicts the driver log is refused, in both directions", () => {
  const states = driverStatesFrom(DRIVER_LOGS);
  const withRole = (name, role) => ({
    providers: PLAN.providers.map((provider) =>
      provider.name === name ? { ...provider, _role: role } : provider,
    ),
  });
  // The exact defect this PR was reviewed for: claiming a geometry was never attempted when the log
  // shows it was begun.
  assert.throws(
    () =>
      coverageOf(
        withRole("mlx-ltx-2-3-q8-768x512-f449-fps30", "not_attempted_host_limit"),
        [],
        states,
      ),
    /declared not_attempted_host_limit but the driver log records it as failed/,
  );
  // And the mirror: claiming an attempt the log has no BEGIN for.
  assert.throws(
    () =>
      coverageOf(
        withRole("mlx-ltx-2-3-bf16-1280x704-f297-fps30", "attempted_failed_host_limit"),
        [],
        states,
      ),
    /declared attempted_failed_host_limit but the driver log has no BEGIN line/,
  );
  // The committed plan itself passes both.
  assert.ok(coverageOf(PLAN, [], states).plannedEntries > 0);
});
