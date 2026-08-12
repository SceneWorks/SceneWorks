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
} from "./fit-ltx-temporal-form.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const PLAN = JSON.parse(
  readFileSync(path.join(ROOT, "docs/calibration/sc-18810/ltx-mlx-geometry-sweep.json"), "utf8"),
);
const DRIVER_LOG = readFileSync(
  path.join(ROOT, "docs/calibration/sc-18810/sweep-run.log"),
  "utf8",
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
  const states = driverStatesFrom(DRIVER_LOG);
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
  const coverage = coverageOf(PLAN, points, driverStatesFrom(DRIVER_LOG));
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
  assert.equal(coverage.byState.captured, 1);
  assert.equal(coverage.byState.attempted_failed_host_limit, 3);
  // Withholding the other seven captured records puts them in the retention-hole bucket rather than
  // silently in "failed" — the seven the log says completed and this call was not given.
  assert.equal(coverage.byState.completed_without_record, 7);
  assert.equal(coverage.plannedEntries, PLAN.providers.length);
});

test("a plan that contradicts the driver log is refused, in both directions", () => {
  const states = driverStatesFrom(DRIVER_LOG);
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
