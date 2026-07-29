// Coverage for the ONLY Shortcut-mutating script in the repo (sc-15812 review).
//
// `apply()` performs ~230 irreversible writes against the live story graph, and every one of them is
// driven by `buildPlan`'s classification of the generated matrix. A mis-classified model creates a
// Candle twin under the wrong parent, or — worse — creates one for an mlx-only entry, which can never
// be closed. That is the same "credited to a story that cannot close it" failure the epic exists to
// kill, so the classification is tested directly rather than trusted.
//
// `buildPlan` is pure: it takes a matrix and the expected shape, and throws `PlanError` instead of
// exiting, so the rejection paths are assertable in-process. Nothing here touches the network.

import assert from "node:assert/strict";
import test from "node:test";

import { PlanError, buildPlan, candleTwinName } from "./split-memory-strategy-stories.mjs";

/** One matrix cell. Only the four fields `buildPlan` reads are populated. */
function cell(modelId, backend, model, family) {
  return { modelId, backend, owningModelStory: model, owningFamilyStory: family };
}

// alpha is dual (family 200), beta is mlx-only in the SAME family, gamma is mlx-only in an mlx-only
// family 201. Two cells per (model, backend) so the agreement branch is exercised, not just the first
// write. The mlx and candle ids differ everywhere, which is the point of sc-15812.
const DUAL_MATRIX = {
  cells: [
    cell("alpha", "mlx", 100, 200),
    cell("alpha", "mlx", 100, 200),
    cell("alpha", "candle", 300, 400),
    cell("alpha", "candle", 300, 400),
    cell("beta", "mlx", 101, 200),
    cell("gamma", "mlx", 102, 201),
  ],
};

const DUAL_EXPECT = {
  totalModels: 3,
  dualModels: 1,
  mlxOnlyModels: 2,
  candleOnlyModels: 0,
  dualFamilies: 1,
  mlxOnlyFamilies: 1,
};

test("a dual model is planned from per-backend ownership, and the two backends may differ", () => {
  const plan = buildPlan(DUAL_MATRIX, DUAL_EXPECT);

  assert.equal(plan.dualModels.length, 1);
  assert.deepEqual(plan.dualModels[0], {
    model: "alpha",
    mlxStory: 100,
    family: 200,
    candleStory: 300,
    candleFamily: 400,
    backends: ["candle", "mlx"],
  });

  // The Candle family twin is read off the matrix, not the gitignored state file — that is what makes
  // --verify work on a fresh clone.
  assert.deepEqual(plan.candleFamilyByFamily, { 200: 400 });
  assert.deepEqual(plan.dualFamilies, [200]);

  // The mlx-only set is what apply() must leave alone, so its membership is load-bearing. beta shares
  // alpha's family, which is exactly where a family-level split would wrongly drag it along.
  assert.deepEqual(
    plan.mlxOnly.map((m) => m.model).sort(),
    ["beta", "gamma"],
  );
  assert.deepEqual(plan.mlxOnly.map((m) => m.candleStory), [null, null]);
  assert.deepEqual(plan.mlxOnlyFamilies, [201]);
  assert.deepEqual(plan.actual, DUAL_EXPECT);
});

// Before sc-15812 every cell of a model carried the MLX pair, so a single ownership check across all
// of a model's cells was correct. Now it is not: an mlx cell and a candle cell of the same model SHOULD
// name different stories. A guard still comparing across backends would fire on every correct matrix,
// which is why the disagreement check is scoped WITHIN a backend.
test("differing mlx and candle ownership is agreement, not disagreement", () => {
  assert.doesNotThrow(() => buildPlan(DUAL_MATRIX, DUAL_EXPECT));
});

test("cells of one model and one backend may not disagree about ownership", () => {
  // Same model, same backend, two different owning model stories: the generator changed under us.
  assert.throws(
    () =>
      buildPlan(
        { cells: [...DUAL_MATRIX.cells, cell("alpha", "mlx", 999, 200)] },
        DUAL_EXPECT,
      ),
    (error) =>
      error instanceof PlanError && /alpha: mlx cells disagree about owning story\/family/.test(error.message),
  );
  // ...and the same for the family half, on the candle side.
  assert.throws(
    () =>
      buildPlan(
        { cells: [...DUAL_MATRIX.cells, cell("alpha", "candle", 300, 999)] },
        DUAL_EXPECT,
      ),
    (error) =>
      error instanceof PlanError && /alpha: candle cells disagree about owning story\/family/.test(error.message),
  );
});

test("two dual models in one family may not name different Candle family twins", () => {
  assert.throws(
    () =>
      buildPlan(
        {
          cells: [
            ...DUAL_MATRIX.cells,
            cell("delta", "mlx", 103, 200),
            cell("delta", "candle", 301, 401),
          ],
        },
        { ...DUAL_EXPECT, totalModels: 4, dualModels: 2 },
      ),
    (error) =>
      error instanceof PlanError &&
      /family SC-200: models disagree about the Candle family twin \(SC-400 vs SC-401\)/.test(error.message),
  );
});

// A candle-only entry has no MLX story to split from and no MLX twin to rescope, so `apply` would
// create a parentless Candle story or skip it silently. sc-15812 asserts the catalog has none; if one
// appears the split must be re-derived, not half-applied.
test("a candle-only model stops the plan instead of being half-applied", () => {
  // totalModels is raised to 4 so the candle-only entry itself is what trips the plan, rather than the
  // head count noticing it first.
  assert.throws(
    () =>
      buildPlan(
        { cells: [...DUAL_MATRIX.cells, cell("epsilon", "candle", 302, 402)] },
        { ...DUAL_EXPECT, totalModels: 4 },
      ),
    (error) =>
      error instanceof PlanError && /invariant candleOnlyModels: matrix says 1, sc-15812 says 0/.test(error.message),
  );

  // The invariant is a real comparison, not a formality: a drifted total is rejected too.
  assert.throws(
    () => buildPlan(DUAL_MATRIX, { ...DUAL_EXPECT, totalModels: 4 }),
    (error) =>
      error instanceof PlanError && /invariant totalModels: matrix says 3, sc-15812 says 4/.test(error.message),
  );
});

test("an unrecognised cell backend is a defect, not a skipped row", () => {
  assert.throws(
    () => buildPlan({ cells: [cell("alpha", "rocm", 100, 200)] }, DUAL_EXPECT),
    (error) => error instanceof PlanError && /alpha: unknown cell backend rocm/.test(error.message),
  );
});

// `apply` creates twins under this name and `verify` looks them up by it, so the two must derive it
// identically — including for an mlx-only story that was never rescoped, which is the case the
// mlx-only guard depends on.
test("the Candle twin name is stable whether or not the MLX story was rescoped", () => {
  assert.equal(candleTwinName("Z-Image ladder"), "Z-Image ladder — Candle/CUDA");
  assert.equal(candleTwinName("Z-Image ladder — MLX/Metal"), "Z-Image ladder — Candle/CUDA");
  assert.equal(
    candleTwinName(candleTwinName("Z-Image ladder").replace(" — Candle/CUDA", "")),
    "Z-Image ladder — Candle/CUDA",
  );
});
