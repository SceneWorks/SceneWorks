import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { deriveMargins } from "./derive-ladder-margins.mjs";
import { inferencePinFromCargo } from "./inference-closure-digest.mjs";
import {
  MARGIN_SOURCE,
  buildStaleLaneReport,
  evidenceBindings,
  formatReport,
  loadSources,
  manifestBindings,
  rankLanes,
} from "./stale-lane-report.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const RUST_POLICY_PATH = path.join(ROOT, "crates", "sceneworks-worker", "src", "ladder_margin_policy.rs");

const digest = (seed) => seed.repeat(64).slice(0, 64);
const LIVE_MLX = digest("a");
const LIVE_CANDLE = digest("b");
const OLD = digest("c");

function record({ id, backend, provider, modelId, closureDigest, rung = "resident" }) {
  return {
    id,
    backend,
    target: { modelId, provider, mode: "text_to_image", tier: "q4", overlay: "none", geometry: { height: 1024, width: 1024 } },
    strategy: { rung, parameters: {} },
    repositories: { inference: { closureDigest } },
  };
}

function manifest(models) {
  return { models };
}

/**
 * Two lanes, one per backend, both fully stale — with MORE stale bindings on candle than on mlx.
 *
 * That inversion is the point: ranked on stale-binding COUNT alone candle would lead; ranked on the
 * margin the runtime actually widens each lane by (mlx 5% vs candle 2%), mlx leads. Any future
 * ranking that drops the margin term reds `ranking weighs the margin, not just the count`.
 */
function twoLaneFixture() {
  const records = [
    record({ id: "r-mlx-1", backend: "mlx", provider: "alpha", modelId: "alpha_model", closureDigest: OLD }),
    record({ id: "r-mlx-2", backend: "mlx", provider: "alpha", modelId: "alpha_model", closureDigest: OLD, rung: "bounded_decode" }),
    record({ id: "r-candle-1", backend: "candle", provider: "beta", modelId: "beta_model", closureDigest: OLD }),
  ];
  const models = [
    {
      id: "alpha_model",
      mlx: {
        calibrations: [
          { provider: "alpha", inferenceClosureDigest: OLD },
          { provider: "alpha", inferenceClosureDigest: OLD },
          { provider: "alpha", inferenceClosureDigest: OLD },
        ],
      },
    },
    {
      id: "beta_model",
      candle: {
        calibrations: [
          { provider: "beta", inferenceClosureDigest: OLD },
          { provider: "beta", inferenceClosureDigest: OLD },
          { provider: "beta", inferenceClosureDigest: OLD },
          { provider: "beta", inferenceClosureDigest: OLD },
          { provider: "beta", inferenceClosureDigest: OLD },
        ],
      },
    },
  ];
  return {
    liveDigests: new Map([
      ["mlx:alpha", LIVE_MLX],
      ["candle:beta", LIVE_CANDLE],
    ]),
    declarations: { "mlx:alpha": { crate: "crates/alpha" }, "candle:beta": { crate: "crates/beta" } },
    records,
    manifest: manifest(models),
  };
}

test("a lane whose captured digest differs from the live one is reported stale", () => {
  const report = buildStaleLaneReport(twoLaneFixture());
  assert.deepEqual(
    report.staleLanes.map((lane) => lane.lane).sort(),
    ["candle:beta", "mlx:alpha"],
  );
  assert.equal(report.totals.staleLanes, 2);
  assert.equal(report.totals.staleBindings, 8);
  assert.equal(report.totals.staleRecords, 3);
  assert.ok(report.staleLanes.every((lane) => lane.status === "stale"));
});

test("a lane whose captured digest matches the live one is current, and is not ranked", () => {
  const fixture = twoLaneFixture();
  for (const item of fixture.records) if (item.backend === "mlx") item.repositories.inference.closureDigest = LIVE_MLX;
  for (const binding of fixture.manifest.models[0].mlx.calibrations) binding.inferenceClosureDigest = LIVE_MLX;
  const report = buildStaleLaneReport(fixture);
  assert.deepEqual(report.staleLanes.map((lane) => lane.lane), ["candle:beta"]);
  assert.deepEqual(report.currentLanes.map((lane) => lane.lane), ["mlx:alpha"]);
  assert.equal(report.totals.staleBindings, 5);
  assert.equal(report.totals.staleRecords, 1);
});

test("a lane with a mixture is partially-stale, and only the stale half counts as impact", () => {
  const fixture = twoLaneFixture();
  fixture.records[0].repositories.inference.closureDigest = LIVE_MLX;
  fixture.manifest.models[0].mlx.calibrations[0].inferenceClosureDigest = LIVE_MLX;
  const report = buildStaleLaneReport(fixture);
  const mlx = report.staleLanes.find((lane) => lane.lane === "mlx:alpha");
  assert.equal(mlx.status, "partially-stale");
  assert.deepEqual(mlx.bindings, { total: 3, stale: 2, current: 1 });
  assert.deepEqual(mlx.records, { total: 2, stale: 1, current: 1 });
});

test("a declared lane with no measurement is unmeasured, never stale", () => {
  const fixture = twoLaneFixture();
  fixture.liveDigests.set("candle:gamma", digest("d"));
  fixture.declarations["candle:gamma"] = { crate: "crates/gamma" };
  const report = buildStaleLaneReport(fixture);
  assert.deepEqual(report.unmeasuredLanes.map((lane) => lane.lane), ["candle:gamma"]);
  assert.ok(!report.staleLanes.some((lane) => lane.lane === "candle:gamma"));
  assert.equal(report.totals.unmeasuredLanes, 1);
  assert.equal(report.totals.declaredLanes, 3);
});

test("ranking weighs the margin, not just the count", () => {
  const report = buildStaleLaneReport(twoLaneFixture());
  // candle has 5 stale bindings to mlx's 3, so a count-only ranking would put candle first.
  const [first, second] = report.staleLanes;
  assert.equal(first.lane, "mlx:alpha", "the wider-margin lane outranks a larger candle lane");
  assert.equal(second.lane, "candle:beta");
  assert.equal(first.rank, 1);
  assert.equal(second.rank, 2);
  assert.equal(first.impact.widenedAdmissionSurface, 3 * first.margin.staleMeasuredMargin);
  assert.equal(second.impact.widenedAdmissionSurface, 5 * second.margin.staleMeasuredMargin);
  assert.ok(first.impact.widenedAdmissionSurface > second.impact.widenedAdmissionSurface);
});

test("equal admission surface falls through to evidence surface, then to the lane name", () => {
  const margin = 0.05;
  const lane = (name, bindings, records) => ({
    lane: name,
    impact: { widenedAdmissionSurface: bindings * margin, widenedEvidenceSurface: records * margin },
  });
  assert.deepEqual(
    rankLanes([lane("mlx:b", 2, 1), lane("mlx:a", 2, 1), lane("mlx:c", 2, 9)]).map((item) => item.lane),
    ["mlx:c", "mlx:a", "mlx:b"],
  );
});

test("the lane key carries the backend, so one provider id on two backends stays two lanes", () => {
  const bindings = manifestBindings(
    manifest([
      {
        id: "krea_2_turbo",
        mlx: { calibrations: [{ provider: "krea_2_turbo_control", inferenceClosureDigest: OLD }] },
        candle: { calibrations: [{ provider: "krea_2_turbo_control", inferenceClosureDigest: OLD }] },
      },
    ]),
  );
  assert.deepEqual(bindings.map((binding) => binding.lane).sort(), [
    "candle:krea_2_turbo_control",
    "mlx:krea_2_turbo_control",
  ]);
  assert.deepEqual(
    evidenceBindings([
      record({ id: "x", backend: "candle", provider: "krea_2_turbo_control", modelId: "krea_2_turbo", closureDigest: OLD }),
    ]).map((item) => item.lane),
    ["candle:krea_2_turbo_control"],
  );
});

test("a record that carries no captured digest at all counts as stale, never as current", () => {
  const fixture = twoLaneFixture();
  delete fixture.records[0].repositories.inference.closureDigest;
  const report = buildStaleLaneReport(fixture);
  const mlx = report.staleLanes.find((lane) => lane.lane === "mlx:alpha");
  assert.equal(mlx.records.stale, 2);
  assert.ok(mlx.capturedDigests.includes("(absent)"));
});

test("the margin column is the derivation's, not a literal in this script", async () => {
  const fixture = twoLaneFixture();
  const report = buildStaleLaneReport(fixture);
  const derived = deriveMargins(fixture.records);
  for (const lane of report.staleLanes) {
    assert.equal(lane.margin.staleMeasuredMargin, derived[lane.backend].margins.staleMeasuredMargin);
    assert.equal(lane.margin.estimateMargin, derived[lane.backend].margins.estimateMargin);
  }
  // No margin literal may be reintroduced here: a hardcoded copy is exactly how this column would
  // drift away from the runtime it claims to describe.
  const source = await readFile(path.join(ROOT, "scripts", "stale-lane-report.mjs"), "utf8");
  const code = source.replace(/\/\*[\s\S]*?\*\/|(^|\s)\/\/.*$/gm, "");
  assert.equal(code.match(/0\.0[0-9]+/g), null, "no margin-shaped literal in the report's code");
});

test("the real corpus reports the margins the worker actually applies", async () => {
  const sources = await loadSources();
  const report = buildStaleLaneReport(sources);
  const rust = {};
  for (const match of (await readFile(RUST_POLICY_PATH, "utf8")).matchAll(
    /pub const ([A-Z0-9_]+): f64 = ([0-9.]+);/g,
  )) {
    rust[match[1]] = Number(match[2]);
  }
  const expected = {
    mlx: { stale: rust.MLX_STALE_MEASURED_MARGIN, estimate: rust.MLX_ESTIMATE_MARGIN },
    candle: { stale: rust.CANDLE_STALE_MEASURED_MARGIN, estimate: rust.CANDLE_ESTIMATE_MARGIN },
  };
  assert.ok(expected.mlx.stale > 0 && expected.candle.stale > 0, "the Rust policy constants parsed");
  for (const lane of report.staleLanes) {
    assert.equal(lane.margin.staleMeasuredMargin, expected[lane.backend].stale, lane.lane);
    assert.equal(lane.margin.estimateMargin, expected[lane.backend].estimate, lane.lane);
  }
});

test("the real corpus is entirely stale today, and the report says so", async () => {
  const sources = await loadSources();
  const report = buildStaleLaneReport(sources);
  // Not a target — an observation, and the reason this report is worth having: as of this corpus
  // NO lane's captured closure matches the live derivation, and CI is green anyway (sc-18098).
  assert.equal(report.totals.currentLanes, 0);
  assert.ok(report.totals.staleLanes > 0);
  assert.equal(report.totals.staleRecords, sources.records.length);
  assert.equal(report.staleLanes[0].lane, "mlx:qwen_image", "the largest measured lane leads");
  for (const lane of report.staleLanes) {
    assert.match(lane.liveDigest, /^[0-9a-f]{64}$/);
    assert.ok(lane.crate, `${lane.lane} resolves its declared crate`);
    assert.ok(lane.models.length, `${lane.lane} names the models it affects`);
  }
  for (const lane of report.unmeasuredLanes) {
    assert.equal(lane.records.total, 0);
    assert.equal(lane.bindings.total, 0);
  }
});

test("the human report names the ranked lanes, the widening, and its provenance", () => {
  const text = formatReport(buildStaleLaneReport(twoLaneFixture()));
  assert.match(text, /Staleness is a SIGNAL, not a gate/);
  assert.match(text, /1\s+mlx:alpha/);
  assert.match(text, /2\s+candle:beta/);
  assert.match(text, /5\.00%/);
  assert.match(text, /2\.00%/);
  assert.ok(text.includes(MARGIN_SOURCE));
});

test("the report is graded against a closure table keyed to the live pin", async () => {
  // Same predicate the matrix generator uses (`validatedInferenceClosures`), reached through the
  // same pin resolver the closure-digest derivation uses. A report that graded currency against a
  // table keyed to an older pin would name the wrong lanes.
  const cargo = await readFile(path.join(ROOT, "Cargo.toml"), "utf8");
  const closures = JSON.parse(
    await readFile(path.join(ROOT, "config", "inference-provider-closures.json"), "utf8"),
  );
  assert.equal(closures.inferenceRevision, inferencePinFromCargo(cargo));
  assert.throws(
    () => inferencePinFromCargo("[dependencies]\nserde = \"1\"\n"),
    /could not resolve the pinned SceneWorks\/inference revision/,
  );
});
