import assert from "node:assert/strict";
import { copyFile, mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { digestOccurrences, recordsNeedingDigest } from "./backfill-closure-digests.mjs";
import { deriveMargins } from "./derive-ladder-margins.mjs";
import { inferencePinFromCargo } from "./inference-closure-digest.mjs";
import {
  CAPTURABILITY_SOURCE,
  MARGIN_SOURCE,
  SOURCE_PATHS,
  adapterCapturableProviders,
  buildStaleLaneReport,
  evidenceBindings,
  formatReport,
  laneModelAttribution,
  loadSources,
  manifestBindings,
  planLaneCoverage,
  rankLanes,
} from "./stale-lane-report.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const RUST_POLICY_PATH = path.join(ROOT, "crates", "sceneworks-worker", "src", "ladder_margin_policy.rs");
const MANIFEST_PATH = path.join(ROOT, "config", "manifests", "builtin.models.jsonc");

const digest = (seed) => seed.repeat(64).slice(0, 64);
const LIVE_MLX = digest("a");
const LIVE_CANDLE = digest("b");
const OLD = digest("c");
const REVISION = "1".repeat(40);

function binding(provider, closureDigest) {
  return { provider, inferenceRevision: REVISION, inferenceClosureDigest: closureDigest };
}

function record({
  id,
  backend,
  provider,
  modelId,
  closureDigest,
  rung = "resident",
  // Defaults keep fixture records inside `recordsNeedingDigest`'s population (sc-18252): only a
  // complete/runtime_complete authoritative record ever reaches the currency comparison.
  status = "complete",
  evidenceScope = "authoritative",
}) {
  return {
    id,
    backend,
    status,
    evidenceScope,
    target: { modelId, provider, mode: "text_to_image", tier: "q4", overlay: "none", geometry: { height: 1024, width: 1024 } },
    strategy: { rung, parameters: {} },
    repositories: { inference: { closureDigest } },
  };
}

/**
 * A synthetic adapter binary source whose `run()` dispatches exactly `providers` (sc-18212).
 *
 * The shape matters: `adapterCapturableProviders` anchors on the shared refusal phrase and parses
 * the surrounding match arms, so this fixture exercises the REAL parser end to end — string-literal
 * arms, an ALL_CAPS const arm when a provider is spelled `NAME=value`, and a lowercase fallback
 * binding carrying the refusal.
 */
function adapterSource(providers, { extra = "" } = {}) {
  const consts = [];
  const arms = providers.map((provider) => {
    const named = /^([A-Z][A-Z0-9_]*)=(.+)$/.exec(provider);
    if (!named) return `        "${provider}" => run_arm(request),`;
    consts.push(`const ${named[1]}: &str = "${named[2]}";`);
    return `        ${named[1]} => run_arm(request),`;
  });
  return `${consts.join("\n")}
fn run(request: &Value) -> Result<Value, String> {
    match provider {
${arms.join("\n")}
        other => Err(format!(
            "synthetic five-rung calibration does not implement provider {other:?}"
        )),
    }
}
${extra}`;
}

function planEntry(backend, provider, evidenceScope = "authoritative") {
  return { name: `${backend}-${provider}-${evidenceScope}`, backend, evidenceScope, target: { provider } };
}

/**
 * Assemble a fixture whose parsed manifest and raw JSONC body are the SAME object.
 *
 * The body matters: `manifestBindings` locates the binding population with the CI gate's own
 * line-based locator (`backfill-closure-digests.mjs#stampManifest`), so a fixture that supplied only
 * a parsed object would exercise none of it. `JSON.stringify(_, null, 2)` is valid JSONC and puts
 * `"mlx": {` / `"candle": {` at end of line, which is what the locator's backend walk pairs on.
 */
function fixtureFrom({
  models,
  records,
  lanes,
  plan = { providers: [] },
  // Every provider the fixtures declare has an arm BY DEFAULT, so pre-sc-18212 expectations (an
  // unmeasured lane is "pending capture") keep holding; capturability tests override these.
  adapterSources = { mlx: adapterSource(["alpha"]), candle: adapterSource(["beta", "gamma"]) },
}) {
  return {
    liveDigests: new Map(lanes),
    declarations: Object.fromEntries(
      lanes.map(([lane]) => [lane, { crate: `crates/${lane.replace(":", "-")}` }]),
    ),
    records,
    manifest: { models },
    manifestBody: JSON.stringify({ models }, null, 2),
    plan,
    adapterSources,
  };
}

/**
 * Two lanes, one per backend, both fully stale — with MORE stale bindings on candle than on mlx.
 *
 * That inversion is the point: ranked on stale-binding COUNT alone candle would lead; ranked on the
 * margin the runtime actually widens each lane by (mlx 5% vs candle 2%), mlx leads. Any future
 * ranking that drops the margin term reds `ranking weighs the margin, not just the count`.
 */
function twoLaneFixture({
  mlxBindings = [OLD, OLD, OLD],
  candleBindings = [OLD, OLD, OLD, OLD, OLD],
  mlxRecords = [OLD, OLD],
  candleRecords = [OLD],
  extraModels = [],
  extraLanes = [],
  extraRecords = [],
  ...rest
} = {}) {
  return fixtureFrom({
    ...rest,
    models: [
      { id: "alpha_model", mlx: { calibrations: mlxBindings.map((item) => binding("alpha", item)) } },
      { id: "beta_model", candle: { calibrations: candleBindings.map((item) => binding("beta", item)) } },
      ...extraModels,
    ],
    records: [
      ...mlxRecords.map((item, index) =>
        record({
          id: `r-mlx-${index}`,
          backend: "mlx",
          provider: "alpha",
          modelId: "alpha_model",
          closureDigest: item,
          rung: index ? "bounded_decode" : "resident",
        }),
      ),
      ...candleRecords.map((item, index) =>
        record({ id: `r-candle-${index}`, backend: "candle", provider: "beta", modelId: "beta_model", closureDigest: item }),
      ),
      ...extraRecords,
    ],
    lanes: [["mlx:alpha", LIVE_MLX], ["candle:beta", LIVE_CANDLE], ...extraLanes],
  });
}

test("a lane whose captured digest differs from the live one is reported stale", () => {
  const report = buildStaleLaneReport(twoLaneFixture());
  assert.deepEqual(report.staleLanes.map((lane) => lane.lane).sort(), ["candle:beta", "mlx:alpha"]);
  assert.equal(report.totals.staleLanes, 2);
  assert.equal(report.totals.staleBindings, 8);
  assert.equal(report.totals.staleRecords, 3);
  assert.ok(report.staleLanes.every((lane) => lane.status === "stale"));
});

test("a lane whose captured digest matches the live one is current, and is not ranked", () => {
  const report = buildStaleLaneReport(
    twoLaneFixture({ mlxBindings: [LIVE_MLX, LIVE_MLX, LIVE_MLX], mlxRecords: [LIVE_MLX, LIVE_MLX] }),
  );
  assert.deepEqual(report.staleLanes.map((lane) => lane.lane), ["candle:beta"]);
  assert.deepEqual(report.currentLanes.map((lane) => lane.lane), ["mlx:alpha"]);
  assert.equal(report.totals.staleBindings, 5);
  assert.equal(report.totals.staleRecords, 1);
});

test("a lane with a mixture is partially-stale, and only the stale half counts as impact", () => {
  const report = buildStaleLaneReport(
    twoLaneFixture({ mlxBindings: [LIVE_MLX, OLD, OLD], mlxRecords: [LIVE_MLX, OLD] }),
  );
  const mlx = report.staleLanes.find((lane) => lane.lane === "mlx:alpha");
  assert.equal(mlx.status, "partially-stale");
  assert.deepEqual(mlx.bindings, { total: 3, stale: 2, current: 1 });
  assert.deepEqual(mlx.records, { total: 2, stale: 1, current: 1, ineligible: 0 });
});

test("a declared lane with no measurement is unmeasured, never stale", () => {
  const report = buildStaleLaneReport(twoLaneFixture({ extraLanes: [["candle:gamma", digest("d")]] }));
  assert.deepEqual(report.unmeasuredLanes.map((lane) => lane.lane), ["candle:gamma"]);
  assert.ok(!report.staleLanes.some((lane) => lane.lane === "candle:gamma"));
  assert.equal(report.totals.unmeasuredLanes, 1);
  assert.equal(report.totals.declaredLanes, 3);
});

test("a whole-block fit outside calibrations[] is a binding, not an unmeasured lane", () => {
  // The regression this test exists to prevent. `turboFit` and `candle.control` carry
  // `provider`/`inferenceRevision`/`inferenceClosureDigest` directly on the block, outside any
  // `calibrations[]` array, and the worker reads them (`vram_gate.rs`, `krea_control_fit.rs`). The
  // first version of this report walked `calibrations[]` and printed both of their lanes as
  // "declared but never captured" while they were stale production bindings — reopening in a new
  // file exactly the hole sc-17989 closed.
  const report = buildStaleLaneReport(
    twoLaneFixture({
      extraModels: [{ id: "gamma_model", candle: { turboFit: binding("gamma", OLD) } }],
      extraLanes: [["candle:gamma", LIVE_CANDLE]],
    }),
  );
  const gamma = report.staleLanes.find((lane) => lane.lane === "candle:gamma");
  assert.ok(gamma, "the whole-block fit's lane must be reported stale, not unmeasured");
  assert.deepEqual(gamma.bindings, { total: 1, stale: 1, current: 0 });
  assert.deepEqual(gamma.records, { total: 0, stale: 0, current: 0, ineligible: 0 });
  assert.deepEqual(report.unmeasuredLanes, []);
  assert.deepEqual(gamma.models, ["gamma_model"]);
});

test("the located population and the parsed attribution must agree, or the report refuses", () => {
  // The attribution walk feeds only the MODELS column, but a silent disagreement with the locator
  // would mean one of the two cannot see a binding shape. Simulated by handing `manifest` a model
  // the raw body does not contain.
  const fixture = twoLaneFixture();
  fixture.manifest.models.push({ id: "ghost_model", candle: { calibrations: [binding("ghost", OLD)] } });
  assert.throws(
    () => buildStaleLaneReport(fixture),
    /manifest binding walks disagree on candle:ghost: the locator found 0, the parsed attribution found 1/,
  );
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
  const models = [
    {
      id: "krea_2_turbo",
      mlx: { calibrations: [binding("krea_2_turbo_control", OLD)] },
      candle: { control: binding("krea_2_turbo_control", OLD) },
    },
  ];
  const bindings = manifestBindings({
    manifest: { models },
    manifestBody: JSON.stringify({ models }, null, 2),
  });
  assert.deepEqual(bindings.map((item) => item.lane).sort(), [
    "candle:krea_2_turbo_control",
    "mlx:krea_2_turbo_control",
  ]);
  assert.deepEqual(
    [...laneModelAttribution({ models }).keys()].sort(),
    ["candle:krea_2_turbo_control", "mlx:krea_2_turbo_control"],
  );
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

test("the located binding population covers every closure digest the manifest carries", async () => {
  // Derived from the manifest, not pinned to a number: a NEW whole-block fit lands as one more
  // `inferenceClosureDigest` occurrence, and a population walk that cannot see it reds here. Same
  // "derive coverage from the source" discipline the pre-push trigger test applies. A
  // `calibrations[]`-only walk fails this by two on the shipped manifest.
  // The occurrence count comes from `digestOccurrences` — the SAME comment-excluding scan the
  // locator's own orphan check consumes — not a resembling regex over the raw body (sc-18208). The
  // manifest is JSONC and narrates digest provenance in prose, so a raw-body regex also counts a
  // digest shape quoted inside a `//` comment and would false-red this check the day one lands.
  const manifestBody = await readFile(MANIFEST_PATH, "utf8");
  const { manifest } = await loadSources();
  const located = manifestBindings({ manifestBody, manifest });
  const occurrences = digestOccurrences(manifestBody.split("\n")).length;
  assert.ok(occurrences > 0, "the manifest carries closure digests at all");
  assert.equal(located.length, occurrences, "every manifest closure digest is in the population");
  assert.ok(located.every((item) => /^(mlx|candle):.+/.test(item.lane)));
  assert.ok(located.every((item) => item.digest === null || /^[0-9a-f]{64}$/.test(item.digest)));
});

test("a digest shape quoted in a // comment is not part of the population, nor of the derived count", () => {
  // Regression guard for the sc-18208 fix above. The commented copy of the binding shape must be
  // invisible to BOTH sides of the coverage equation: the locator (which would otherwise report an
  // orphan and refuse) and the occurrence count (which would otherwise exceed the population and
  // false-red the coverage test on a perfectly healthy manifest).
  const models = [{ id: "alpha_model", mlx: { calibrations: [binding("alpha", OLD)] } }];
  const body = JSON.stringify({ models }, null, 2);
  const commented =
    `// provenance prose quoting the shape: "inferenceClosureDigest": "${digest("f")}"\n${body}`;

  assert.equal(digestOccurrences(commented.split("\n")).length, 1, "the commented digest is not counted");
  const located = manifestBindings({ manifestBody: commented, manifest: { models } });
  assert.equal(located.length, digestOccurrences(commented.split("\n")).length);
  assert.deepEqual(located.map((item) => item.digest), [OLD]);

  // The exact hazard the raw-body regex had: it sees 2 where the gate's own predicate sees 1.
  const naive = [...commented.matchAll(/"inferenceClosureDigest":\s*"[0-9a-f]{64}"/g)].length;
  assert.equal(naive, 2, "a resembling raw-body regex DOES count the commented digest");
});

test("the real corpus reports the margins the worker actually applies", async () => {
  const report = buildStaleLaneReport(await loadSources());
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
  for (const lane of [...report.staleLanes, ...report.currentLanes]) {
    assert.equal(lane.margin.staleMeasuredMargin, expected[lane.backend].stale, lane.lane);
    assert.equal(lane.margin.estimateMargin, expected[lane.backend].estimate, lane.lane);
  }
});

test("the real corpus report is internally consistent, whatever the corpus currently is", async () => {
  // INVARIANTS ONLY — deliberately no snapshot of how stale the corpus happens to be today. This
  // file runs in `npm run check` on every PR, so a pin like "0 lanes are current" or "qwen ranks
  // first" would red the moment someone lands a re-capture: the reverse-direction friction epic
  // 18093 exists to remove, installed by the very report that surfaces it. The ones that already
  // exist elsewhere are recorded on sc-18104; this must not add another.
  const sources = await loadSources();
  const report = buildStaleLaneReport(sources);
  // Declared+unmeasured+armless lanes live ONLY in `uncapturableLanes` (status "uncapturable");
  // measured armless lanes stay in the staleness partition and appear in `uncapturableLanes` as a
  // second, cross-cutting membership.
  const all = [
    ...report.staleLanes,
    ...report.currentLanes,
    ...report.unmeasuredLanes,
    ...report.uncapturableLanes.filter((lane) => lane.status === "uncapturable"),
  ];
  const universe = [...all, ...report.undeclaredLanes];

  assert.equal(all.length, report.totals.declaredLanes);
  assert.equal(all.length, sources.liveDigests.size);
  assert.equal(
    report.totals.declaredLanes,
    report.totals.staleLanes +
      report.totals.currentLanes +
      report.totals.unmeasuredLanes +
      report.uncapturableLanes.filter((lane) => lane.status === "uncapturable").length,
  );
  assert.equal(report.totals.staleBindings, all.reduce((sum, lane) => sum + lane.bindings.stale, 0));
  assert.equal(report.totals.staleRecords, all.reduce((sum, lane) => sum + lane.records.stale, 0));
  const eligibleRecords = recordsNeedingDigest({ records: sources.records }).length;
  assert.equal(
    universe.reduce((sum, lane) => sum + lane.records.total, 0),
    eligibleRecords,
    "every digest-eligible evidence record is attributed to exactly one lane",
  );
  assert.equal(
    universe.reduce((sum, lane) => sum + lane.records.ineligible, 0),
    sources.records.length - eligibleRecords,
    "every non-eligible record is attributed too, outside the currency tallies",
  );
  assert.equal(
    universe.reduce((sum, lane) => sum + lane.bindings.total, 0),
    manifestBindings(sources).length,
    "every manifest binding is attributed to exactly one lane",
  );

  for (const lane of all) {
    assert.match(lane.liveDigest, /^[0-9a-f]{64}$/, lane.lane);
    assert.ok(lane.crate, `${lane.lane} resolves its declared crate`);
    assert.ok(lane.margin, `${lane.lane} carries a margin`);
    assert.equal(lane.bindings.stale + lane.bindings.current, lane.bindings.total, lane.lane);
    assert.equal(lane.records.stale + lane.records.current, lane.records.total, lane.lane);
  }

  // Capturability coherence: the per-lane flag IS the derived arms table, everywhere, and the
  // cross-cutting list is exactly the flagged lanes. `unmeasuredLanes` may promise a capture only
  // for lanes an adapter arm can actually serve — the sc-18212 defect was `candle:z_image`
  // (declared, 90 plan entries, no arm) printing as pending measurement work.
  assert.ok(report.capturability.arms.mlx.length > 0, "the mlx adapter dispatch parsed");
  assert.ok(report.capturability.arms.candle.length > 0, "the candle adapter dispatch parsed");
  for (const lane of universe) {
    assert.equal(
      lane.capturable,
      report.capturability.arms[lane.backend].includes(lane.provider),
      `${lane.lane} capturable flag matches the parsed adapter arms`,
    );
  }
  assert.deepEqual(
    universe.filter((lane) => !lane.capturable).map((lane) => lane.lane).sort(),
    [...report.capturability.uncapturableLanes].sort(),
    "the uncapturable list is exactly the lanes without an arm",
  );
  for (const lane of report.unmeasuredLanes) {
    assert.ok(lane.capturable, `${lane.lane} is pending capture, so an adapter arm must exist`);
  }
  for (const lane of report.uncapturableLanes) {
    assert.equal(lane.capturable, false, lane.lane);
    if (lane.status === "uncapturable") {
      assert.equal(lane.bindings.total + lane.records.total, 0, `${lane.lane} is unmeasured`);
    }
  }
  // Every planned lane is somewhere in the report: the pre-sc-18212 report enumerated declared
  // lanes only, so a planned-but-undeclared lane (candle:qwen_image_edit, sc-18104 §2d) was
  // invisible in exactly the view an operator books capture hosts from.
  const reported = new Set(universe.map((lane) => lane.lane));
  for (const lane of planLaneCoverage(sources.plan).keys()) {
    assert.ok(reported.has(lane), `planned lane ${lane} appears in the report`);
  }
  for (const lane of report.undeclaredLanes) {
    assert.equal(lane.declared, false, lane.lane);
    assert.ok(lane.plan.entries > 0, `${lane.lane} is only in the universe because the plan names it`);
  }
  for (const lane of report.staleLanes) {
    assert.ok(["stale", "partially-stale"].includes(lane.status), lane.lane);
    assert.ok(lane.bindings.stale + lane.records.stale > 0, `${lane.lane} is ranked for a reason`);
    assert.ok(lane.models.length, `${lane.lane} names the models it affects`);
    if (lane.status === "stale") {
      assert.ok(
        !lane.capturedDigests.includes(lane.liveDigestShort),
        `${lane.lane} is fully stale, so no captured digest may equal the live one`,
      );
    }
  }
  for (const lane of report.currentLanes) {
    assert.equal(lane.bindings.stale + lane.records.stale, 0, lane.lane);
    assert.ok(lane.bindings.total + lane.records.total > 0, `${lane.lane} is current because it was measured`);
  }
  // The POSITIVE form of the classification check. Asserting "unmeasured lanes have 0 bindings"
  // instead passed happily while the population walk was blind to whole-block fits — it was a
  // consequence of the bug, not a check on it.
  const bound = new Set(manifestBindings(sources).map((item) => item.lane));
  const measured = new Set(
    // The gate's own population (sc-18252): a fixture/candidate record measures nothing that the
    // currency comparison would ever see, so it must not disqualify a lane from "unmeasured".
    recordsNeedingDigest({ records: sources.records }).map(
      (item) => `${item.backend}:${item.target.provider}`,
    ),
  );
  for (const lane of report.unmeasuredLanes) {
    assert.ok(!bound.has(lane.lane), `${lane.lane} has a manifest binding, so it is not unmeasured`);
    assert.ok(!measured.has(lane.lane), `${lane.lane} has evidence records, so it is not unmeasured`);
  }
  // Ranking is monotone in the declared primary key.
  const surfaces = report.staleLanes.map((lane) => lane.impact.widenedAdmissionSurface);
  assert.deepEqual(surfaces, [...surfaces].sort((left, right) => right - left));
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

test("the report is graded against a closure table keyed to the live pin", async (t) => {
  // Same predicate the matrix generator uses (`validatedInferenceClosures`), reached through the
  // same pin resolver the closure-digest derivation uses. A report that graded currency against a
  // table keyed to an older pin would name the wrong lanes.
  //
  // sc-18252: the first version of this test only compared the two files to each other and never
  // called the report, so replacing the `validatedInferenceClosures` call inside `loadSources` with
  // a raw `closures.providers` read kept every test green (mutation-verified). The gate must be
  // exercised THROUGH the report's own loader: copy the real sources into a scratch root, re-key
  // the closure table to a different revision, and the load itself must refuse.
  const cargo = await readFile(path.join(ROOT, "Cargo.toml"), "utf8");
  const closures = JSON.parse(
    await readFile(path.join(ROOT, "config", "inference-provider-closures.json"), "utf8"),
  );
  assert.equal(closures.inferenceRevision, inferencePinFromCargo(cargo));
  assert.throws(
    () => inferencePinFromCargo("[dependencies]\nserde = \"1\"\n"),
    /could not resolve the pinned SceneWorks\/inference revision/,
  );

  const scratch = await mkdtemp(path.join(os.tmpdir(), "stale-lane-pin-"));
  t.after(() => rm(scratch, { recursive: true, force: true }));
  for (const relative of Object.values(SOURCE_PATHS)) {
    await mkdir(path.dirname(path.join(scratch, relative)), { recursive: true });
    await copyFile(path.join(ROOT, relative), path.join(scratch, relative));
  }
  // The untampered copy loads and builds — so the refusal below is the re-keying, nothing else.
  buildStaleLaneReport(await loadSources(scratch));
  const rekeyed = { ...closures, inferenceRevision: "f".repeat(40) };
  await writeFile(path.join(scratch, SOURCE_PATHS.closures), JSON.stringify(rekeyed, null, 2));
  await assert.rejects(loadSources(scratch), /is keyed to ffffffff but Cargo pins/);
});

test("capturability is parsed from the dispatch arms — literals, consts, and no test scaffolding", () => {
  const source = adapterSource(["alpha", "Z_PROVIDER=zeta"], {
    extra: `
#[cfg(test)]
mod tests {
    // The phrase appears in test prose too: "five-rung calibration does not implement provider".
    fn pins_the_refusal() {
        let expected = format!("five-rung calibration does not implement provider {p:?}");
        let ghost = match p { "phantom_provider" => 1, _ => 0 };
    }
}
`,
  });
  assert.deepEqual(adapterCapturableProviders(source, "synthetic"), ["alpha", "zeta"]);
});

test("a provider is capturable only if EVERY dispatch gate admits it", () => {
  // The candle adapter dispatches twice (entry dispatch and generator loading); a provider present
  // in one match but missing from the other cannot complete a capture and must not be reported
  // capturable.
  const source = `
fn entry(request: &Value) -> Result<&'static str, String> {
    match planned_provider(request)? {
        "alpha" => Ok(ALPHA_PATH),
        "beta" => Ok(BETA_PATH),
        provider => Err(format!(
            "synthetic five-rung calibration does not implement provider {provider:?}"
        )),
    }
}
fn load(request: &Value) -> Result<Loaded, String> {
    match planned_provider(request)? {
        "alpha" => Ok(load_alpha()),
        provider => {
            return Err(format!(
                "synthetic five-rung calibration does not implement provider {provider:?}"
            ))
        }
    }
}
`;
  assert.deepEqual(adapterCapturableProviders(source, "synthetic"), ["alpha"]);
});

test("losing the dispatch anchor is loud, never an empty (or full) capturable set", () => {
  assert.throws(
    () => adapterCapturableProviders("fn run() -> u32 { 42 }", "synthetic"),
    /no provider dispatch found/,
  );
  assert.throws(
    () =>
      adapterCapturableProviders(
        `fn run(r: &Value) -> Result<Value, String> {
    match provider {
        UNDECLARED_CONST => run_arm(r),
        other => Err(format!("five-rung calibration does not implement provider {other:?}")),
    }
}`,
        "synthetic",
      ),
    /does not resolve to a &str const/,
  );
  assert.throws(
    () => buildStaleLaneReport({ ...twoLaneFixture(), adapterSources: { mlx: adapterSource(["alpha"]) } }),
    /needs both adapter sources/,
  );
});

test("a declared, planned lane with no adapter arm is uncapturable, never pending capture", () => {
  // The sc-18212 defect, as a permanent mutation check in both directions: the SAME lane fixture
  // flips between "pending capture" and "uncapturable" purely on whether the adapter source carries
  // its arm — proving the categorization is derived from the dispatch, not from a hand list.
  const fixture = twoLaneFixture({
    extraLanes: [["candle:gamma", digest("d")]],
    plan: { providers: [planEntry("candle", "gamma"), planEntry("candle", "gamma", "candidate")] },
  });
  const armed = buildStaleLaneReport(fixture);
  assert.deepEqual(armed.unmeasuredLanes.map((lane) => lane.lane), ["candle:gamma"]);
  assert.deepEqual(armed.capturability.uncapturableLanes, []);
  assert.deepEqual(armed.unmeasuredLanes[0].plan, { entries: 2, authoritative: 1 });

  const disarmed = buildStaleLaneReport({
    ...fixture,
    adapterSources: { mlx: adapterSource(["alpha"]), candle: adapterSource(["beta"]) },
  });
  assert.deepEqual(disarmed.unmeasuredLanes, [], "an armless lane may not print as pending capture");
  const gamma = disarmed.uncapturableLanes.find((lane) => lane.lane === "candle:gamma");
  assert.equal(gamma.status, "uncapturable");
  assert.equal(gamma.declared, true);
  assert.deepEqual(gamma.plan, { entries: 2, authoritative: 1 });
  assert.equal(disarmed.totals.unmeasuredLanes, 0);
  assert.equal(disarmed.totals.uncapturableLanes, 1);
});

test("a planned lane the closure table never declared joins the universe, with its arm status", () => {
  // sc-18104 §2d rows one and four: candle:qwen_image_edit (planned, undeclared, armless) and
  // candle:qwen_image (planned, undeclared, arm exists) were invisible to the report entirely.
  const report = buildStaleLaneReport(
    twoLaneFixture({
      plan: {
        providers: [
          planEntry("candle", "delta", "candidate"),
          planEntry("candle", "gamma", "candidate"),
        ],
      },
    }),
  );
  assert.deepEqual(
    report.undeclaredLanes.map((lane) => [lane.lane, lane.capturable]),
    [["candle:delta", false], ["candle:gamma", true]],
  );
  assert.equal(report.totals.undeclaredLanes, 2);
  assert.deepEqual(report.capturability.uncapturableLanes, ["candle:delta"]);
  assert.ok(report.undeclaredLanes.every((lane) => lane.status === "undeclared"));
});

test("a non-eligible record neither counts as stale nor flips a lane out of pending capture", () => {
  // sc-18252: the currency population is `recordsNeedingDigest`'s — a fixture/candidate record
  // legitimately carries no closure digest. Counted naively it would (a) inflate the widened
  // evidence surface and (b) mark a never-captured lane as measured, hiding it from the
  // pending-capture list this report exists to keep honest.
  const report = buildStaleLaneReport(
    twoLaneFixture({
      extraLanes: [["candle:gamma", digest("d")]],
      extraRecords: [
        record({ id: "r-candidate", backend: "candle", provider: "gamma", modelId: "gamma_model", closureDigest: undefined, evidenceScope: "candidate" }),
        record({ id: "r-gated", backend: "candle", provider: "gamma", modelId: "gamma_model", closureDigest: undefined, status: "gated" }),
      ],
    }),
  );
  const gamma = report.unmeasuredLanes.find((lane) => lane.lane === "candle:gamma");
  assert.ok(gamma, "a lane with only non-eligible records is still pending capture");
  assert.deepEqual(gamma.records, { total: 0, stale: 0, current: 0, ineligible: 2 });
  assert.equal(gamma.impact.widenedEvidenceSurface, 0);
  assert.equal(report.totals.staleRecords, 3, "the non-eligible records added nothing stale");
});

test("the human report separates pending capture from uncapturable, and prints the derivation", () => {
  const fixture = twoLaneFixture({
    extraLanes: [["candle:gamma", digest("d")], ["candle:delta", digest("e")]],
    plan: { providers: [planEntry("candle", "gamma"), planEntry("candle", "delta"), planEntry("candle", "epsilon", "candidate")] },
    // No "beta" arm: the stale candle:beta lane doubles as the measured-but-armless case, so the
    // ranked table's CAPTURE column shows both values at once.
    adapterSources: { mlx: adapterSource(["alpha"]), candle: adapterSource(["gamma", "epsilon"]) },
  });
  const text = formatReport(buildStaleLaneReport(fixture));
  assert.match(text, /PENDING CAPTURE \(declared, adapter arm exists, no measurement yet\): candle:gamma \(1 plan entries, 1 authoritative\)/);
  assert.match(text, /DECLARED\/PLANNED BUT UNCAPTURABLE/);
  assert.match(text, /candle:delta\s+declared=yes\s+plan=1 entries \(1 authoritative\)\s+evidence=no evidence\s+status=uncapturable/);
  assert.match(text, /PLANNED BUT NEVER DECLARED/);
  assert.match(text, /candle:epsilon\s+plan=1 entries \(0 authoritative\)/);
  assert.match(text, /CAPTURE/);
  assert.match(text, /NO ARM/, "an armless stale lane is flagged in the ranked table");
  assert.ok(text.includes(CAPTURABILITY_SOURCE));
});
