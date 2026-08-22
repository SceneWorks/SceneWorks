import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { spawnSync } from "node:child_process";
import test from "node:test";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  FORMS,
  buildReport,
  buildVideoMemoryCurveBundle,
  coverageOf,
  driverStatesFrom,
  fitSlice,
  geometryHull,
  heldOutCoefficientTransfer,
  latentTemporalDepth,
  latentTokens,
  leastSquares,
  mergeVideoMemoryCurveLane,
  nonNegativeLeastSquares,
  noiseFloor,
  phaseFlipVerdict,
  pointsFrom,
  recordTerminalStatesFrom,
  rolesFromPlan,
  sessionsFrom,
  VIDEO_MEMORY_CURVE_CALIBRATION_ABI,
  VIDEO_MEMORY_CURVE_FIT_PATH_PATTERN,
  videoCurveSchemaErrors,
} from "./fit-ltx-temporal-form.mjs";
import { stripJsoncComments } from "./lib/jsonc.mjs";

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
const DATASET_RAW = readFileSync(
  path.join(ROOT, "docs/generated/ltx-mlx-geometry-sweep-sc-18810.json"),
  "utf8",
);
const DATASET = JSON.parse(DATASET_RAW);
const DATASET_SOURCES = [{
  path: "docs/generated/ltx-mlx-geometry-sweep-sc-18810.json",
  raw: DATASET_RAW,
}];
const MANIFEST = JSON.parse(
  stripJsoncComments(
    readFileSync(path.join(ROOT, "config/manifests/builtin.models.jsonc"), "utf8"),
  ),
);

function mixedCurveInputs() {
  const scaleActiveBytes = (record, factor) => {
    for (const phase of Object.values(record.observedMemory)) {
      if (Number.isSafeInteger(phase?.activeBytes)) phase.activeBytes *= factor;
    }
  };
  const cloneGroup = (suffix, mutateRecord) => {
    const records = structuredClone(DATASET.records).map((record) => {
      record.id = `imc-${createHash("sha256").update(`${record.id}:${suffix}`).digest("hex").slice(0, 20)}`;
      mutateRecord(record);
      return record;
    });
    return { records };
  };
  const q4 = cloneGroup(
    "q4",
    (record) => {
      record.target.tier = "q4";
      scaleActiveBytes(record, 2);
    },
  );
  const bounded = cloneGroup(
    "bounded",
    (record) => {
      record.strategy.rung = "bounded_decode";
      scaleActiveBytes(record, 3);
    },
  );
  const records = [...DATASET.records, ...q4.records, ...bounded.records];
  const mixedReport = buildReport(pointsFrom(records, rolesFromPlan(PLAN), MANIFEST));
  const sources = [
    { path: "evidence/a.json", raw: `${JSON.stringify({ records: DATASET.records.slice(0, 6) }, null, 2)}\n` },
    { path: "evidence/b.json", raw: `${JSON.stringify({ records: [...DATASET.records.slice(6), ...q4.records.slice(0, 5)] }, null, 2)}\n` },
    { path: "evidence/c.json", raw: `${JSON.stringify({ records: [...q4.records.slice(5), ...bounded.records] }, null, 2)}\n` },
  ];
  return { bounded, mixedReport, q4, records, sources };
}

const MLX_FIT = "docs/generated/ltx-temporal-form-fit-sc-18810.json";
const CANDLE_FIT = "docs/generated/wan-temporal-form-fit-sc-19057.json";
const CANDLE_SOURCE_PATH = "docs/generated/wan-candle-video-sc-19057.json";
const COMMITTED_BUNDLE = JSON.parse(
  readFileSync(path.join(ROOT, "docs/generated/video-memory-curves.json"), "utf8"),
);
/** The committed MLX lane, selected BY LANE — the container is multi-lane, so a positional
 * `curves[0]` would silently start asserting about a candle curve once sc-19057 lands. */
const COMMITTED_MLX = COMMITTED_BUNDLE.curves.find((curve) => curve.backend === "mlx");
/** Catalog entries that survive a candle promotion: everything the MLX lane still consumes. */
const COMMITTED_MLX_SOURCES = COMMITTED_BUNDLE.sourceCatalog.filter((entry) =>
  COMMITTED_MLX.evidence.sources.some((source) => source.path === entry.path),
);

/**
 * A synthetic `candle:wan2_2_ti2v_5b` promotion: the committed MLX sweep relabelled onto the Wan
 * candle identity sc-19057 will actually capture, with its own evidence file and its own fit
 * report. The COEFFICIENTS are irrelevant here — what these tests exercise is the container's
 * ability to hold this lane beside the MLX one.
 */
function candleLaneInputs() {
  const records = structuredClone(DATASET.records).map((record) => {
    record.id = `imc-${createHash("sha256").update(`${record.id}:candle`).digest("hex").slice(0, 20)}`;
    record.backend = "candle";
    record.target.modelId = "wan_2_2";
    record.target.provider = "wan2_2_ti2v_5b";
    record.calibrationFingerprint = "sc-19057-wan-2-2-ti2v-5b-candle-t2v-staged-capture-v1";
    return record;
  });
  return {
    records,
    report: buildReport(pointsFrom(records, rolesFromPlan(PLAN), MANIFEST)),
    sources: [{
      path: CANDLE_SOURCE_PATH,
      raw: `${JSON.stringify({ records }, null, 2)}\n`,
    }],
    sourceFit: CANDLE_FIT,
  };
}

const promoteCandleOnto = (existing) => {
  const { report, records, sources, sourceFit } = candleLaneInputs();
  return buildVideoMemoryCurveBundle(report, records, MANIFEST, sources, sourceFit, existing);
};

test("the Candle lane, not one story id, owns the non-negative cross-fit contract", () => {
  const { report } = candleLaneInputs();
  assert.equal(report.story, "sc-18810", "the fixture must not borrow SC-19057's story id");
  assert.equal(
    report.fits.maxPhaseActivePlusCache.q8.candidates.cross.coefficients.perMpxGb,
    0,
    "the known negative unconstrained MLX slope must be refit on the Candle boundary",
  );
  for (const byTier of Object.values(report.fits)) {
    for (const slice of Object.values(byTier)) {
      assert.ok(
        Object.values(slice.candidates.cross.coefficients).every((value) => value >= 0),
        "every Candle cross candidate must be non-negative by construction",
      );
    }
  }
});

test("promoting the candle lane preserves the committed MLX curve exactly", () => {
  // Lane-scoped: this test is about the MLX lane surviving a candle promotion, not about the
  // container holding nothing else. Once sc-19057 commits its candle curve the bundle holds two
  // lanes, and asserting a total of 1 here would red the very capture this container exists for.
  assert.equal(
    COMMITTED_BUNDLE.curves.filter((curve) => curve.backend === "mlx").length,
    1,
    "the committed bundle carries exactly one MLX curve",
  );
  const merged = promoteCandleOnto(COMMITTED_BUNDLE);

  assert.deepEqual(
    merged.curves.map((curve) => curve.backend).sort(),
    ["candle", "mlx"],
    "both lanes coexist — this is the whole point of schema v3",
  );
  const preservedMlx = merged.curves.find((curve) => curve.backend === "mlx");
  assert.deepEqual(
    preservedMlx,
    COMMITTED_MLX,
    "the MLX curve survives a foreign-lane promotion byte-for-byte, coefficients included",
  );
  assert.equal(preservedMlx.sourceFit, MLX_FIT);
  assert.equal(merged.curves.find((curve) => curve.backend === "candle").sourceFit, CANDLE_FIT);
  assert.deepEqual(
    merged.sourceCatalog,
    [
      ...COMMITTED_MLX_SOURCES,
      {
        path: CANDLE_SOURCE_PATH,
        sha256: createHash("sha256")
          .update(candleLaneInputs().sources[0].raw)
          .digest("hex"),
      },
    ].sort((left, right) => (left.path < right.path ? -1 : 1)),
    "the preserved lane's evidence source stays in the catalog beside the promoted lane's",
  );
  assert.deepEqual(
    merged.curves.map((curve) => curve.id),
    merged.curves.map((curve) => curve.id).slice().sort(),
    "the merged curve list stays deterministically ordered by id",
  );

  // The symmetric direction: re-promoting MLX must not delete the candle lane either.
  const report = JSON.parse(
    readFileSync(path.join(ROOT, "docs/generated/ltx-temporal-form-fit-sc-18810.json"), "utf8"),
  );
  const reMlx = buildVideoMemoryCurveBundle(
    report,
    DATASET.records,
    MANIFEST,
    DATASET_SOURCES,
    MLX_FIT,
    merged,
  );
  assert.deepEqual(reMlx, merged, "re-running one lane's fitter is idempotent for the other lane");
});

test("a lane promotion replaces its OWN lane rather than accumulating stale curves", () => {
  const merged = promoteCandleOnto(COMMITTED_BUNDLE);
  const { report, records, sources } = candleLaneInputs();
  // A second candle campaign, at a different closure, promoted from a different report.
  const nextRecords = structuredClone(records).map((record) => {
    record.repositories.inference.closureDigest = "a".repeat(64);
    return record;
  });
  const nextSources = [{
    path: "docs/generated/wan-candle-video-sc-19999.json",
    raw: `${JSON.stringify({ records: nextRecords }, null, 2)}\n`,
  }];
  const reCandle = buildVideoMemoryCurveBundle(
    buildReport(pointsFrom(nextRecords, rolesFromPlan(PLAN), MANIFEST)),
    nextRecords,
    MANIFEST,
    nextSources,
    "docs/generated/wan-temporal-form-fit-sc-19999.json",
    merged,
  );
  assert.equal(reCandle.curves.filter((curve) => curve.backend === "candle").length, 1);
  assert.equal(
    reCandle.curves.find((curve) => curve.backend === "candle").closureDigest,
    "a".repeat(64),
    "the superseded candle curve is gone, not left beside its replacement",
  );
  assert.deepEqual(
    reCandle.curves.find((curve) => curve.backend === "mlx"),
    COMMITTED_MLX,
    "replacing the candle lane twice still leaves MLX untouched",
  );
  assert.deepEqual(
    reCandle.sourceCatalog.map(({ path: sourcePath }) => sourcePath),
    [...COMMITTED_MLX_SOURCES.map(({ path: sourcePath }) => sourcePath),
      "docs/generated/wan-candle-video-sc-19999.json"].sort(),
    "the replaced lane's evidence source leaves the catalog with it, rather than being orphaned",
  );
  assert.ok(report && sources, "fixture builder returns its inputs");
});

test("a merge cannot silently resolve a cross-lane conflict", () => {
  const merged = promoteCandleOnto(COMMITTED_BUNDLE);
  const twoLanePromotion = { ...merged, sourceCatalog: merged.sourceCatalog, curves: merged.curves };
  assert.throws(
    () => mergeVideoMemoryCurveLane(COMMITTED_BUNDLE, twoLanePromotion),
    /replaces exactly one measurement lane, got 2/,
    "a promotion spanning two lanes has no well-defined replace target",
  );

  const straddling = structuredClone(merged);
  straddling.curves = straddling.curves.filter((curve) => curve.backend === "candle");
  straddling.sourceCatalog = [COMMITTED_BUNDLE.sourceCatalog[0]];
  assert.throws(
    () => mergeVideoMemoryCurveLane(COMMITTED_BUNDLE, straddling),
    /may not straddle two lanes/,
    "one evidence file feeding both lanes would orphan records the moment either lane is replaced",
  );

  const legacy = { ...COMMITTED_BUNDLE, schemaVersion: 2 };
  assert.throws(
    () => promoteCandleOnto(legacy),
    /schema v2, not v3/,
    "merging across schema versions must be an explicit migration, never a silent one",
  );

  const unattributed = structuredClone(COMMITTED_BUNDLE);
  // Strip provenance from a PRESERVED (non-candle) curve; stripping it from a curve the promotion
  // is about to replace would prove nothing.
  delete unattributed.curves.find((curve) => curve.backend === "mlx").sourceFit;
  assert.throws(
    () => promoteCandleOnto(unattributed),
    /carries no canonical fit provenance/,
    "a preserved curve with no fit report may not be carried forward unattributed",
  );

  const detachedCatalog = structuredClone(COMMITTED_BUNDLE);
  detachedCatalog.sourceCatalog.find(
    (entry) => entry.path === COMMITTED_MLX_SOURCES[0].path,
  ).sha256 = "0".repeat(64);
  assert.throws(
    () => promoteCandleOnto(detachedCatalog),
    /has two digests between the catalog and/,
    "a preserved source whose catalog digest disagrees with its curve is not preserved on trust",
  );

  assert.throws(
    () => promoteCandleOnto({ ...COMMITTED_BUNDLE, generatedBy: "scripts/other.mjs" }),
    /produced by another generator/,
  );
  assert.throws(() => promoteCandleOnto([]), /not a curve container/);

  // F4: the single-lane guard is a property of the PROMOTION, so a first-ever bundle (no existing
  // container to merge into) must not be the one path that can emit a mixed or empty one.
  const merged2 = promoteCandleOnto(COMMITTED_BUNDLE);
  assert.throws(
    () => mergeVideoMemoryCurveLane(null, merged2),
    /replaces exactly one measurement lane, got 2/,
    "a from-scratch two-lane promotion is refused, not waved through by the early return",
  );
  assert.throws(
    () => mergeVideoMemoryCurveLane(null, { ...merged2, curves: [] }),
    /replaces exactly one measurement lane, got 0/,
    "a from-scratch empty promotion is refused too",
  );
});

test("a promoted curve may not name a fit report outside the canonical family", () => {
  const { report, records, sources } = candleLaneInputs();
  for (const spelling of [
    "docs/generated/wan-temporal-form-fit.json",
    "docs/generated/nested/wan-temporal-form-fit-sc-19057.json",
    "../docs/generated/wan-temporal-form-fit-sc-19057.json",
    "docs/generated/WAN-temporal-form-fit-sc-19057.json",
    "",
  ]) {
    assert.throws(
      () => buildVideoMemoryCurveBundle(report, records, MANIFEST, sources, spelling, null),
      /sourceFit must be a canonical temporal-form fit report/,
      `${JSON.stringify(spelling)} must not be promotable as provenance`,
    );
  }
});

test("held-out tier analysis uses only the exact staged single-pass selector", () => {
  const { records } = mixedCurveInputs();
  const report = buildReport(
    pointsFrom(records, rolesFromPlan(PLAN), MANIFEST),
    null,
    new Map(),
    [],
    "sc-18946",
  );
  assert.equal(report.coefficientTransfer.verdict, "open");
  assert.deepEqual(report.coefficientTransfer.missingTiers, ["bf16"]);
  assert.equal(report.phaseFlip.tiers.q8.status, "measured");
  assert.equal(report.phaseFlip.tiers.q8.selector.rung, "staged_residency");
  assert.equal(report.phaseFlip.tiers.q8.selector.decodePass, "single_pass");

});

function analysisSelector(tier, phaseValues, roles = ["fit", "fit", "fit", "held_out"]) {
  const geometries = [
    geometry(768, 512, 121),
    geometry(768, 512, 241),
    geometry(1280, 704, 121),
    geometry(640, 640, 177),
  ];
  const points = geometries.map((entry, index) => ({
    fixture: `${tier}-${index}`,
    role: roles[index],
    geometry: entry,
    series: phaseValues[index],
  }));
  const candidate = {
    singular: false,
    coefficients: { fixedGb: 1, perMpxGb: 1, perMpxFrameGb: 0.01 },
    heldOut: { maxAbsGib: 0.5 },
  };
  return {
    selector: { tier, rung: "staged_residency", decodePass: "single_pass" },
    points,
    fits: Object.fromEntries(["text", "denoise", "decode"].map((phase) => [
      phase,
      { candidates: { cross: structuredClone(candidate) } },
    ])),
  };
}

test("coefficient transfer stays open unless q8 q4 and bf16 are all complete", () => {
  const values = [
    { text: 2, denoise: 3, decode: 4 },
    { text: 3, denoise: 4, decode: 5 },
    { text: 4, denoise: 5, decode: 6 },
    { text: 5, denoise: 6, decode: 7 },
  ];
  const q8 = analysisSelector("q8", values);
  const q4 = analysisSelector("q4", values);
  const incomplete = heldOutCoefficientTransfer([q8, q4]);
  assert.equal(incomplete.status, "insufficient_data");
  assert.equal(incomplete.verdict, "open");
  assert.deepEqual(incomplete.missingTiers, ["bf16"]);

  const bf16 = analysisSelector("bf16", values);
  delete bf16.fits.decode.candidates.cross;
  const missingPhase = heldOutCoefficientTransfer([q8, q4, bf16]);
  assert.equal(missingPhase.status, "insufficient_data");
  assert.equal(missingPhase.verdict, "open");
  assert.equal(missingPhase.temporalSlopeStatus, "insufficient_data");
  assert.equal(missingPhase.temporalSlopeVerdict, "open");
  assert.equal(missingPhase.phases.decode.bf16.status, "insufficient_data");
});

test("the tier phase-flip verdict compares q4 and bf16 only at matched scored geometries", () => {
  const q4Values = [
    { text: 8, denoise: 4, decode: 3 },
    { text: 7, denoise: 5, decode: 4 },
    { text: 9, denoise: 6, decode: 5 },
    { text: 8, denoise: 7, decode: 6 },
  ];
  const bf16Values = q4Values.map((value, index) => ({
    text: value.text,
    denoise: index === 2 ? value.text + 1 : value.denoise,
    decode: value.decode,
  }));
  const verdict = phaseFlipVerdict([
    analysisSelector("q4", q4Values),
    analysisSelector("bf16", bf16Values),
  ]);
  assert.equal(verdict.status, "measured");
  assert.equal(verdict.tierPhaseFlip.verdict, "tier_phase_flip_observed_at_matched_geometry");
  assert.equal(verdict.tierPhaseFlip.differingGeometries, 1);
  assert.equal(verdict.tierPhaseFlip.matchedGeometries.length, 4);
});

test("an exact phase tie keeps the tier phase-flip question open", () => {
  const q4Values = Array.from({ length: 4 }, () => ({ text: 8, denoise: 4, decode: 3 }));
  const bf16Values = structuredClone(q4Values);
  bf16Values[0].denoise = 8;
  const verdict = phaseFlipVerdict([
    analysisSelector("q4", q4Values),
    analysisSelector("bf16", bf16Values),
  ]);
  assert.equal(verdict.status, "insufficient_data");
  assert.equal(verdict.tierPhaseFlip.verdict, "open");
  assert.match(verdict.tierPhaseFlip.reason, /ambiguous binding phase/);
  assert.equal(verdict.phaseFlip, undefined);
  assert.equal(verdict.tiers.bf16.bindingCounts.ambiguous, 1);
});

test("video curve generation leaves the image calibration corpus byte-identical", () => {
  const imageCorpus = path.join(ROOT, "docs/generated/memory-calibration-evidence.json");
  const before = createHash("sha256").update(readFileSync(imageCorpus)).digest("hex");
  // Renewed for the sc-17137 main sync merge: the corpus gained the epic's 19 records (the
  // sc-19721 re-captures), which were themselves projected through the sc-18864 v5 alias rule
  // (deviceBytes/wiredBytes verbatim copies stripped) on ingest. The claim this test owns —
  // video curve generation never mutates the image corpus — is the before/after equality below;
  // this pin only names the reviewed snapshot.
  assert.equal(
    before,
    "5b7b48f127aa0339c73876f1babb2117eb5cfa32c2405fce0133c539192c9538",
    "the explicit pre-video image evidence outcome remains the reviewed corpus",
  );
  const output = mkdtempSync(path.join(tmpdir(), "sceneworks-video-curves-"));
  try {
    const run = spawnSync(
      process.execPath,
      [
        path.join(ROOT, "scripts/fit-ltx-temporal-form.mjs"),
        "--write", path.join(output, "fit.json"),
        "--curve-write", path.join(output, "curves.json"),
      ],
      { cwd: ROOT, encoding: "utf8" },
    );
    assert.equal(run.status, 0, run.stderr);
    const after = createHash("sha256").update(readFileSync(imageCorpus)).digest("hex");
    assert.equal(after, before, "the video-only producer must not rewrite the image evidence corpus");
  } finally {
    rmSync(output, { recursive: true, force: true });
  }
});

test("the committed report and curve pass the producer's --check round trip", () => {
  const run = spawnSync(
    process.execPath,
    [path.join(ROOT, "scripts/fit-ltx-temporal-form.mjs"), "--check"],
    { cwd: ROOT, encoding: "utf8" },
  );
  assert.equal(run.status, 0, run.stderr);
  assert.match(run.stdout, /video-memory-curves\.json are current/);
});

test("the immutable SC-19057 records promote the constrained cross fit and preserve MLX", () => {
  const args = [
    path.join(ROOT, "scripts/fit-ltx-temporal-form.mjs"),
    "--story", "sc-19057",
    "--dataset", "docs/generated/wan-candle-video-sc-19057.json",
    "--plan", "docs/calibration/sc-19057/wan-candle-video-capture-plan.json",
    "--record-terminals",
    "--write", "docs/generated/wan-temporal-form-fit-sc-19057.json",
    "--source-fit", "docs/generated/wan-temporal-form-fit-sc-19057.json",
    "--check",
  ];
  const run = spawnSync(process.execPath, args, { cwd: ROOT, encoding: "utf8" });
  assert.equal(run.status, 0, run.stderr);

  const report = JSON.parse(
    readFileSync(path.join(ROOT, "docs/generated/wan-temporal-form-fit-sc-19057.json"), "utf8"),
  );
  assert.deepEqual(report.terminalProvenance, {
    mode: "record_terminals",
    authority: "runtime_complete_records_from_clean_repositories",
  });
  assert.deepEqual(
    [report.capturedRecords, report.coverage.plannedEntries, report.coverage.capturedFixtures],
    [6, 6, 6],
  );
  assert.deepEqual(report.fits.denoise.q4.candidates.cross.coefficients, {
    fixedGb: 11.207667472795762,
    perMpxGb: 0,
    perMpxFrameGb: 0.00025117197517360795,
  });
  assert.equal(report.fits.denoise.q4.candidates.cross.heldOut.maxAbsGib, 0.013348825976386536);
  assert.deepEqual(report.fits.decode.q4.candidates.cross.coefficients, {
    fixedGb: 7.6801715507055235,
    perMpxGb: 0,
    perMpxFrameGb: 0.06946188490884356,
  });
  assert.equal(report.fits.decode.q4.candidates.cross.fit.maxAbsGib, 2.057363514575272);
  assert.equal(report.fits.decode.q4.candidates.cross.heldOut.maxAbsGib, 1.9182051277155239);

  const bundle = JSON.parse(
    readFileSync(path.join(ROOT, "docs/generated/video-memory-curves.json"), "utf8"),
  );
  const mlx = bundle.curves.find((curve) => curve.backend === "mlx");
  const candle = bundle.curves.find((curve) => curve.backend === "candle");
  assert.equal(
    createHash("sha256").update(`${JSON.stringify(mlx)}\n`).digest("hex"),
    "6f0c94b8fa3a5bb6b1fb6df964d5a23171a8ff50ee231e8e1084c55c3e32fd93",
    "the SC-19057 promotion may not rewrite the independently fitted MLX lane",
  );
  assert.equal(candle.evidence.sources[0].sha256, "1eb425b3bb795a6b1b5408be6888e9f20f94b7de7a8821209b70779286ef8909");
  assert.deepEqual(candle.phases.denoise, {
    ...report.fits.denoise.q4.candidates.cross.coefficients,
    maxResidualGb: 0.013348825976386536,
  });
});

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

test("the promoted curve container is derived from the exact sc-18810 identity and cross fits", () => {
  const report = JSON.parse(
    readFileSync(path.join(ROOT, "docs/generated/ltx-temporal-form-fit-sc-18810.json"), "utf8"),
  );
  const bundle = buildVideoMemoryCurveBundle(report, DATASET.records, MANIFEST, DATASET_SOURCES);
  assert.equal(bundle.schemaVersion, 3);
  assert.equal(bundle.sourceFit, undefined, "fit provenance is per-curve since v3, not per-bundle");
  assert.equal(bundle.curves.length, 1);
  const curve = bundle.curves[0];
  assert.equal(curve.sourceFit, "docs/generated/ltx-temporal-form-fit-sc-18810.json");
  assert.deepEqual(
    {
      modelId: curve.modelId,
      modelFamily: curve.modelFamily,
      provider: curve.provider,
      backend: curve.backend,
      tier: curve.tier,
      mode: curve.mode,
      rung: curve.rung,
      loadShape: curve.loadShape,
      batch: curve.batch,
      closureDigest: curve.closureDigest,
      calibrationAbi: curve.calibrationAbi,
      calibrationFingerprint: curve.calibrationFingerprint,
      decodePass: curve.decodePass,
    },
    {
      modelId: "ltx_2_3",
      modelFamily: "ltx-video",
      provider: "ltx_2_3",
      backend: "mlx",
      tier: "q8",
      mode: "text_to_video",
      rung: "staged_residency",
      loadShape: "eager_materialization",
      batch: 1,
      closureDigest: DATASET.records[0].repositories.inference.closureDigest,
      calibrationAbi: VIDEO_MEMORY_CURVE_CALIBRATION_ABI,
      calibrationFingerprint: DATASET.records[0].calibrationFingerprint,
      decodePass: "single_pass",
    },
  );
  assert.deepEqual(
    {
      text: report.fits.text.q8.chosen,
      denoise: report.fits.denoise.q8.chosen,
      decode: report.fits.decode.q8.chosen,
    },
    { text: "area_only", denoise: "latent_tokens", decode: "cross" },
    "the fit report's generic winner is deliberately not the ratified cross-curve selector",
  );
  for (const [wirePhase, fitPhase] of [
    ["conditioning", "text"],
    ["denoise", "denoise"],
    ["decode", "decode"],
  ]) {
    assert.deepEqual(
      Object.fromEntries(
        Object.entries(curve.phases[wirePhase]).filter(([key]) => key !== "maxResidualGb"),
      ),
      report.fits[fitPhase].q8.candidates.cross.coefficients,
      `${wirePhase} must carry the ratified cross coefficients verbatim`,
    );
  }
  assert.equal(curve.evidence.records, DATASET.records.length);
  assert.deepEqual(
    curve.evidence.sources[0].recordIds,
    DATASET.records.map((record) => record.id).sort(),
    "the runtime artifact must name every immutable source record",
  );
  assert.equal(
    curve.evidence.sources[0].sha256,
    createHash("sha256").update(DATASET_RAW).digest("hex"),
    "the evidence digest must bind every byte of the committed source evidence",
  );
});

test("the video curve ABI stays pinned to SceneWorks' gen-core mirror", () => {
  const source = readFileSync(
    path.join(ROOT, "crates/sceneworks-core/src/memory_calibration.rs"),
    "utf8",
  );
  const abi = source.match(/pub const MEMORY_CALIBRATION_ABI: u32 = (\d+);/);
  assert.ok(abi, "SceneWorks memory calibration ABI constant exists");
  assert.equal(Number(abi[1]), VIDEO_MEMORY_CURVE_CALIBRATION_ABI);
});

test("the persisted video curve has an explicit strict schema contract", () => {
  const schema = JSON.parse(
    readFileSync(path.join(ROOT, "packages/schemas/video-memory-curves.schema.json"), "utf8"),
  );
  const bundle = JSON.parse(
    readFileSync(path.join(ROOT, "docs/generated/video-memory-curves.json"), "utf8"),
  );
  assert.equal(schema.$schema, "https://json-schema.org/draft/2020-12/schema");
  assert.equal(schema.properties.schemaVersion.const, 3);
  assert.equal(schema.properties.generatedBy.const, "scripts/fit-ltx-temporal-form.mjs");
  assert.equal(
    schema.properties.sourceFit,
    undefined,
    "v3 moved fit provenance onto the curve so the container can hold more than one lane",
  );
  assert.equal(
    schema.$defs.curve.properties.sourceFit.pattern,
    "^docs/generated/[a-z0-9]+(?:-[a-z0-9]+)*-temporal-form-fit-sc-[0-9]+\\.json$",
  );
  // F6: tie the producer's own predicate to the schema pattern. Without this, a producer that
  // drifted STRICTER than the schema would never be caught — schema validation at promotion only
  // catches a producer that drifted looser.
  // Compared through the same RegExp normalization on both sides: `.source` escapes forward
  // slashes, so comparing it against the raw schema string would fail on spelling alone.
  assert.equal(
    VIDEO_MEMORY_CURVE_FIT_PATH_PATTERN.source,
    new RegExp(schema.$defs.curve.properties.sourceFit.pattern).source,
    "the producer predicate and the schema pattern are one contract, not two spellings",
  );
  assert.ok(schema.$defs.curve.required.includes("sourceFit"));
  assert.deepEqual(schema.required, [
    "schemaVersion",
    "generatedBy",
    "sourceCatalog",
    "curves",
  ]);
  assert.equal(schema.additionalProperties, false);
  assert.deepEqual(videoCurveSchemaErrors(schema, bundle), []);

  const unknown = structuredClone(bundle);
  unknown.curves[0].phases.decode.perFrameGb = 1;
  assert.match(videoCurveSchemaErrors(schema, unknown).join("\n"), /unknown property "perFrameGb"/);
  const staleAbi = structuredClone(bundle);
  staleAbi.curves[0].calibrationAbi += 1;
  assert.match(videoCurveSchemaErrors(schema, staleAbi).join("\n"), /expected constant 3/);
  // The generalized pattern is still a constraint, not a rubber stamp.
  const bundleLevel = structuredClone(bundle);
  delete bundleLevel.curves[0].sourceFit;
  bundleLevel.sourceFit = "docs/generated/ltx-temporal-form-fit-sc-18810.json";
  assert.match(
    videoCurveSchemaErrors(schema, bundleLevel).join("\n"),
    /unknown property "sourceFit"[\s\S]*missing required property "sourceFit"/,
    "a v2-shaped bundle-level sourceFit is refused at BOTH ends",
  );
  for (const spelling of [
    "docs/generated/wan-temporal-form-fit.json",
    "docs/generated/nested/wan-temporal-form-fit-sc-19057.json",
    "../docs/generated/wan-temporal-form-fit-sc-19057.json",
    "docs/generated/WAN-temporal-form-fit-sc-19057.json",
  ]) {
    const malformed = structuredClone(bundle);
    malformed.curves[0].sourceFit = spelling;
    assert.match(
      videoCurveSchemaErrors(schema, malformed).join("\n"),
      /sourceFit: does not match/,
      `${spelling} must fail the fit-report family pattern`,
    );
  }
  assert.deepEqual(
    videoCurveSchemaErrors(schema, {
      ...bundle,
      curves: bundle.curves.map((curve) => ({
        ...curve,
        sourceFit: "docs/generated/wan2-2-ti2v-5b-temporal-form-fit-sc-19057.json",
      })),
    }),
    [],
    "a candle campaign's own report name is expressible, which v2's ltx- pin made impossible",
  );
});

test("the applicability hull is the convex measured area-by-voxel hull, not a loose bounding box", () => {
  const hull = geometryHull([
    { width: 768, height: 512, frames: 121 },
    { width: 768, height: 512, frames: 241 },
    { width: 512, height: 768, frames: 361 },
    { width: 640, height: 640, frames: 177 },
    { width: 1280, height: 704, frames: 121 },
    { width: 1280, height: 704, frames: 145 },
    { width: 704, height: 1280, frames: 177 },
  ]);
  assert.deepEqual(hull, [
    { pixels: 393216, voxels: 47579136 },
    { pixels: 901120, voxels: 109035520 },
    { pixels: 901120, voxels: 159498240 },
    { pixels: 393216, voxels: 141950976 },
  ]);
  assert.ok(
    !hull.some(({ pixels, voxels }) => pixels === 409600 && voxels === 72499200),
    "the held-out 640-square point is inside the hull, not a redundant vertex",
  );
  assert.throws(
    () => geometryHull([
      { width: Number.MAX_SAFE_INTEGER, height: 2, frames: 1 },
      { width: 1, height: 1, frames: 1 },
      { width: 2, height: 2, frames: 2 },
    ]),
    /exceeds exact JSON integer arithmetic/,
  );
});

test("mixed complete selectors and sources produce deterministic independent curves", () => {
  const { bounded, mixedReport, q4, records, sources } = mixedCurveInputs();
  const forward = buildVideoMemoryCurveBundle(mixedReport, records, MANIFEST, sources);
  const reversed = buildVideoMemoryCurveBundle(
    { ...mixedReport, observations: mixedReport.observations.slice().reverse() },
    records.slice().reverse(),
    MANIFEST,
    sources.slice().reverse(),
  );
  assert.deepEqual(reversed, forward, "record and source order must not affect emitted bytes");
  assert.deepEqual(
    JSON.parse(JSON.stringify(forward)),
    forward,
    "the multi-curve artifact must round-trip through its wire representation",
  );
  assert.equal(forward.curves.length, 3);
  assert.deepEqual(forward.sourceCatalog.map(({ path }) => path), ["evidence/a.json", "evidence/b.json", "evidence/c.json"]);
  const original = forward.curves.find((curve) => curve.tier === "q8" && curve.rung === "staged_residency");
  const q4Curve = forward.curves.find((curve) => curve.tier === "q4");
  const boundedCurve = forward.curves.find((curve) => curve.rung === "bounded_decode");
  assert.deepEqual(original.evidence.sources.map(({ path }) => path), ["evidence/a.json", "evidence/b.json"]);
  assert.deepEqual(q4Curve.evidence.sources.map(({ path }) => path), ["evidence/b.json", "evidence/c.json"]);
  assert.deepEqual(boundedCurve.evidence.sources.map(({ path }) => path), ["evidence/c.json"]);
  assert.deepEqual(
    q4Curve.evidence.sources.flatMap(({ recordIds }) => recordIds).sort(),
    q4.records.map(({ id }) => id).sort(),
  );
  assert.ok(
    Math.abs(
      q4Curve.phases.decode.perMpxFrameGb - original.phases.decode.perMpxFrameGb * 2,
    ) < 1e-12,
    "q4 is fitted from its own scaled records rather than a tier/rung mixture",
  );
  assert.ok(
    Math.abs(
      boundedCurve.phases.decode.perMpxFrameGb - original.phases.decode.perMpxFrameGb * 3,
    ) < 1e-12,
    "bounded decode is fitted independently even though it shares q8 tier identity",
  );
  assert.equal(mixedReport.selectorFits.length, 3);
  assert.deepEqual(
    mixedReport.legacyFitsOmittedForTiers,
    ["q8"],
    "the legacy tier-only view may not pool two complete q8 selectors",
  );
});

test("multi-dataset CLI output stays current when source order is reversed", () => {
  const { records, sources } = mixedCurveInputs();
  const target = path.join(ROOT, "target");
  mkdirSync(target, { recursive: true });
  const output = mkdtempSync(path.join(target, "sceneworks-mixed-video-curves-"));
  try {
    const datasetPaths = sources.map((source, index) => {
      const file = path.join(output, `${index}.json`);
      writeFileSync(file, source.raw);
      return file;
    });
    const providerNameByFixture = new Map(
      PLAN.providers.map((provider) => [provider.fixture, provider.name]),
    );
    const logPath = path.join(output, "mixed.log");
    const capturedLines = records.flatMap((record) => {
      const name = providerNameByFixture.get(record.fixture);
      assert.ok(name, `plan provider exists for ${record.fixture}`);
      return [`BEGIN ${name} tier=${record.target.tier} free=512GiB 10:00:00`, `OK ${name} 1s`];
    });
    const failedHostLines = PLAN.providers
      .filter((provider) => provider._role === "attempted_failed_host_limit")
      .flatMap((provider) => [
        `BEGIN ${provider.name} tier=${provider.target.tier} free=1GiB 10:00:00`,
        `FAIL ${provider.name} 1s :: synthetic host limit`,
      ]);
    writeFileSync(
      logPath,
      [...capturedLines, ...failedHostLines].join("\n") + "\n",
    );
    const reportPath = path.join(output, "fit.json");
    const curvePath = path.join(output, "curves.json");
    const args = (orderedPaths, check = false) => [
      path.join(ROOT, "scripts/fit-ltx-temporal-form.mjs"),
      "--story", "sc-18946",
      "--plan", path.join(ROOT, "docs/calibration/sc-18810/ltx-mlx-geometry-sweep.json"),
      ...orderedPaths.flatMap((file) => ["--dataset", file]),
      "--driver-log", logPath,
      "--write", reportPath,
      "--curve-write", curvePath,
      ...(check ? ["--check"] : []),
    ];
    const generated = spawnSync(process.execPath, args(datasetPaths), {
      cwd: ROOT,
      encoding: "utf8",
    });
    assert.equal(generated.status, 0, generated.stderr);
    const checked = spawnSync(process.execPath, args(datasetPaths.slice().reverse(), true), {
      cwd: ROOT,
      encoding: "utf8",
    });
    assert.equal(checked.status, 0, checked.stderr);
    assert.match(checked.stdout, /curves\.json are current/);
    assert.equal(JSON.parse(readFileSync(curvePath, "utf8")).curves.length, 3);
    const generatedReport = JSON.parse(readFileSync(reportPath, "utf8"));
    assert.equal(generatedReport.story, "sc-18946");
    assert.deepEqual(generatedReport.terminalProvenance, {
      mode: "driver_logs",
      logs: [path.relative(ROOT, logPath)],
    });
    assert.ok(generatedReport.coefficientTransfer);
    assert.ok(generatedReport.phaseFlip);
    assert.deepEqual(
      [...new Set(JSON.parse(readFileSync(curvePath, "utf8")).curves.map((curve) => curve.sourceFit))],
      ["docs/generated/ltx-temporal-form-fit-sc-18946.json"],
      "every curve this campaign promoted names this campaign's report",
    );
    assert.equal(generatedReport.selectorFits.length, 3);
    assert.deepEqual(generatedReport.legacyFitsOmittedForTiers, ["q8"]);
    assert.equal(
      generatedReport.noiseFloors.decode.replicatedGeometries,
      12,
      "four replicate geometries in each of three selectors stay twelve same-cell groups",
    );
  } finally {
    rmSync(output, { recursive: true, force: true });
  }
});

test("the spawned CLI accepts exactly one terminal provenance mode and rejects both or neither", () => {
  const target = path.join(ROOT, "target");
  mkdirSync(target, { recursive: true });
  const output = mkdtempSync(path.join(target, "sceneworks-terminal-provenance-"));
  try {
    const dataset = structuredClone(DATASET);
    for (const record of dataset.records) {
      record.status = "runtime_complete";
      record.repositories.sceneWorks.dirty = false;
      record.repositories.inference.dirty = false;
    }
    const fixtures = new Set(dataset.records.map((record) => record.fixture));
    const plan = {
      ...structuredClone(PLAN),
      providers: PLAN.providers.filter((provider) => fixtures.has(provider.fixture)),
    };
    const datasetPath = path.join(output, "records.json");
    const planPath = path.join(output, "plan.json");
    writeFileSync(datasetPath, `${JSON.stringify(dataset, null, 2)}\n`);
    writeFileSync(planPath, `${JSON.stringify(plan, null, 2)}\n`);

    const run = (...modeArgs) => {
      const suffix = modeArgs.includes("--record-terminals") ? "records" : "logs";
      const selectedPlan = modeArgs.includes("--record-terminals")
        ? planPath
        : path.join(ROOT, "docs/calibration/sc-18810/ltx-mlx-geometry-sweep.json");
      return spawnSync(process.execPath, [
        path.join(ROOT, "scripts/fit-ltx-temporal-form.mjs"),
        "--story", "sc-18946",
        "--dataset", datasetPath,
        "--plan", selectedPlan,
        "--write", path.join(output, `${suffix}-fit.json`),
        "--curve-write", path.join(output, `${suffix}-curves.json`),
        ...modeArgs,
      ], { cwd: ROOT, encoding: "utf8" });
    };

    const recordMode = run("--record-terminals");
    assert.equal(recordMode.status, 0, recordMode.stderr);
    const recordReport = JSON.parse(readFileSync(path.join(output, "records-fit.json"), "utf8"));
    assert.deepEqual(recordReport.terminalProvenance, {
      mode: "record_terminals",
      authority: "runtime_complete_records_from_clean_repositories",
    });
    assert.deepEqual(
      recordReport.sourceSessions,
      [],
      "record-terminal provenance must not fabricate a driver-log session",
    );

    const driverMode = run("--driver-log", path.join(ROOT, LOG_PATHS[0]),
      "--driver-log", path.join(ROOT, LOG_PATHS[1]));
    assert.equal(driverMode.status, 0, driverMode.stderr);
    const driverReport = JSON.parse(readFileSync(path.join(output, "logs-fit.json"), "utf8"));
    assert.deepEqual(driverReport.terminalProvenance, {
      mode: "driver_logs",
      logs: LOG_PATHS,
    });
    assert.ok(driverReport.sourceSessions.length > 0);

    const both = run("--record-terminals", "--driver-log", path.join(ROOT, LOG_PATHS[0]));
    assert.notEqual(both.status, 0);
    assert.match(both.stderr, /mutually exclusive provenance modes/);

    const neither = run();
    assert.notEqual(neither.status, 0);
    assert.match(neither.stderr, /fit needs terminal provenance/);
  } finally {
    rmSync(output, { recursive: true, force: true });
  }
});

test("duplicate or malformed source record identities cannot be promoted", () => {
  const report = JSON.parse(
    readFileSync(path.join(ROOT, "docs/generated/ltx-temporal-form-fit-sc-18810.json"), "utf8"),
  );
  assert.throws(
    () => buildVideoMemoryCurveBundle(report, DATASET.records, MANIFEST, DATASET_RAW),
    /one or more exact source evidence inputs/,
    "raw bytes without their immutable source path may not inherit a guessed legacy path",
  );
  assert.throws(
    () => buildVideoMemoryCurveBundle(report, DATASET.records, MANIFEST, [{
      path: "../detached.json",
      raw: DATASET_RAW,
    }]),
    /canonical repository-relative path/,
    "source paths must be stable artifact identities rather than host-relative escapes",
  );
  const records = structuredClone(DATASET.records);
  records[0].id = records[1].id;
  assert.throws(
    () => buildVideoMemoryCurveBundle(report, records, MANIFEST, DATASET_SOURCES),
    /unique immutable imc ids/,
  );

  const duplicateObservation = structuredClone(report);
  duplicateObservation.observations[0] = duplicateObservation.observations[1];
  assert.throws(
    () => buildVideoMemoryCurveBundle(duplicateObservation, DATASET.records, MANIFEST, DATASET_SOURCES),
    /observations do not match the immutable promoted record ids/,
  );

  const detachedObservation = structuredClone(report);
  detachedObservation.observations[0].activeGib.decode += 1;
  assert.throws(
    () => buildVideoMemoryCurveBundle(detachedObservation, DATASET.records, MANIFEST, DATASET_SOURCES),
    /does not match its immutable source record/,
    "report-only phase mutations must not alter a curve detached from source evidence",
  );

  const detachedSelectorSubset = structuredClone(report);
  detachedSelectorSubset.selectorFits[0].recordIds.pop();
  assert.throws(
    () => buildVideoMemoryCurveBundle(detachedSelectorSubset, DATASET.records, MANIFEST, DATASET_SOURCES),
    /fit report record subset is detached/,
    "a selector fit must name exactly the immutable records its coefficients consume",
  );

  const wrongLegacyFit = structuredClone(report);
  wrongLegacyFit.selectorFits[0].legacyFitTier = "q4";
  assert.throws(
    () => buildVideoMemoryCurveBundle(wrongLegacyFit, DATASET.records, MANIFEST, DATASET_SOURCES),
    /malformed or duplicate complete selector/,
    "a selector may not borrow another tier's compatibility fit",
  );

  const missingDecodeIdentity = structuredClone(DATASET.records);
  missingDecodeIdentity[0].diagnostics.measurements = missingDecodeIdentity[0].diagnostics.measurements
    .filter(({ name }) => name !== "decodeTilingEngaged");
  assert.throws(
    () => buildVideoMemoryCurveBundle(report, missingDecodeIdentity, MANIFEST, [{
      path: "evidence/missing-decode.json",
      raw: `${JSON.stringify({ records: missingDecodeIdentity }, null, 2)}\n`,
    }]),
    /exact decodeTilingEngaged measurement/,
  );
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

test("cross NNLS preserves valid OLS bytes and refits the active boundary instead of clamping", () => {
  const increasingRows = [[1, 0], [1, 1], [1, 2]];
  const increasingTargets = [1, 2, 3];
  assert.deepEqual(
    nonNegativeLeastSquares(increasingRows, increasingTargets),
    leastSquares(increasingRows, increasingTargets),
    "a historical positive fit must retain the exact ordinary-least-squares doubles",
  );

  const decreasingTargets = [3, 2, 1];
  const unconstrained = leastSquares(increasingRows, decreasingTargets);
  assert.deepEqual(unconstrained, [3, -1], "the fixture must actually leave the monotone domain");
  const constrained = nonNegativeLeastSquares(increasingRows, decreasingTargets);
  assert.deepEqual(
    constrained,
    [2, 0],
    "the active intercept must be refit at the non-negative boundary; [3, 0] would be a clamp",
  );
  assert.ok(constrained.every((coefficient) => coefficient >= 0));
});

test("cross NNLS searches every active set and breaks exact ties by stable mask order", () => {
  const rows = [
    [1, 0, 0],
    [1, 1, 0],
    [1, 0, 1],
    [1, 1, 1],
  ];
  const targets = [4, 2, 7, 5];
  assert.deepEqual(
    nonNegativeLeastSquares(rows, targets),
    [3, 0, 3],
    "the optimum is the intercept+third-column face, not the first non-negative partial fit",
  );

  assert.deepEqual(
    nonNegativeLeastSquares([[1, 1]], [1]),
    [1, 0],
    "equal single-column faces choose the lower active-set mask deterministically",
  );
  assert.equal(
    nonNegativeLeastSquares([[1, Number.NaN]], [1]),
    null,
    "non-finite regressors must not yield a fabricated boundary solution",
  );
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

test("an explicit arithmetic-unmeasurable terminal is distinct from a host failure and never run", () => {
  const providers = structuredClone(PLAN.providers.slice(0, 3));
  const [arithmetic, hostFailure, neverRun] = providers;
  const states = driverStatesFrom([
    [
      `ARITHMETIC_UNMEASURABLE ${arithmetic.name} :: exact predicted floor exceeds the host budget`,
      `BEGIN ${hostFailure.name} tier=${hostFailure.target.tier} free=1GiB 10:00:01`,
      `FAIL ${hostFailure.name} 1s :: process killed by the host`,
    ].join("\n"),
  ]);
  const coverage = coverageOf({ providers }, [], states);
  const stateOf = (provider) =>
    coverage.entries.find((entry) => entry.fixture === provider.fixture).state;
  assert.equal(stateOf(arithmetic), "arithmetic_unmeasurable");
  assert.equal(states.get(arithmetic.name).begins, 0, "arithmetic proof is not mislabeled as a run");
  assert.equal(stateOf(hostFailure), "attempted_failed_host_limit");
  assert.equal(stateOf(neverRun), "not_reached");
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

test("record terminals are explicit and require clean runtime-complete harness records", () => {
  const record = structuredClone(DATASET.records[0]);
  const provider = PLAN.providers.find((entry) => entry.fixture === record.fixture);
  assert.ok(provider);
  record.status = "runtime_complete";
  record.repositories.sceneWorks.dirty = false;
  record.repositories.inference.dirty = false;
  const oneProviderPlan = { providers: [provider] };
  const states = recordTerminalStatesFrom([record], oneProviderPlan);
  assert.equal(states.get(provider.name).terminal, "completed");
  assert.equal(states.get(provider.name).oks, 1);
  assert.equal(
    coverageOf(
      oneProviderPlan,
      pointsFrom([record], rolesFromPlan(PLAN), MANIFEST),
      states,
    ).byState.captured.total,
    1,
  );

  for (const mutate of [
    (candidate) => { candidate.status = "gated"; },
    (candidate) => { candidate.repositories.sceneWorks.dirty = true; },
    (candidate) => { candidate.repositories.inference.dirty = true; },
  ]) {
    const candidate = structuredClone(record);
    mutate(candidate);
    assert.throws(
      () => recordTerminalStatesFrom([candidate], oneProviderPlan),
      /runtime_complete|dirty repository/,
    );
  }

  assert.throws(
    () => recordTerminalStatesFrom([], oneProviderPlan),
    /requires the exact planned fixture set/,
    "a missing runtime-complete row must not become a not_reached row that still promotes",
  );
  assert.throws(
    () => recordTerminalStatesFrom([{ ...record, fixture: "not-in-the-plan" }], oneProviderPlan),
    /requires the exact planned fixture set/,
    "an unexpected runtime-complete row must not be promoted outside the declared plan",
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
