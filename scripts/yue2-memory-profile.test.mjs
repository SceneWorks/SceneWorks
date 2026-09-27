// sc-23001: the YuE2 memory-profile harness — plan, identity keying, stage attribution, currency.
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import {
  BACKENDS,
  CUDA_RESERVE_BYTES,
  REQUEST_KEYS,
  ROOT,
  admitToCorpus,
  buildRecord,
  cargoTestArgv,
  caseFile,
  caseIdentity,
  corpusFileName,
  corpusStatus,
  coverage,
  expandCases,
  gradeRecord,
  parseArgs,
  parseStageMarks,
  parseWatchdogSamples,
  planCapture,
  readSources,
  stagePeaks,
  validatePlan,
  validateRecord,
} from "./yue2-memory-profile.mjs";

const sources = await readSources();
const clone = (value) => JSON.parse(JSON.stringify(value));
const GIB = 1024 ** 3;

function record(item, overrides = {}) {
  const declared = sources.closures.providers[sources.plan.lane];
  const marks = [
    { stage: "load", at: 10 },
    { stage: "plan", at: 20 },
    { stage: "semantic", at: 30 },
    { stage: "acoustic", at: 40 },
    { stage: "decode", at: 50 },
    { stage: "done", at: 60 },
  ];
  const samples = [
    { at: 5, bytes: 1 * GIB },
    { at: 15, bytes: 8 * GIB },
    { at: 25, bytes: 9 * GIB },
    { at: 35, bytes: 10 * GIB },
    { at: 45, bytes: 11 * GIB },
    { at: 55, bytes: 12 * GIB },
    { at: 65, bytes: 99 * GIB },
  ];
  const estimate = (total) => ({ deviceBytes: total, hostBytes: 0, totalBytes: total });
  return buildRecord({
    item,
    identity: caseIdentity(item, sources.manifest),
    engine: {
      lane: sources.plan.lane,
      inferenceRevision: "a".repeat(40),
      closureDigest: declared.digest,
      digestVersion: sources.closures.digestVersion,
    },
    sceneworks: { revision: "b".repeat(40), dirty: false },
    hardware: { memoryBytes: 64 * GIB },
    admission: {
      outcome: "admitted",
      estimate: {
        stages: {
          load: estimate(9 * GIB),
          plan: estimate(10 * GIB),
          semantic: estimate(10 * GIB),
          acoustic: estimate(12 * GIB),
          decode: estimate(11 * GIB),
        },
      },
    },
    outcome: { status: "completed", audioSeconds: 30, rms: 0.1 },
    sampler: "synthetic",
    samples,
    marks,
    capturedAt: "2026-09-26T00:00:00Z",
    ...overrides,
  });
}

test("the checked-in plan validates against the catalog and closure table and expands to its cases", () => {
  validatePlan(sources.plan, sources);
  const cases = expandCases(sources.plan);
  // Every backend measures every catalog tier at the product default request.
  for (const backend of BACKENDS) {
    const tiers = cases
      .filter((item) => item.backend === backend && item.requestName === "default")
      .map((item) => item.tier)
      .sort();
    assert.deepEqual(tiers, ["bf16", "q4", "q8"], backend);
  }
  for (const item of cases) {
    assert.match(item.id, /^yue2:(bf16|q8|q4):(metal|cuda):[a-z][a-z0-9-]*$/);
    const identity = caseIdentity(item, sources.manifest);
    const entry = sources.manifest.models.find((model) => model.id === "yue2");
    const row = entry.downloads.find((download) => !download.coRequisite && download.variant === item.tier);
    assert.equal(identity.model.repo, row.repo);
    assert.equal(identity.model.revision, row.revision);
    // A derived tier is identified by its pinned weights; the released bf16 by repo@revision.
    assert.equal(identity.model.weightsSha256 === null, item.tier === "bf16", item.id);
    assert.equal(identity.decoder.componentId, item.decoder === "legacy" ? "vae_legacy" : "vae");
  }
});

test("the plan refuses an undeclared lane, a non-catalog tier and a field the entrypoint cannot read", () => {
  const undeclared = clone(sources.plan);
  undeclared.lane = "candle:not_a_lane";
  assert.throws(() => validatePlan(undeclared, sources), /not declared/);
  const tier = clone(sources.plan);
  tier.requests.default.tiers.push("nvfp4");
  assert.throws(() => validatePlan(tier, sources), /not a catalog tier/);
  const field = clone(sources.plan);
  field.requests.default.request.temperature = 0.7;
  assert.throws(() => validatePlan(field, sources), /unknown request field temperature/);
  const inverted = clone(sources.plan);
  inverted.requests["long-context"].request.semanticSampling = { minTokens: 10, maxTokens: 5 };
  assert.throws(() => validatePlan(inverted, sources), /minTokens exceeds maxTokens/);
});

test("the case file's request fields are exactly the native entrypoint's (deny_unknown_fields)", async () => {
  const source = await readFile(path.join(ROOT, "crates/sceneworks-worker/src/yue2_memory_profile.rs"), "utf8");
  const struct = source.match(/pub\(crate\) struct CaseRequest \{([\s\S]*?)\n\}/)[1];
  const fields = [...struct.matchAll(/^\s+([a-z_]+): /gm)]
    .map(([, name]) => name.replace(/_([a-z])/g, (_, letter) => letter.toUpperCase()))
    .sort();
  assert.deepEqual(fields, [...REQUEST_KEYS].sort());
  // The entrypoint the controller invokes exists under the name it invokes.
  assert.match(source, /\nfn capture_case\(\)/);
  const lib = await readFile(path.join(ROOT, "crates/sceneworks-worker/src/lib.rs"), "utf8");
  assert.match(lib, /\nmod yue2_memory_profile;/);
  for (const item of expandCases(sources.plan)) {
    assert.deepEqual(Object.keys(caseFile(item)), ["id", "tier", "decoder", "request"]);
  }
});

test("samples are attributed to the stage in force; start-up and teardown belong to none", () => {
  const item = expandCases(sources.plan)[0];
  const built = record(item);
  assert.deepEqual(built.measured.stages, {
    load: { peakBytes: 8 * GIB, samples: 1 },
    plan: { peakBytes: 9 * GIB, samples: 1 },
    semantic: { peakBytes: 10 * GIB, samples: 1 },
    acoustic: { peakBytes: 11 * GIB, samples: 1 },
    decode: { peakBytes: 12 * GIB, samples: 1 },
  });
  assert.equal(built.measured.peakBytes, 12 * GIB, "the 99 GiB teardown sample is no stage's");
  assert.throws(() => parseStageMarks('{"stage":"plan","at":1}\n'), /start at the load/);
  assert.throws(() => parseStageMarks('{"stage":"load","at":2}\n{"stage":"plan","at":1}\n'), /back in time/);
  assert.throws(() => parseStageMarks('{"stage":"load","at":1}\n{"stage":"nar","at":2}\n'), /unknown stage/);
  const samples = parseWatchdogSamples(
    '{"event":"started"}\n{"event":"sample","at":3.5,"physicalFootprintBytes":42}\n',
  );
  assert.deepEqual(samples, [{ at: 3.5, bytes: 42 }]);
  assert.deepEqual(stagePeaks(samples, parseStageMarks('{"stage":"load","at":3}\n')), {
    load: { peakBytes: 42, samples: 1 },
  });
});

test("a record is current only under the declared closure, the catalog identity and a clean tree", () => {
  const item = expandCases(sources.plan).find((candidate) => candidate.tier === "q8");
  const current = record(item);
  assert.equal(gradeRecord(current, sources).status, "current");

  const staleClosure = clone(current);
  staleClosure.identity.engine.closureDigest = "0".repeat(64);
  const graded = gradeRecord(staleClosure, sources);
  assert.equal(graded.status, "stale");
  assert.match(graded.reasons.join(), /stale closure/);

  const oldModel = clone(current);
  oldModel.identity.model.revision = "c".repeat(40);
  assert.match(gradeRecord(oldModel, sources).reasons.join(), /model.revision/);
  const oldWeights = clone(current);
  oldWeights.identity.model.weightsSha256 = "d".repeat(64);
  assert.match(gradeRecord(oldWeights, sources).reasons.join(), /model.weightsSha256/);
  const oldDecoder = clone(current);
  oldDecoder.identity.decoder.revision = "e".repeat(40);
  assert.match(gradeRecord(oldDecoder, sources).reasons.join(), /decoder.revision/);
  const dirty = clone(current);
  dirty.identity.sceneworks.dirty = true;
  assert.match(gradeRecord(dirty, sources).reasons.join(), /dirty/);
  const retired = clone(current);
  retired.caseId = "yue2:q8:metal:retired";
  assert.match(gradeRecord(retired, sources).reasons.join(), /not in the plan/);
  const version = clone(current);
  version.identity.engine.digestVersion = "inference-closure-digest v3";
  assert.match(gradeRecord(version, sources).reasons.join(), /digest version/);
});

test("ingest refuses stale and failed records and admits current ones", () => {
  const item = expandCases(sources.plan)[0];
  const current = record(item);
  assert.equal(admitToCorpus(current, sources), current);
  const stale = clone(current);
  stale.identity.engine.closureDigest = "1".repeat(64);
  assert.throws(() => admitToCorpus(stale, sources), /rejected .*stale closure/);
  const failed = clone(current);
  failed.outcome = { status: "failed", exit: 101 };
  assert.throws(() => admitToCorpus(failed, sources), /capture failed/);
  assert.equal(corpusFileName(current), `${item.id.replaceAll(":", "__")}.json`);
  const refused = clone(current);
  refused.outcome = { status: "refused", message: "yue2: … short" };
  assert.equal(admitToCorpus(refused, sources), refused, "a refusal on real hardware is evidence");
});

test("coverage compares the footprint with the stage total, and nvidia-smi with device + reserve", () => {
  const cases = expandCases(sources.plan);
  const metal = record(cases.find((item) => item.backend === "metal"));
  const rows = Object.fromEntries(coverage(metal).map((row) => [row.stage, row]));
  assert.equal(rows.load.covered, true);
  assert.equal(rows.decode.covered, false, "12 GiB measured over an 11 GiB estimate is under-priced");
  const cuda = record(cases.find((item) => item.backend === "cuda"));
  const cudaRows = Object.fromEntries(coverage(cuda).map((row) => [row.stage, row]));
  assert.equal(cudaRows.decode.estimatedBytes, 11 * GIB + CUDA_RESERVE_BYTES);
  assert.equal(cudaRows.decode.covered, true);
  const refused = clone(metal);
  refused.outcome = { status: "refused" };
  assert.deepEqual(coverage(refused), []);
});

test("the corpus status resumes a campaign: current cases are not captured again", () => {
  const cases = expandCases(sources.plan);
  const done = record(cases[0]);
  const stale = clone(record(cases[1]));
  stale.identity.engine.closureDigest = "2".repeat(64);
  const status = Object.fromEntries(
    corpusStatus(sources.plan, [done, stale], sources).map((row) => [row.id, row.status]),
  );
  assert.equal(status[cases[0].id], "current");
  assert.equal(status[cases[1].id], "stale");
  assert.equal(status[cases[2].id], "missing");
});

test("a record must carry its identity", () => {
  const item = expandCases(sources.plan)[0];
  const missing = clone(record(item));
  delete missing.identity.engine.closureDigest;
  assert.throws(() => validateRecord(missing), /closure digest/);
  const unidentified = clone(record(item));
  unidentified.identity.decoder.revision = "";
  assert.throws(() => validateRecord(unidentified), /repo@revision/);
});

test("capture planning: CUDA builds the candle feature, Metal runs under the footprint guard, --out stays outside the repo", async () => {
  assert.ok(cargoTestArgv("cuda").includes("backend-candle"));
  assert.ok(!cargoTestArgv("metal").includes("backend-candle"));
  assert.deepEqual(cargoTestArgv("metal").slice(-6), ["--", "--ignored", "--exact", "--nocapture", "--test-threads", "1"]);
  assert.ok(cargoTestArgv("metal", { noRun: true }).includes("--no-run"));
  const out = path.join(os.tmpdir(), "yue2-profile-test");
  const metal = await planCapture({ caseId: "yue2:q4:metal:default", outDir: out, sources });
  assert.equal(metal.guarded.eventFile, path.join(out, "yue2__q4__metal__default", "watchdog.jsonl"));
  assert.equal(metal.env.SCENEWORKS_ENABLE_YUE2_MEMORY_PROFILE, "1");
  const cuda = await planCapture({ caseId: "yue2:q4:cuda:default", outDir: out, gpuId: 1, sources });
  assert.equal(cuda.guarded, null);
  assert.equal(cuda.env.SCENEWORKS_GPU_ID, "1");
  await assert.rejects(
    planCapture({ caseId: "yue2:q4:metal:default", outDir: path.join(ROOT, ".tmp"), sources }),
    /outside the repository/,
  );
  await assert.rejects(planCapture({ caseId: "yue2:q2:metal:default", outDir: out, sources }), /unknown case/);
  assert.deepEqual(parseArgs(["capture", "--case", "x", "--gpu-id", "0", "--dry-run"]), {
    command: "capture", files: [], caseId: "x", gpuId: 0, dryRun: true,
  });
  assert.throws(() => parseArgs(["run", "--backend", "rocm"]), /unknown backend/);
});

test("a record states the run's truncation, stage times and identities, or none of them (sc-23002)", () => {
  const item = expandCases(sources.plan).find((candidate) => candidate.backend === "metal");
  const run = {
    truncated: { abc: false, semantic: true },
    stageSeconds: { load: 2.5, plan: 7.5, semantic: 10, acoustic: 4, decode: 1.5 },
    engineTiming: { nar_seconds: 3.5, vae_seconds: 1.25 },
    runIdentity: "run-identity",
    planIdentity: "plan-identity",
    decoder: { release: "standard" },
    latent: { sha256: "c".repeat(64) },
  };
  const withRun = (outcome) => record(item, { outcome: { status: "completed", audioSeconds: 30, rms: 0.1, ...outcome } });
  // A record captured before these fields existed stays valid and ingestible.
  validateRecord(record(item));
  admitToCorpus(record(item), sources);
  // A record carrying them is valid and ingestible with them intact.
  const current = withRun(run);
  validateRecord(current);
  assert.deepEqual(admitToCorpus(current, sources).outcome.latent, run.latent);
  // A partial set is a broken capture, never a weaker record.
  assert.throws(() => validateRecord(withRun({ ...run, latent: undefined })), /but not latent/);
  assert.throws(() => validateRecord(withRun({ ...run, truncated: { abc: false, semantic: "yes" } })), /truncated\.semantic/);
  assert.throws(() => validateRecord(withRun({ ...run, stageSeconds: { ...run.stageSeconds, warmup: 1 } })), /unknown stage warmup/);
  assert.throws(() => validateRecord(withRun({ ...run, latent: { sha256: "short" } })), /latent has no sha256/);
  assert.throws(() => admitToCorpus(withRun({ ...run, runIdentity: null }), sources), /runIdentity/);
});
