// sc-23002: the YuE2 terminal acceptance driver's pure parts — case planning, dependency skips,
// stage timing, WAV evidence, memory parsing, record validation and summary rendering. The driver's
// service half (API + worker + real renders) runs only on the terminal hosts.
import assert from "node:assert/strict";
import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import {
  CASES,
  COVERED_ELSEWHERE,
  RECORD_SCHEMA,
  SONG_RUN_FILES,
  buildSummary,
  caseRunnerIds,
  dependencySkip,
  deviceMatches,
  footprintBytes,
  insideRepository,
  nvidiaUsedBytes,
  parseArgs,
  parseWav,
  peakOf,
  planCases,
  renderAssertions,
  renderMarkdown,
  serviceEnv,
  stageOf,
  stageTimeline,
  validateRecord,
  verifyRunArtifacts,
} from "./yue2-acceptance.mjs";

const sha = (bytes) => createHash("sha256").update(bytes).digest("hex");

test("every case has a runner and a unique id; dependencies name earlier cases", () => {
  const ids = CASES.map((item) => item.id);
  assert.equal(new Set(ids).size, ids.length);
  assert.deepEqual([...caseRunnerIds()].sort(), [...ids].sort());
  for (const [index, item] of CASES.entries()) {
    for (const need of [...item.needs, ...(item.dryNeeds ?? [])]) {
      assert.ok(ids.indexOf(need) >= 0 && ids.indexOf(need) < index, `${item.id} needs ${need}, which must run first`);
    }
  }
});

test("the plan runs every case on a real run and skips only with a stated reason", () => {
  const cuda = planCases({ platform: "cuda" });
  assert.ok(cuda.every((entry) => entry.action === "run"), JSON.stringify(cuda.filter((entry) => entry.action !== "run")));
  // A dry run starts no worker: only the worker-free cases run, every other one says why it did not.
  const dry = planCases({ platform: "metal", dryRun: true });
  assert.deepEqual(
    dry.filter((entry) => entry.action === "run").map((entry) => entry.id),
    ["catalog-preflight", "isolation-v1-v2", "transcription-blocked", "licence-gate"],
  );
  for (const entry of dry.filter((item) => item.action === "skip")) assert.match(entry.reason, /dry run/);
  // The operator's --skip is recorded as such.
  const skipped = planCases({ platform: "cuda", skip: ["batch-serial"] }).find((entry) => entry.id === "batch-serial");
  assert.deepEqual(skipped, { id: "batch-serial", action: "skip", reason: "skipped by the operator (--skip)" });
  assert.throws(() => planCases({ platform: "cuda", skip: ["no-such-case"] }), /unknown case/);
  assert.throws(() => planCases({ platform: "rocm" }), /metal or cuda/);
});

test("a Metal worker kill needs the owner's explicit opt-in; CUDA always runs it", () => {
  const metal = planCases({ platform: "metal" }).find((entry) => entry.id === "worker-kill-resume");
  assert.equal(metal.action, "skip");
  assert.match(metal.reason, /--allow-metal-worker-kill/);
  assert.equal(planCases({ platform: "metal", allowMetalWorkerKill: true }).find((entry) => entry.id === "worker-kill-resume").action, "run");
  assert.equal(planCases({ platform: "cuda" }).find((entry) => entry.id === "worker-kill-resume").action, "run");
});

test("a case whose dependency did not pass is skipped naming the dependency and its reason", () => {
  const decode = CASES.find((item) => item.id === "decode-standard-legacy");
  const records = new Map([
    ["install-cold", { status: "passed" }],
    ["create-cot-full", { status: "failed", reason: "job completed: status failed" }],
  ]);
  assert.equal(dependencySkip(decode, records), "dependency create-cot-full failed: job completed: status failed");
  assert.equal(dependencySkip(decode, new Map([["install-cold", { status: "passed" }]])), "dependency create-cot-full has not run");
  records.set("create-cot-full", { status: "passed" });
  assert.equal(dependencySkip(decode, records), null);
  // The licence gate needs installed weights to execute its job — except in a dry run, which only
  // exercises the refusal and the submission.
  const licence = CASES.find((item) => item.id === "licence-gate");
  const notInstalled = new Map([["install-cold", { status: "skipped", reason: "dry run" }]]);
  assert.match(dependencySkip(licence, notInstalled), /install-cold skipped/);
  assert.equal(dependencySkip(licence, notInstalled, { dryRun: true }), null);
});

test("arguments: platform and out are required; --skip repeats; unknown input is refused", () => {
  const options = parseArgs(["--platform", "cuda", "--out", "E:/evidence", "--skip", "batch-serial", "--skip", "admission", "--hf-hub", "E:/huggingface/hub", "--dry-run"]);
  assert.equal(options.platform, "cuda");
  assert.deepEqual(options.skip, ["batch-serial", "admission"]);
  assert.equal(options.hfHub, "E:/huggingface/hub");
  assert.equal(options.dryRun, true);
  assert.throws(() => parseArgs(["--out", "x"]), /--platform/);
  assert.throws(() => parseArgs(["--platform", "metal"]), /--out/);
  assert.throws(() => parseArgs(["--platform", "metal", "--out", "x", "--skip", "nope"]), /unknown case/);
  assert.throws(() => parseArgs(["--platform", "metal", "--out", "x", "--bogus"]), /unknown argument/);
  assert.throws(() => parseArgs(["--platform", "metal", "--out", "x", "--port", "70000"]), /TCP port/);
});

test("the stage comes from the worker's progress message; timings run from one stage to the next", () => {
  const at = (stage, message, status = "running") => ({ status, message, stage });
  assert.equal(stageOf(at(null, "Checking YuE2 eligibility.", "preparing")), "eligibility");
  assert.equal(stageOf(at(null, "Loading YuE2.")), "load");
  assert.equal(stageOf(at(null, "Planning the score: token 4 of 4096.")), "plan");
  assert.equal(stageOf(at(null, "Generating semantic tokens: 9 of 1125.")), "semantic");
  assert.equal(stageOf(at(null, "Acoustic synthesis: step 2 of 32.")), "acoustic");
  assert.equal(stageOf(at(null, "Decoding audio.")), "decode");
  assert.equal(stageOf(at(null, "Saving the YuE2 run.", "saving")), "saving");
  assert.equal(stageOf({ status: "queued" }), "queued");
  assert.equal(stageOf({ status: "canceled", message: "YuE2 job canceled by user." }), "canceled");
  assert.equal(stageOf(at(null, "Cancellation requested. Waiting for worker acknowledgement.")), "cancel-requested");
  const timeline = stageTimeline([
    { at: 0, stage: "queued" },
    { at: 1000, stage: "load" },
    { at: 5000, stage: "semantic" },
    { at: 6000, stage: "semantic" },
    { at: 15000, stage: "acoustic" },
    { at: 18000, stage: "decode" },
    { at: 19500, stage: "completed" },
  ]);
  assert.deepEqual(timeline.order, ["queued", "load", "semantic", "acoustic", "decode", "completed"]);
  assert.deepEqual(timeline.seconds, { queued: 1, load: 4, semantic: 10, acoustic: 3, decode: 1.5, completed: 0 });
});

function wav({ tag, bits, channels, rate, samples, extensible = false }) {
  const bytes = bits / 8;
  const data = Buffer.alloc(samples.length * bytes);
  samples.forEach((value, index) => {
    if (tag === 1) data.writeInt16LE(Math.round(value * 32767), index * bytes);
    else data.writeFloatLE(value, index * bytes);
  });
  const fmt = Buffer.alloc(extensible ? 40 : 16);
  fmt.writeUInt16LE(extensible ? 0xfffe : tag, 0);
  fmt.writeUInt16LE(channels, 2);
  fmt.writeUInt32LE(rate, 4);
  fmt.writeUInt32LE(rate * channels * bytes, 8);
  fmt.writeUInt16LE(channels * bytes, 12);
  fmt.writeUInt16LE(bits, 14);
  if (extensible) {
    fmt.writeUInt16LE(22, 16);
    fmt.writeUInt16LE(tag, 24);
  }
  // An INFO list before the data, as the worker's licence-tagged asset carries.
  const info = Buffer.concat([Buffer.from("LIST"), Buffer.from([4, 0, 0, 0]), Buffer.from("INFO")]);
  const chunk = (id, body) => Buffer.concat([Buffer.from(id), Buffer.from(Uint32Array.of(body.length).buffer), body]);
  const body = Buffer.concat([Buffer.from("WAVE"), chunk("fmt ", fmt), info, chunk("data", data)]);
  return Buffer.concat([Buffer.from("RIFF"), Buffer.from(Uint32Array.of(body.length).buffer), body]);
}

test("WAV evidence: duration and RMS of the run's float WAV and the PCM16 asset", () => {
  const samples = Array.from({ length: 48000 * 2 }, (_, index) => (index % 2 ? 0.5 : -0.5));
  const float = parseWav(wav({ tag: 3, bits: 32, channels: 2, rate: 48000, samples }));
  assert.equal(float.encoding, "float32");
  assert.equal(float.sampleRate, 48000);
  assert.equal(float.channels, 2);
  assert.equal(float.durationSeconds, 1);
  assert.ok(Math.abs(float.rms - 0.5) < 1e-6, float.rms);
  const pcm = parseWav(wav({ tag: 1, bits: 16, channels: 2, rate: 48000, samples, extensible: true }));
  assert.equal(pcm.encoding, "pcm16");
  assert.equal(pcm.durationSeconds, 1);
  assert.ok(Math.abs(pcm.rms - 0.5) < 1e-3, pcm.rms);
  assert.equal(parseWav(wav({ tag: 3, bits: 32, channels: 2, rate: 48000, samples: [0, 0, 0, 0] })).rms, 0);
  assert.throws(() => parseWav(Buffer.from("not a wav file at all")), /RIFF/);
  assert.throws(() => parseWav(wav({ tag: 1, bits: 16, channels: 1, rate: 8000, samples: [0] }).subarray(0, 36)), /fmt or data/);
});

test("memory samples: footprint sums the service pids, nvidia-smi is MiB, the peak keeps its gaps", () => {
  const payload = { processes: [
    { pid: 10, auxiliary: { phys_footprint: 1_000 } },
    { pid: 11, auxiliary: { phys_footprint: 2_000 } },
    { pid: 99, auxiliary: { phys_footprint: 9_999_999 } },
  ] };
  assert.equal(footprintBytes(payload, [10, 11]), 3_000);
  assert.throws(() => footprintBytes({ processes: [{ pid: 10, auxiliary: {} }] }, [10]), /phys_footprint/);
  assert.throws(() => footprintBytes({ processes: [] }, [10]), /none of the service pids/);
  assert.equal(nvidiaUsedBytes("2048\n"), 2048 * 1024 * 1024);
  assert.throws(() => nvidiaUsedBytes("[N/A]"), /memory\.used/);
  assert.deepEqual(peakOf([{ bytes: 5 }, { bytes: 9 }, { bytes: 7 }], ["timeout"], "s"), {
    sampler: "s", samples: 3, faults: 1, firstFault: "timeout", peakBytes: 9, baselineBytes: 5,
  });
  assert.equal(peakOf([], ["timeout"], "s").peakBytes, null);
});

function renderEvidence(overrides = {}) {
  const identity = "run-1";
  return {
    platform: "metal",
    job: { status: "completed", result: { yue2: { run: { identity }, truncated: { abc: false, semantic: true } } } },
    runResult: { identity, truncated: { abc: false, semantic: true } },
    runConfig: { device: "metal", model_dtype: "bfloat16" },
    artifactCheck: { ok: true, detail: "10 artifacts re-hashed" },
    runAudio: { sampleRate: 48000, channels: 2, rms: 0.12 },
    ...overrides,
  };
}

test("render evidence: production device, the engine's own identity and truncation, verified artifacts, audible 48 kHz stereo", () => {
  const ok = renderAssertions(renderEvidence());
  assert.ok(ok.every((assertion) => assertion.ok), JSON.stringify(ok.filter((assertion) => !assertion.ok)));
  const failing = (overrides) => renderAssertions(renderEvidence(overrides)).filter((assertion) => !assertion.ok).map((assertion) => assertion.name);
  // A CPU fallback is not production-device evidence, whatever else passed.
  assert.deepEqual(failing({ runConfig: { device: "cpu" } }), ["production device"]);
  assert.deepEqual(failing({ platform: "cuda" }), ["production device"]);
  assert.deepEqual(failing({ runResult: { identity: "other", truncated: { abc: false, semantic: true } } }), ["run identity"]);
  // The app must report the engine's truncation, not a default.
  assert.deepEqual(failing({ job: { status: "completed", result: { yue2: { run: { identity: "run-1" }, truncated: { abc: false, semantic: false } } } } }), ["truncation reported"]);
  assert.deepEqual(failing({ runResult: { identity: "run-1" } }), ["truncation reported"]);
  assert.deepEqual(failing({ artifactCheck: { ok: false, detail: "latent.npy hashes to x" } }), ["artifacts verify"]);
  assert.deepEqual(failing({ runAudio: { sampleRate: 44100, channels: 2, rms: 0.1 } }), ["48 kHz stereo"]);
  assert.deepEqual(failing({ runAudio: { sampleRate: 48000, channels: 2, rms: 0 } }), ["audible"]);
  assert.equal(deviceMatches("cuda", "cuda"), true);
  assert.equal(deviceMatches("metal", "cuda"), false);
});

test("the artifact check re-hashes every recorded file and names the first that does not verify", async () => {
  const dir = await mkdtemp(path.join(os.tmpdir(), "yue2-acceptance-run-"));
  try {
    const artifacts = {};
    for (const name of [...SONG_RUN_FILES, "stages/plan.json"]) {
      const body = Buffer.from(`bytes of ${name}`);
      await mkdir(path.dirname(path.join(dir, name)), { recursive: true });
      await writeFile(path.join(dir, name), body);
      artifacts[name] = { sha256: sha(body), bytes: body.length };
    }
    const result = { kind: "song", artifacts };
    assert.deepEqual(await verifyRunArtifacts(dir, result), { ok: true, detail: `${SONG_RUN_FILES.length + 1} artifacts re-hashed`, files: SONG_RUN_FILES.length + 1 });
    await writeFile(path.join(dir, "latent.npy"), "bytes of latent.npX");
    assert.match((await verifyRunArtifacts(dir, result)).detail, /latent\.npy hashes to/);
    await rm(path.join(dir, "audio.wav"));
    assert.match((await verifyRunArtifacts(dir, result)).detail, /audio\.wav is missing|latent\.npy/);
    const { "audio.wav": _dropped, ...partial } = artifacts;
    assert.match((await verifyRunArtifacts(dir, { kind: "song", artifacts: partial })).detail, /omits audio\.wav/);
    assert.equal((await verifyRunArtifacts(dir, { kind: "song" })).ok, false);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

function passedRecord(caseId = "create-cot-full", overrides = {}) {
  return {
    schema: RECORD_SCHEMA,
    caseId,
    acceptance: ["AT1"],
    platform: "metal",
    memorySampled: true,
    status: "passed",
    reason: null,
    assertions: [{ name: "job completed", ok: true }],
    requests: [],
    jobs: [{
      jobId: "job_1", kind: "create", status: "completed", rendered: true, tier: "bf16", runIdentity: "run-1",
      output: { runAudioSha256: "a".repeat(64), durationSeconds: 31.5, rms: 0.1 },
      truncated: { abc: false, semantic: false }, device: "metal", dtype: { model: "bfloat16", vae: "float32" },
      stageTimings: { observed: { order: [], seconds: {} } }, durationSeconds: 120,
    }],
    peakMemory: { peakBytes: 12 * 1024 ** 3 },
    ...overrides,
  };
}

test("a record stands as evidence only when it is complete; a non-pass must say why", () => {
  validateRecord(passedRecord());
  const job = passedRecord().jobs[0];
  const broken = (overrides) => () => validateRecord(passedRecord("create-cot-full", overrides));
  assert.throws(broken({ assertions: [{ name: "audible", ok: false }] }), /failing assertions: audible/);
  assert.throws(broken({ assertions: [] }), /asserted nothing/);
  assert.throws(broken({ status: "skipped" }), /must say why/);
  assert.throws(broken({ status: "failed", reason: " " }), /must say why/);
  assert.throws(broken({ status: "blocked", reason: "x" }), /only an expected refusal/);
  validateRecord(passedRecord("transcription-blocked", { status: "blocked", reason: "component_blocked", jobs: [] }));
  assert.throws(broken({ jobs: [{ ...job, output: { ...job.output, runAudioSha256: undefined } }] }), /no output sha256/);
  assert.throws(broken({ jobs: [{ ...job, truncated: { abc: false } }] }), /no truncation flags/);
  assert.throws(broken({ jobs: [{ ...job, device: null }] }), /device\/dtype/);
  assert.throws(broken({ jobs: [{ ...job, stageTimings: undefined }] }), /stage timings/);
  assert.throws(broken({ peakMemory: { peakBytes: null, firstFault: "footprint exited 1" } }), /no peak memory sample \(footprint exited 1\)/);
  validateRecord(passedRecord("catalog-preflight", { memorySampled: false, peakMemory: null, jobs: [] }));
  assert.throws(broken({ caseId: "long-context" }), /unknown case/);
});

function summaryOf(statuses, meta = {}) {
  const records = CASES.map((item) => ({
    ...passedRecord(item.id, { jobs: [] }),
    status: statuses[item.id] ?? (item.expectBlocked ? "blocked" : "passed"),
    reason: statuses[item.id] && statuses[item.id] !== "passed" ? `why ${item.id}` : item.expectBlocked ? "blocked as designed" : null,
  }));
  return buildSummary(records, { platform: "cuda", ...meta });
}

test("the verdict passes only a complete run: any failure fails it, any skip or dry run leaves it incomplete", () => {
  assert.equal(summaryOf({}).verdict, "pass");
  assert.equal(summaryOf({}).counts.blocked, 1);
  assert.equal(summaryOf({ "batch-serial": "failed" }).verdict, "fail");
  assert.equal(summaryOf({ "batch-serial": "skipped" }).verdict, "incomplete");
  assert.equal(summaryOf({ "batch-serial": "skipped", admission: "failed" }).verdict, "fail");
  assert.equal(summaryOf({}, { dryRun: true }).verdict, "incomplete");
  // A run that stopped early fails, even when nothing it recorded failed.
  const stopped = buildSummary([passedRecord("catalog-preflight", { jobs: [] })], { platform: "cuda", fatal: "the worker did not register idle in 180 s" });
  assert.equal(stopped.verdict, "fail");
  assert.equal(stopped.fatal, "the worker did not register idle in 180 s");
  const partial = buildSummary([passedRecord("catalog-preflight", { jobs: [] })], { platform: "cuda" });
  assert.equal(partial.verdict, "incomplete");
  assert.equal(partial.missing.length, CASES.length - 1);
  assert.deepEqual(summaryOf({}).coveredElsewhere, COVERED_ELSEWHERE);
});

test("the markdown table has one row per case, carries hashes not audio, and escapes table syntax", () => {
  const records = [passedRecord(), { ...passedRecord("batch-serial", { jobs: [] }), status: "failed", reason: "take 2 | overlapped\ntake 1" }];
  const markdown = renderMarkdown(buildSummary(records, { platform: "metal", identity: { sceneworksRevision: "b".repeat(40), inferencePin: "c".repeat(40), apiBinarySha256: "d".repeat(64) } }));
  assert.match(markdown, /Verdict: \*\*fail\*\*/);
  assert.match(markdown, /\| create-cot-full \| AT1 \| passed \| create·bf16·metal\/bfloat16 completed \| aaaaaaaaaaaaaaaa \| 31\.5 \| 0\.1000 \| false\/false \| 12\.00 GiB \|/);
  assert.match(markdown, /take 2 \\\| overlapped take 1/);
  assert.match(markdown, /long-context/);
  assert.match(markdown, /CC BY-NC 4\.0/);
  assert.doesNotMatch(markdown, /\.wav/);
});

test("the service environment isolates the app, pins the worker device and really takes the hub away offline", () => {
  const base = { PATH: "/bin", HF_HOME: "/home/hf", SCENEWORKS_ACCESS_TOKEN: "t", SCENEWORKS_MLX_MEMORY_CAP_GB: "8", HF_HUB_OFFLINE: "1" };
  const online = serviceEnv({ platform: "cuda", base, url: "http://127.0.0.1:5000", port: 5000, dataDir: "D", configDir: "C", hfHub: "E:/hub", workerId: "w" });
  assert.equal(online.PATH, "/bin");
  for (const gone of ["HF_HOME", "SCENEWORKS_ACCESS_TOKEN", "SCENEWORKS_MLX_MEMORY_CAP_GB", "HF_HUB_OFFLINE", "SCENEWORKS_HUGGINGFACE_BASE_URL"]) assert.equal(online[gone], undefined, gone);
  assert.equal(online.SCENEWORKS_GPU_ID, "0");
  assert.equal(online.HF_HUB_CACHE, "E:/hub");
  assert.equal(online.SCENEWORKS_API_URL, "http://127.0.0.1:5000");
  assert.equal(online.SCENEWORKS_JOBS_DB_PATH, path.join("D", "cache", "jobs.db"));
  const offline = serviceEnv({ platform: "metal", base, url: "u", port: 1, dataDir: "D", configDir: "C", hfHub: "H", workerId: "w", offline: true, extra: { SCENEWORKS_WORKER_ONLY: "1" } });
  assert.equal(offline.SCENEWORKS_GPU_ID, "mlx");
  assert.equal(offline.HF_HUB_OFFLINE, "1");
  assert.equal(offline.SCENEWORKS_HUGGINGFACE_BASE_URL, "http://127.0.0.1:9");
  assert.equal(offline.SCENEWORKS_WORKER_ONLY, "1");
});

test("evidence and app state must live outside the repository", () => {
  const root = path.resolve("/repo/SceneWorks");
  assert.equal(insideRepository(root, root), true);
  assert.equal(insideRepository(path.join(root, "target", "evidence"), root), true);
  assert.equal(insideRepository(path.resolve("/tmp/yue2"), root), false);
  assert.equal(insideRepository(path.resolve("/repo/SceneWorks-evidence"), root), false);
});
