// SW-7182 (feature SW-F371): pure-function and static-contract tests for PAG acceptance driver.
import test from "node:test";
import assert from "node:assert/strict";
import { deflateSync } from "node:zlib";
import { decodePng } from "./tier2-ab.mjs";
import {
  ACCEPTANCE_THRESHOLDS,
  FAMILIES,
  PAG_ROUTES,
  buildSyntheticZImageLoraSafetensors,
  dispatchPagAcceptanceOptions,
  evaluateT2iAbVerdict,
  imageQualityStats,
  pagDefaultHubs,
  pagExitCode,
  readWorkflowShareFromPng,
  resolvePagConfig,
  summarizePagReport,
  syntheticReferencePng,
} from "./pag-acceptance.mjs";

const SRC_SHA = "c".repeat(40);

test("PAG_ROUTES covers all six target families across T2I, Edit, 2-pass CFG-free, and LoRA", () => {
  const byFamily = new Map(FAMILIES.map((f) => [f, []]));
  for (const r of PAG_ROUTES) {
    assert.ok(byFamily.has(r.family), `unexpected family ${r.family}`);
    byFamily.get(r.family).push(r);
  }
  for (const [family, list] of byFamily.entries()) {
    assert.ok(list.some((r) => r.mode === "text_to_image"), `${family} missing T2I route`);
    assert.ok(list.some((r) => r.mode === "edit_image"), `${family} missing edit_image route`);
  }
  assert.ok(PAG_ROUTES.some((r) => r.syntheticLora), "missing AT7 LoRA+PAG route");
});

test("dispatchPagAcceptanceOptions validates inputs and defaults preflight_only=true", () => {
  const inputs = {
    profile: "tier2-ab",
    ref: SRC_SHA,
    backend: "candle",
    inference_sha: "",
    moss_options_json: JSON.stringify({ suite: "pag-acceptance" }),
  };
  const resolved = dispatchPagAcceptanceOptions(inputs, SRC_SHA);
  assert.deepEqual(resolved, {
    suite: "pag-acceptance",
    source_sha: SRC_SHA,
    backend: "cuda",
    routes: "",
    hubs: "",
    preflight_only: true,
    inference_sha: "",
  });
  assert.ok(pagDefaultHubs("mlx", "/Users/runner").includes("/Users/runner/sceneworks-rw-weights/hub"));
  const cfg = resolvePagConfig(
    {
      PAG_ACCEPTANCE_BACKEND: "mlx",
      PAG_ACCEPTANCE_SOURCE_SHA: SRC_SHA,
      PAG_ACCEPTANCE_SOURCE_DIR: "/w/src",
      PAG_ACCEPTANCE_WORK: "/t/work",
    },
    { platform: "darwin", home: "/Users/runner" },
  );
  assert.equal(cfg.routes.length, PAG_ROUTES.length);
  assert.ok(cfg.hubs.includes("/Users/runner/sceneworks-rw-weights/hub"));
});

test("syntheticReferencePng, readWorkflowShareFromPng, and synthetic LoRA safetensors are well-formed", () => {
  for (const variant of ["primary", "secondary"]) {
    const png = syntheticReferencePng(variant, 128);
    const decoded = decodePng(png);
    assert.equal(decoded.width, 128);
    assert.equal(decoded.height, 128);
    const stats = imageQualityStats(decoded.rgba);
    assert.ok(stats.channelStdDev >= ACCEPTANCE_THRESHOLDS.minChannelStdDev);
    assert.ok(stats.dynamicRange >= ACCEPTANCE_THRESHOLDS.minDynamicRange);
  }

  // Build PNG with compressed sceneworks:workflow iTXt chunk and parse back.
  const basePng = syntheticReferencePng("primary", 32);
  const payload = deflateSync(Buffer.from(JSON.stringify({ advanced: { pag: { scale: 1.5 }, pagExecutionReport: { scale: 1.5, layers: [14], activeSteps: 3, totalSteps: 4 } } }), "utf8"));
  const itxtData = Buffer.concat([Buffer.from("sceneworks:workflow\0", "latin1"), Buffer.from([1, 0, 0, 0]), payload]);
  const crc = (buf) => { let c = 0xffffffff; for (const b of buf) { c ^= b; for (let i = 0; i < 8; i++) c = (c >>> 1) ^ ((c & 1) ? 0xedb88320 : 0); } return (c ^ 0xffffffff) >>> 0; };
  const chunk = Buffer.alloc(itxtData.length + 12);
  chunk.writeUInt32BE(itxtData.length, 0);
  chunk.write("iTXt", 4, "ascii");
  itxtData.copy(chunk, 8);
  chunk.writeUInt32BE(crc(Buffer.concat([Buffer.from("iTXt", "ascii"), itxtData])), itxtData.length + 8);
  const withChunk = Buffer.concat([basePng.subarray(0, basePng.length - 12), chunk, basePng.subarray(basePng.length - 12)]);
  const parsed = readWorkflowShareFromPng(withChunk);
  assert.deepEqual(parsed.advanced.pagExecutionReport, { scale: 1.5, layers: [14], activeSteps: 3, totalSteps: 4 });

  const lora = buildSyntheticZImageLoraSafetensors(64, 4);
  const headerLen = Number(lora.readBigUInt64LE(0));
  const header = JSON.parse(lora.subarray(8, 8 + headerLen).toString("utf8"));
  assert.equal(header.__metadata__.networkType, "lora");
  assert.deepEqual(header["layers.15.attention.to_q.lora_A.weight"].shape, [4, 64]);
});

test("evaluateT2iAbVerdict and pagExitCode enforce AT1, AT2, AT6 and 6-family completeness", () => {
  const baseRgba = decodePng(syntheticReferencePng("primary", 64)).rgba;
  const pagRgba = decodePng(syntheticReferencePng("secondary", 64)).rgba;
  const statsA = imageQualityStats(baseRgba);
  const statsB = imageQualityStats(pagRgba);
  const rep = { scale: 1.5, layers: [14], activeSteps: 3, totalSteps: 4 };

  const ok = evaluateT2iAbVerdict({
    off1: { status: "completed", stats: statsA, pagReport: null, sharePagReport: null, buffer: baseRgba },
    zero: { status: "completed", stats: statsA, pagReport: null, sharePagReport: null, buffer: Buffer.from(baseRgba) },
    pagOn: { status: "completed", stats: statsB, pagReport: rep, sharePagReport: rep, buffer: pagRgba },
    expectedReport: { layers: [14], activeSteps: 3, totalSteps: 4 },
    pagScale: 1.5,
  });
  assert.equal(ok.verdict, "passed");

  const identicalPag = evaluateT2iAbVerdict({
    off1: { status: "completed", stats: statsA, pagReport: null, sharePagReport: null, buffer: baseRgba },
    zero: { status: "completed", stats: statsA, pagReport: null, sharePagReport: null, buffer: baseRgba },
    pagOn: { status: "completed", stats: statsA, pagReport: rep, sharePagReport: rep, buffer: baseRgba },
    expectedReport: { layers: [14], activeSteps: 3, totalSteps: 4 },
    pagScale: 1.5,
  });
  assert.equal(identicalPag.verdict, "error");

  const allSix = FAMILIES.map((f, i) => ({ id: `r${i}`, family: f, status: "ran", verdict: "passed" }));
  assert.equal(summarizePagReport(allSix).missingFamilies.length, 0);
  assert.equal(pagExitCode({ mode: "run", routes: allSix }), 0);
  assert.equal(pagExitCode({ mode: "run", routes: allSix.slice(0, 5) }), 1);
});
