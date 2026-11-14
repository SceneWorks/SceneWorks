// @vitest-environment node
import { readFileSync } from "node:fs";
import { beforeAll, expect, it } from "vitest";

import init, { validate_request, validate_shape } from "./wasm/options-validator/options_validator.js";

const corpus = JSON.parse(readFileSync(new URL("../../../crates/sceneworks-options/tests/fixtures/validation_corpus.json", import.meta.url), "utf8"));

beforeAll(async () => {
  const bytes = readFileSync(new URL("./wasm/options-validator/options_validator_bg.wasm", import.meta.url));
  await init({ module_or_path: bytes });
});

it("runs the same native fixture corpus inside the actual WASM export", () => {
  expect(corpus.schemaVersion).toBe(1);
  for (const kind of [
    "image", "video", "audio", "vector", "vectorPrompt", "videoUpscale",
    "vqa", "interleave", "characterTest", "promptRefine", "yue2",
    "yue2Execution", "training", "lora_train", "control_training",
  ]) {
    expect(corpus.cases.some((sample) => sample.kind === kind && sample.expected.ok), kind).toBe(true);
    expect(corpus.cases.some((sample) => sample.kind === kind && !sample.expected.ok), kind).toBe(true);
  }
  for (const sample of corpus.cases) {
    const entry = corpus.modelFacts[sample.model] ?? {};
    const result = JSON.parse(validate_request(sample.kind, JSON.stringify(sample.payload), JSON.stringify(entry)));
    expect(result.validatorSchemaVersion, sample.name).toBe(1);
    expect(result.result, sample.name).toEqual(sample.expected);
  }
});

it("returns the same typed version refusal with and without model facts", () => {
  const payload = JSON.stringify({ optionsSchemaVersion: 2, model: "flux_dev", prompt: "x" });
  for (const result of [
    JSON.parse(validate_request("image", payload, "{}")),
    JSON.parse(validate_shape("image", payload)),
  ]) {
    expect(result.result.ok).toBe(false);
    expect(result.result.refusal).toMatchObject({
      code: "UnsupportedSchemaVersion", field: "optionsSchemaVersion",
    });
  }
});

it("refuses conflicting legacy aliases before static validation", () => {
  const payload = JSON.stringify({
    model: "flux_dev", prompt: "x", ipAdapterScale: 0.4, ip_adapter_scale: 0.8,
  });
  for (const result of [
    JSON.parse(validate_request("image", payload, "{}")),
    JSON.parse(validate_shape("image", payload)),
  ]) {
    expect(result.result.refusal).toMatchObject({
      code: "ConflictingAliases", field: "ipAdapterScale",
    });
  }
});
