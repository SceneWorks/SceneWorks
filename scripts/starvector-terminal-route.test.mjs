import assert from "node:assert/strict";
import test from "node:test";
import { assembleRun, assembleSuites, validateBundle, vectorRequest } from "./starvector-terminal-route.mjs";

test("route runner can only construct the typed project-owned vector_generate request", () => {
  const request = vectorRequest({ projectId: "project", sourceAssetId: "asset", model: "starvector_1b", prompt: "icon" });
  assert.deepEqual(request, { projectId: "project", projectName: undefined, mode: "image_to_svg", model: "starvector_1b", sourceAssetId: "asset", prompt: "icon", sampling: undefined, detailBudget: undefined });
  assert.throws(() => vectorRequest({ projectId: "project", model: "starvector_1b" }), /sourceAssetId/);
});

test("route refuses a count-only or incomplete terminal bundle before product calls", () => {
  process.env.STARVECTOR_TERMINAL_PERMANENT_PIN = "65778fb790fa631597fd2739921a669b275d4429";
  const records = (count, prefix) => Array.from({ length: count }, (_, index) => ({ case_id: `${prefix}-${index}`, projectId: "p", sourceAssetId: `a-${index}`, model: "starvector_8b" }));
  const bundle = { schema_version: 1, inference_revision: process.env.STARVECTOR_TERMINAL_PERMANENT_PIN, corpus_sha256: "a".repeat(64), tuples: { "candle-cuda:8b": { image_quality: records(120, "quality"), deterministic_parity: records(20, "parity"), lifecycle: records(4, "lifecycle"), limits: ["complete_root", "eos", "token_limit", "byte_limit", "wall_time_limit", "cancelled"].map((finish_reason, index) => ({ ...records(1, "limit")[0], case_id: `limit-${index}`, finish_reason })) } }, hostile_sanitizer: records(200, "hostile"), prompt_composition: records(60, "prompt") };
  assert.equal(validateBundle(bundle, "candle-cuda:8b"), bundle.tuples["candle-cuda:8b"]);
  bundle.hostile_sanitizer.pop(); assert.throws(() => validateBundle(bundle, "candle-cuda:8b"), /200 hostile/);
});

test("route refuses missing raw metric facts and missing terminal suite output", () => {
  assert.throws(() => assembleRun("mlx:1b", { image_quality: [], run_identity: {}, hardware: {} }, { image_quality: [], deterministic_parity: [], lifecycle: [], limits: [] }, []), /120 unique raw quality facts/);
  assert.throws(() => assembleSuites({}, {}, {}), /lacks source-owned hostile\/prompt suite evidence/);
});
