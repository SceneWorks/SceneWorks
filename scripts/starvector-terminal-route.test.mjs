import assert from "node:assert/strict";
import test from "node:test";
import { assembleRun, assembleSuites, validateBundle, vectorRequest } from "./starvector-terminal-route.mjs";

test("route runner can only construct the typed project-owned vector_generate request", () => {
  const request = vectorRequest({ projectId: "project", sourceAssetId: "asset", model: "starvector_1b", prompt: "icon" });
  assert.deepEqual(request, { projectId: "project", projectName: undefined, mode: "image_to_svg", model: "starvector_1b", sourceAssetId: "asset", prompt: "icon", sampling: undefined, detailBudget: undefined });
  assert.throws(() => vectorRequest({ projectId: "project", model: "starvector_1b" }), /sourceAssetId/);
});

test("route refuses a count-only or incomplete terminal bundle before product calls", () => {
  process.env.STARVECTOR_TERMINAL_PERMANENT_PIN = "c6eb6d8e9545193eac844f6fea2db79e4d14bf2a";
  const records = (count, prefix) => Array.from({ length: count }, (_, index) => ({ case_id: `${prefix}-${index}`, projectId: "p", sourceAssetId: `a-${index}`, model: "starvector_8b" }));
  const bundle = { schema_version: 1, inference_revision: process.env.STARVECTOR_TERMINAL_PERMANENT_PIN, corpus_sha256: "a".repeat(64), tuples: { "candle-cuda:8b": { image_quality: records(120, "quality"), deterministic_parity: records(20, "parity"), lifecycle: records(4, "lifecycle"), limits: ["complete_root", "eos", "token_limit", "byte_limit", "wall_time_limit", "cancelled"].map((finish_reason, index) => ({ ...records(1, "limit")[0], case_id: `limit-${index}`, finish_reason })) } }, hostile_sanitizer: records(200, "hostile"), prompt_composition: records(60, "prompt") };
  assert.equal(validateBundle(bundle, "candle-cuda:8b"), bundle.tuples["candle-cuda:8b"]);
  bundle.hostile_sanitizer.pop(); assert.throws(() => validateBundle(bundle, "candle-cuda:8b"), /200 hostile/);
});

test("route refuses missing raw metric facts and missing terminal suite output", () => {
  assert.throws(() => assembleRun("mlx:1b", { image_quality: [], run_identity: {}, hardware: {} }, { image_quality: [], deterministic_parity: [], lifecycle: [], limits: [] }, []), /120 unique raw quality facts/);
  assert.throws(() => assembleSuites({}, {}, {}), /lacks source-owned hostile\/prompt suite evidence/);
});

test("terminal suite identities come from the same-run controller and observed workflow", () => {
  process.env.STARVECTOR_TERMINAL_RUN_ID = "campaign";
  process.env.STARVECTOR_TERMINAL_PERMANENT_PIN = "6".repeat(40);
  process.env.STARVECTOR_TERMINAL_ROUTE_CLOSURE_SHA256 = "a".repeat(64);
  const prompt = Array.from({ length: 60 }, (_, case_index) => ({ case_index, case_id: `prompt-v1-${case_index}`, prompt_sha256: "b".repeat(64), raster_model: "raster", expected_raster_revision: "revision" }));
  const bundle = { hostile_sanitizer: Array.from({ length: 200 }, () => ({ input_sha256: "c".repeat(64) })), prompt_composition: prompt };
  const events = { hostile_sanitizer: Array(200).fill({}), prompt_composition: prompt.map(() => ({ job: { workflow: { disclosure: "raster_to_vector", rasterStage: { model: "raster", revision: "revision" } }, terminalRasterObservation: { providerId: "native-raster", model: "raster", revision: "revision" } } })) };
  const observation = { packages: { "scikit-image": "0.25.2", lpips: "0.1.4" }, metrics_lock_sha256: "d".repeat(64), lpips_linear_sha256: "e".repeat(64), alexnet_sha256: "f".repeat(64), metric_transcript_path: "/tmp/metric-transcript", metric_transcript_sha256: "1".repeat(64), clip: { provider_id: "open-clip-torch", model: "ViT-B-32", revision: "clip-revision", inventory_sha256: "2".repeat(64) } };
  const metrics = { terminal_suite_measurements: { hostile_cases: [], prompt_cases: [], metric_observation: observation } };
  const context = { campaign_run_id: "campaign", permanent_pin: "6".repeat(40), workflow_run_id: "123", workflow_run_attempt: 1, controller_started_at: "2026-08-29T00:00:00Z", service: { started_at: "2026-08-29T00:00:00Z", sceneworks_revision: "7".repeat(40) }, route: { path: "scripts/starvector-terminal-route.mjs", sha256: "a".repeat(64), sceneworks_revision: "7".repeat(40) }, metrics: { metrics_lock_sha256: "d".repeat(64), packages: [{ name: "scikit-image", version: "0.25.2" }, { name: "lpips", version: "0.1.4" }], weights: { lpips_linear: { sha256: "e".repeat(64) }, alexnet: { sha256: "f".repeat(64) } }, clip: { provider_id: "open-clip-torch", model: "ViT-B-32", revision: "clip-revision", checkpoint: { sha256: "2".repeat(64) } } }, prompt_raster: { provider_id: "native-raster", model: "raster", revision: "revision", inventory_sha256: "3".repeat(64) }, inference_preflight: { workflow_run_id: "9" } };
  const result = assembleSuites(bundle, events, metrics, context, "4".repeat(64), "2026-08-29T01:00:00Z", "sha256:fixture");
  assert.equal(result.execution.head_sha, "7".repeat(40));
  assert.equal(result.prompt_composition.raster_provider_id, "native-raster");
  assert.equal(result.metric_identity.ssim.package_version, "0.25.2");
  const driftedWorkflow = structuredClone(events); driftedWorkflow.prompt_composition[0].job.workflow.rasterStage.revision = "drifted";
  assert.throws(() => assembleSuites(bundle, driftedWorkflow, metrics, context, "4".repeat(64), "2026-08-29T01:00:00Z", "sha256:fixture"), /actual prompt workflow/);
  const driftedProvider = structuredClone(events); driftedProvider.prompt_composition[0].job.terminalRasterObservation.providerId = "claimed-only";
  assert.throws(() => assembleSuites(bundle, driftedProvider, metrics, context, "4".repeat(64), "2026-08-29T01:00:00Z", "sha256:fixture"), /raster provider\/model identity/);
  const driftedMetric = structuredClone(metrics); driftedMetric.terminal_suite_measurements.metric_observation.clip.inventory_sha256 = "5".repeat(64);
  assert.throws(() => assembleSuites(bundle, events, driftedMetric, context, "4".repeat(64), "2026-08-29T01:00:00Z", "sha256:fixture"), /OpenCLIP identity drifted/);
  const driftedRun = structuredClone(context); driftedRun.workflow_run_id = null;
  assert.throws(() => assembleSuites(bundle, events, metrics, driftedRun, "4".repeat(64), "2026-08-29T01:00:00Z", "sha256:fixture"), /workflow execution identity/);
});
