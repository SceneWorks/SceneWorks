import assert from "node:assert/strict";
import { mkdtemp, mkdir, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { acquireLease, inventory, preflight, validateSuites, validateTupleEvidence } from "./starvector-terminal-producer.mjs";

const digest = "a".repeat(64);
const quality = (index) => ({ case_index: index, case_id: `quality-${index}`, source_svg_sha256: digest, source_png_sha256: digest, provider_transcript_sha256: digest, finish_reason: "complete_root", accepted: true, canonical_svg_sha256: digest, preview_png_sha256: digest, ssim: .9, lpips: .1, latency_seconds: 1 });
const parity = (index) => ({ case_index: index, case_id: `parity-${index}`, seed: index, first_preview_sha256: digest, second_preview_sha256: digest, provider_transcript_sha256: digest, rendered_ssim: .999 });
const tupleEvidence = () => ({ schema_version: 1, campaign_run_id: "run", tuple: "mlx:1b", production_route: "vector_generate", execution: { sceneworks_revision: "b".repeat(40), workflow_run_id: "1", workflow_run_attempt: 1, head_ref: "feature/test" }, model: { backend: "mlx", tier: "1b", provider_id: "starvector-1b-mlx", inventory_sha256: digest }, image_quality: Array.from({ length: 120 }, (_, i) => quality(i)), deterministic_parity: Array.from({ length: 20 }, (_, i) => parity(i)), hardware: { runner_name: "fixture", raw_probe_sha256: digest } });
const hostile = (index) => ({ case_index: index, case_id: `hostile-${index}`, input_sha256: digest, actual_disposition: "rejected", result_contains_inline_svg: false, staging_residue: false, published_paths: [], output_svg_sha256: null, output_png_sha256: null, error_code: "rejected", transcript_sha256: digest });
const prompt = (index) => ({ case_index: index, case_id: `prompt-${index}`, prompt_sha256: digest, raster_asset_sha256: digest, transcript_sha256: digest, seed: index, accepted: true, svg_sha256: digest, preview_sha256: digest, clip_cosine: .98, alignment_loss: .02, latency_seconds: 1 });
const suites = () => ({ schema_version: 1, campaign_run_id: "run", hostile_sanitizer: { corpus_sha256: digest, cases: Array.from({ length: 200 }, (_, i) => hostile(i)) }, prompt_composition: { corpus_sha256: digest, cases: Array.from({ length: 60 }, (_, i) => prompt(i)) } });

test("inventory seals bytes and lease is exclusive", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "svt-")); await writeFile(path.join(root, "a"), "x");
  assert.equal((await inventory(root)).entries.length, 1);
  const release = await acquireLease(path.join(root, "lease"), "pin");
  await assert.rejects(() => acquireLease(path.join(root, "lease"), "pin"), /already held/); await release();
});

test("preflight refuses absent pinned corpus before a metrics environment is used", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "svt-")), inf = path.join(root, "inference"), weights = path.join(root, "weights"), metrics = path.join(root, "metrics");
  await mkdir(path.join(inf, "release"), { recursive: true }); await mkdir(weights); await mkdir(metrics);
  await writeFile(path.join(inf, "release/starvector-terminal-receipt-v1.schema.json"), "{}");
  await assert.rejects(() => preflight({ planPath: "release/starvector-terminal-campaign-v1.json", inferenceRoot: inf, weightsRoot: weights, metricsRoot: metrics, command: ["true"] }), /corpus/);
});

test("tuple evidence rejects an accepted token limit and an omitted raw case", () => {
  const raw = tupleEvidence(); validateTupleEvidence(raw, "mlx:1b", "run");
  raw.image_quality[7].finish_reason = "token_limit";
  assert.throws(() => validateTupleEvidence(raw, "mlx:1b", "run"), /nonterminal/);
  raw.image_quality[7].finish_reason = "complete_root"; raw.image_quality.pop();
  assert.throws(() => validateTupleEvidence(raw, "mlx:1b", "run"), /count/);
});

test("raw hostile and prompt suites reject publication leaks and count spoofing", () => {
  const raw = suites(); validateSuites(raw, "run");
  raw.hostile_sanitizer.cases[3].published_paths.push("projects/x.svg");
  assert.throws(() => validateSuites(raw, "run"), /published/);
  raw.hostile_sanitizer.cases[3].published_paths = []; raw.prompt_composition.cases.pop();
  assert.throws(() => validateSuites(raw, "run"), /count/);
});
