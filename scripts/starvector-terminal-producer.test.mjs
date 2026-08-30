import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { mkdtemp, mkdir, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { claimCampaignMarker, claimTupleMarker, consolidateCanonicalArtifacts, inventory, sealReceipt, validateInferencePreflight, verifyInferenceCheckout, verifyPermanentPin } from "./starvector-terminal-producer.mjs";

// CI supplies an exact pinned inference checkout; the local default preserves
// the focused fixture without embedding a developer-specific worktree path.
const inferenceRepository = process.env.STARVECTOR_TERMINAL_INFERENCE_TEST_ROOT ?? path.resolve("../sc-22261-inference-integration");
let inferenceRoot = inferenceRepository;
const sha = (value) => createHash("sha256").update(value).digest("hex");
const digest = sha("artifact");
const inferenceRevision = "65778fb790fa631597fd2739921a669b275d4429";
const permanentPin = inferenceRevision;
const sources = [["starvector/svg-stack-simple", "1d2a96a17cc0c4c1f337b7631adc8c5885bc72ea"], ["starvector/svg-icons-simple", "e1918a27ba6649e856e5db0710d8a6c7046762c1"], ["starvector/svg-emoji-simple", "fa75b3617872ae57e6f3cb450aee65dbccbd69e0"], ["starvector/svg-fonts-simple", "453c739ea13ad2685127f721c333f14d99485299"]];
const models = { "1b": ["starvector-1b-im2svg", "starvector/starvector-1b-im2svg", "380ab95d25a8e9ab1dc825debe238b4953ae13b9"], "8b": ["starvector-8b-im2svg", "starvector/starvector-8b-im2svg", "518beea8dcb5f7a37c5911e92d1d62a76beee7f9"] };
const providers = { "mlx:1b": "mlx-starvector-1b", "mlx:8b": "mlx-starvector-8b", "candle-cuda:1b": "candle-starvector-1b", "candle-cuda:8b": "candle-starvector-8b" };
const caseRecord = (index, lpips) => ({ case_index: index, source: { dataset: sources[Math.floor(index / 30)][0], revision: sources[Math.floor(index / 30)][1], row_index: index % 30 }, source_svg_sha256: digest, input_png_sha256: digest, provider_transcript_sha256: digest, finish_reason: "complete_root", canonical_svg_sha256: digest, preview_png_sha256: digest, accepted: true, ssim: .9, lpips, latency_seconds: 1 });
function run(key, lpips) { const [backend, tier] = key.split(":"); const model = models[tier]; return { backend, provider_id: providers[key], tier, device: backend, model: { key: model[0], repository: model[1], revision: model[2], inventory_sha256: digest }, hardware: { runner_name: "fixture", os: "fixture", arch: "fixture", system_memory_total_bytes: 1000, baseline_available_bytes: 900, peak_process_rss_bytes: 100, accelerator: { name: "fixture", uuid: null, driver_runtime: "fixture", total_bytes: 1000, baseline_free_bytes: 900, peak_used_bytes: tier === "1b" ? 800 : 850, raw_probe_sha256: digest } }, image_quality: { cases: Array.from({ length: 120 }, (_, index) => caseRecord(index, lpips)) }, deterministic_parity: { case_count: 20, cases: Array.from({ length: 20 }, (_, case_index) => ({ case_index, seed: case_index, first_preview_png_sha256: digest, second_preview_png_sha256: digest, rendered_ssim: .999 })) }, lifecycle: { load: true, unload: true, reload: true, memory_reported: true }, limits: { complete_root: true, eos: true, token: true, byte: true, wall_time: true, cancellation: true }, lifecycle_memory_transcript_sha256: digest }; }
function suites() { const prompts = ["geometric badge", "isometric folder", "rounded calendar", "minimal rocket", "layered landscape", "abstract flower"]; return { execution: { repository: "SceneWorks/SceneWorks", workflow_run_id: "123", workflow_run_attempt: 1, head_sha: "0".repeat(40), started_at: "2026-08-29T00:00:00Z", completed_at: "2026-08-29T00:01:00Z", clean_tree: true }, producer: { command: "scripts/starvector-terminal-route.mjs", artifact_name: "fixture", transcript_sha256: digest, artifact_manifest_sha256: digest }, metric_identity: { rasterizer: "resvg-0.45", canvas: { width: 512, height: 512, background: "white", colorspace: "srgb8" }, ssim: { implementation: "skimage.metrics.structural_similarity", package_version: "0.25.2", lock_sha256: digest, data_range: 255, channel_axis: 2, gaussian_weights: true, sigma: 1.5, use_sample_covariance: false }, lpips: { implementation: "richzhang/lpips", package_version: "0.1.4", version: "0.1", net: "alex", eval_mode: true, rgb_normalization: "[-1,1]", lock_sha256: digest, linear_weights_sha256: "df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0", alexnet_weights_sha256: "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02" }, metric_transcript_sha256: digest }, inference_preflight: { workflow_run_id: "1", workflow_run_attempt: 1, head_sha: inferenceRevision, inventory_artifacts: [{ tier: "1b", sha256: digest }, { tier: "8b", sha256: digest }], hook_logs: [{ backend: "mlx", tier: "1b", sha256: digest }, { backend: "mlx", tier: "8b", sha256: digest }, { backend: "candle-cuda", tier: "1b", sha256: digest }, { backend: "candle-cuda", tier: "8b", sha256: digest }] }, hostile_sanitizer: { corpus_sha256: null, sanitizer_version: "fixture", cases: [] }, prompt_composition: { corpus_sha256: null, raster_provider_id: "fixture", raster_model: "fixture", raster_revision: "fixture", raster_inventory_sha256: digest, clip_provider_id: "fixture", clip_model: "fixture", clip_revision: "fixture", clip_inventory_sha256: digest, metric_transcript_sha256: digest, cases: [] } }; }
async function golden(root, sceneWorksRoot) { const validator = await import(pathToFileURL(path.join(inferenceRoot, "scripts/release/starvector_terminal_evidence.mjs")).href); const corpus = JSON.parse(await (await import("node:fs/promises")).readFile(path.join(inferenceRoot, "release/starvector-terminal-corpus-v1.json"), "utf8")); const payload = suites(); payload.execution.head_sha = execFileSync("git", ["rev-parse", "HEAD"], { cwd: sceneWorksRoot }).toString().trim(); const ownedHostile = Array.from({ length: 200 }, (_, case_index) => ({ case_index, case_id: `hostile-v1-${case_index}`, input_sha256: sha(validator.hostilePayload(case_index)), expected_policy: "reject_or_sanitize_inert", outcome: "rejected", error_code: "rejected", canonical_svg_sha256: null, preview_png_sha256: null, published_paths: [], staging_residue: [], result_contains_inline_svg: false })); const promptNames = ["geometric badge", "isometric folder", "rounded calendar", "minimal rocket", "layered landscape", "abstract flower"]; const ownedPrompt = Array.from({ length: 60 }, (_, case_index) => ({ case_index, case_id: `prompt-v1-${case_index}`, prompt_sha256: sha(`Create a ${promptNames[Math.floor(case_index / 10)]} vector illustration, variant ${case_index % 10}, with clear silhouette, balanced composition, and no text.`), raster_png_sha256: digest, vector_provider_transcript_sha256: digest, canonical_svg_sha256: digest, preview_png_sha256: digest, accepted: true, raster_prompt_cosine: .98, preview_prompt_cosine: .97, alignment_loss: .01 })); payload.hostile_sanitizer.corpus_sha256 = sha(ownedHostile.map((entry) => entry.input_sha256).join("\n")); payload.hostile_sanitizer.cases = ownedHostile; payload.prompt_composition.corpus_sha256 = sha(ownedPrompt.map((entry) => entry.prompt_sha256).join("\n")); payload.prompt_composition.cases = ownedPrompt;
  for (const [key, lpips] of [["mlx:1b", .1], ["mlx:8b", .08], ["candle-cuda:1b", .1], ["candle-cuda:8b", .08]]) { const dir = path.join(root, key); await mkdir(dir, { recursive: true }); await writeFile(path.join(dir, "raw-results.json"), JSON.stringify({ tuple: key, run: run(key, lpips) })); }
  await mkdir(path.join(root, "suites"), { recursive: true }); await writeFile(path.join(root, "suites", "terminal-suites.json"), JSON.stringify(payload)); const draft = { schema_version: 1, campaign_run_id: "campaign", inference_revision: inferenceRevision, sceneworks_revision: execFileSync("git", ["rev-parse", "HEAD"], { cwd: sceneWorksRoot }).toString().trim(), corpus_sha256: validator.validatePlan(corpus), execution: { ...payload.execution, head_sha: execFileSync("git", ["rev-parse", "HEAD"], { cwd: sceneWorksRoot }).toString().trim() }, producer: payload.producer, metric_identity: payload.metric_identity, inference_preflight: payload.inference_preflight, runs: [["mlx:1b", .1], ["mlx:8b", .08], ["candle-cuda:1b", .1], ["candle-cuda:8b", .08]].map(([key, value]) => run(key, value)), hostile_sanitizer: payload.hostile_sanitizer, prompt_composition: payload.prompt_composition, artifact_manifest: null }; for (const entry of validator.buildArtifactManifest(draft, corpus).entries) { const tuple = entry.path.match(/^runs\/([^/]+:[^/]+)\//)?.[1], sourceRoot = tuple ? path.join(root, tuple) : path.join(root, "suites"), portable = entry.path.split("/").map((part) => part.replaceAll(":", "__colon__")); const file = path.join(sourceRoot, ...portable); const hostileMatch = entry.path.match(/^hostile\/(\d+)\/input$/), promptMatch = entry.path.match(/^prompt\/(\d+)\/prompt_sha256$/); const bytes = hostileMatch ? validator.hostilePayload(Number(hostileMatch[1])) : promptMatch ? `Create a ${promptNames[Math.floor(Number(promptMatch[1]) / 10)]} vector illustration, variant ${Number(promptMatch[1]) % 10}, with clear silhouette, balanced composition, and no text.` : "artifact"; await mkdir(path.dirname(file), { recursive: true }); await writeFile(file, bytes); } return { validator, corpus, payload }; }
async function fakeSceneWorks(root) { await mkdir(path.join(root, "scripts"), { recursive: true }); await writeFile(path.join(root, "Cargo.toml"), `[workspace]\n[workspace.dependencies]\ncandle-kernels = { git = "https://github.com/SceneWorks/inference", rev = "${inferenceRevision}" }\n`); await writeFile(path.join(root, "scripts", "starvector-terminal-route.mjs"), "export {};\n"); await writeFile(path.join(root, "scripts", "starvector-terminal-metrics.py"), "# fixture\n"); execFileSync("git", ["init", "-q"], { cwd: root }); execFileSync("git", ["config", "user.email", "fixture@example.invalid"], { cwd: root }); execFileSync("git", ["config", "user.name", "fixture"], { cwd: root }); execFileSync("git", ["add", "Cargo.toml", "scripts"], { cwd: root }); execFileSync("git", ["commit", "-qm", "fixture"], { cwd: root }); return root; }
async function pinnedInferenceCheckout(root) { const checkout = path.join(root, "pinned-inference"); execFileSync("git", ["-C", inferenceRepository, "worktree", "add", "--detach", "-q", checkout, inferenceRevision]); return checkout; }
async function removePinnedInference(checkout) { execFileSync("git", ["-C", inferenceRepository, "worktree", "remove", "--force", checkout]); }

test("exact pinned validator accepts sealed golden receipt and rejects every gate mutation", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "starvector-terminal-")); inferenceRoot = await pinnedInferenceCheckout(root); const sceneWorksRoot = await fakeSceneWorks(path.join(root, "sceneworks")); const evidence = path.join(root, "evidence"), output = path.join(root, "output"); await mkdir(evidence); const { validator, corpus } = await golden(evidence, sceneWorksRoot);
  await assert.rejects(() => sealReceipt({ sceneWorksRoot, planPath: path.join(process.cwd(), "release/starvector-terminal-campaign-v1.json"), inferenceRoot, evidenceRoot: evidence, output: path.join(root, "real-output"), campaignRunId: "campaign", permanentPin }), /metrics\/2 source digest drifted/);
  const receipt = await sealReceipt({ sceneWorksRoot, planPath: path.join(process.cwd(), "release/starvector-terminal-campaign-v1.json"), inferenceRoot, evidenceRoot: evidence, output, campaignRunId: "campaign", permanentPin, syntheticFixture: true });
  validator.validateReceipt(receipt, validator.validatePlan(corpus), inferenceRevision, receipt.sceneworks_revision, corpus);
  const reject = (mutate) => { const copy = structuredClone(receipt); mutate(copy); assert.throws(() => validator.validateReceipt(copy, validator.validatePlan(corpus), inferenceRevision, copy.sceneworks_revision, corpus)); };
  reject((r) => { for (let i = 0; i < 7; i += 1) { const c = r.runs[0].image_quality.cases[i]; c.accepted = false; c.ssim = c.lpips = c.canonical_svg_sha256 = c.preview_png_sha256 = null; } });
  reject((r) => { r.runs[0].image_quality.cases.forEach((c) => { c.ssim = .84; }); });
  reject((r) => { r.runs[0].image_quality.cases.forEach((c) => { c.lpips = .21; }); });
  reject((r) => { r.runs[0].image_quality.cases.slice(113).forEach((c) => { c.latency_seconds = 121; }); });
  reject((r) => { r.runs[0].deterministic_parity.cases[0].rendered_ssim = .994; });
  reject((r) => { r.runs[0].hardware.accelerator.peak_used_bytes = 900; });
  reject((r) => { r.runs[0].lifecycle.reload = false; }); reject((r) => { r.runs[0].limits.token = false; });
  reject((r) => { r.runs[1].image_quality.cases.forEach((c) => { c.lpips = .095; }); });
  reject((r) => { for (let i = 0; i < 4; i += 1) { const c = r.runs[1].image_quality.cases[i]; c.accepted = false; c.ssim = c.lpips = c.canonical_svg_sha256 = c.preview_png_sha256 = null; } });
  reject((r) => { r.runs[1].image_quality.cases.forEach((c, index) => { c.lpips = index < 59 ? .2 : .05; }); });
  reject((r) => { r.prompt_composition.cases.forEach((c) => { c.alignment_loss = .03; c.preview_prompt_cosine = .95; }); });
  reject((r) => { r.runs[0].image_quality.cases[0].source.revision = "0".repeat(40); });
  await removePinnedInference(inferenceRoot); inferenceRoot = inferenceRepository;
});

test("sealer reconstructs exact colon-bearing canonical paths from portable tuple artifacts", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "starvector-canonical-")), tupleRoot = path.join(root, "tuple"), suiteRoot = path.join(root, "suite"), canonical = path.join(root, "canonical");
  const bytes = Buffer.from("bound artifact"), digest = sha(bytes), logical = "runs/candle-cuda:8b/cases/0/preview", portable = logical.replace("candle-cuda:8b", "candle-cuda__colon__8b");
  const source = path.join(tupleRoot, ...portable.split("/")); await mkdir(path.dirname(source), { recursive: true }); await writeFile(source, bytes);
  const validator = { buildArtifactManifest: () => ({ entries: [{ path: logical, sha256: digest }] }) };
  await consolidateCanonicalArtifacts({}, {}, validator, canonical, new Map([["candle-cuda:8b", tupleRoot]]), suiteRoot);
  assert.equal(await (await import("node:fs/promises")).readFile(path.join(canonical, ...logical.split("/")), "utf8"), "bound artifact");
  await writeFile(source, "drift");
  await assert.rejects(() => consolidateCanonicalArtifacts({}, {}, validator, path.join(root, "bad"), new Map([["candle-cuda:8b", tupleRoot]]), suiteRoot), /source digest drifted/);
});

test("preflight requires the exact clean inference checkout and current Cargo permanent pin", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "starvector-scene-")); inferenceRoot = await pinnedInferenceCheckout(root); const sceneWorksRoot = await fakeSceneWorks(path.join(root, "sceneworks"));
  await verifyInferenceCheckout(inferenceRoot); await verifyPermanentPin(sceneWorksRoot, permanentPin);
  await assert.rejects(() => verifyPermanentPin(sceneWorksRoot, "0".repeat(40)), /terminal inference revision/);
  await removePinnedInference(inferenceRoot); inferenceRoot = inferenceRepository;
});

test("inference preflight requires every exact inventory and native-hook artifact", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "starvector-preflight-"));
  const file = async (name) => { await writeFile(path.join(root, name), "artifact"); return { path: name, sha256: digest }; };
  const index = { workflow_run_id: "run", workflow_run_attempt: 1, head_sha: permanentPin, inventory_artifacts: [{ tier: "1b", ...(await file("one")) }, { tier: "8b", ...(await file("eight")) }], hook_logs: [] };
  for (const key of ["mlx:1b", "mlx:8b", "candle-cuda:1b", "candle-cuda:8b"]) { const [backend, tier] = key.split(":"); index.hook_logs.push({ backend, tier, ...(await file(key.replace(":", "-"))) }); }
  const source = path.join(root, "preflight.json"); await writeFile(source, JSON.stringify(index));
  const previous = process.env.STARVECTOR_TERMINAL_INFERENCE_PREFLIGHT; process.env.STARVECTOR_TERMINAL_INFERENCE_PREFLIGHT = source;
  const verified = await validateInferencePreflight(root, permanentPin); assert.equal(Object.keys(verified.sources).length, 6);
  await writeFile(path.join(root, "mlx-1b"), "drift"); await assert.rejects(() => validateInferencePreflight(root, permanentPin), /missing or mismatched/);
  await writeFile(path.join(root, "mlx-1b"), "artifact"); await symlink(path.join(root, "mlx-1b"), path.join(root, "linked-hook")); index.hook_logs[0].path = "linked-hook"; await writeFile(source, JSON.stringify(index)); await assert.rejects(() => validateInferencePreflight(root, permanentPin), /missing or mismatched/);
  index.hook_logs[0].path = "mlx-1b";
  index.hook_logs.pop(); await writeFile(source, JSON.stringify(index)); await assert.rejects(() => validateInferencePreflight(root, permanentPin), /identity is invalid/);
  if (previous === undefined) delete process.env.STARVECTOR_TERMINAL_INFERENCE_PREFLIGHT; else process.env.STARVECTOR_TERMINAL_INFERENCE_PREFLIGHT = previous;
});

test("persistent permanent-pin campaign marker refuses a second campaign identity", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "starvector-marker-"));
  await claimCampaignMarker(root, permanentPin, "campaign-a"); await claimCampaignMarker(root, permanentPin, "campaign-a");
  await assert.rejects(() => claimCampaignMarker(root, permanentPin, "campaign-b"), /different terminal campaign marker/);
});

test("tuple marker, bytewise inventory, and symlink policy prevent mixed evidence", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "starvector-marker-"));
  await claimTupleMarker(root, permanentPin, "campaign", "mlx:1b");
  await assert.rejects(() => claimTupleMarker(root, permanentPin, "campaign", "mlx:1b"), /tuple already/);
  await mkdir(path.join(root, "nested")); await writeFile(path.join(root, "nested", "z"), "z"); await writeFile(path.join(root, "A"), "a");
  const listed = await inventory(root); assert.deepEqual(listed.entries.map((entry) => entry.path), [...listed.entries.map((entry) => entry.path)].sort());
  await symlink(path.join(root, "A"), path.join(root, "link")); await assert.rejects(() => inventory(root), /rejects symlink/);
});
