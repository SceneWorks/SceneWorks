#!/usr/bin/env node
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { isExecutedModule } from "./starvector-terminal-cli.mjs";

const sha = (value) => createHash("sha256").update(value).digest("hex");
export const TUPLES = ["mlx:1b", "mlx:8b", "candle-cuda:1b", "candle-cuda:8b"];
export const COUNTS = { image_quality: 120, deterministic_parity: 20, hostile_sanitizer: 200, prompt_composition: 60 };
export const INFERENCE_REVISION = "c6d6a4dbd61ab09c26ff5526632cae2cefea60ed";
export const INFERENCE_PREFLIGHT = Object.freeze({
  workflow_run_id: "33851645747",
  workflow_run_attempt: 1,
  head_sha: INFERENCE_REVISION,
  artifact: {
    id: 9928624696,
    name: "starvector-terminal-preflight-c6d6a4dbd61ab09c26ff5526632cae2cefea60ed-33851645747-1",
    digest: "sha256:4df39fc45d36ef11f968aa82c48eda6292f48c54086a4beee4ff3f6e8ba48226",
  },
  inventory_artifacts: [
    { tier: "1b", path: "inventory/starvector-1b-inventory.json", sha256: "f4b8345ae7b6aa535080191c05694bba68fb3bbfe0391ff95f5bfd9b381812da" },
    { tier: "8b", path: "inventory/starvector-8b-inventory.json", sha256: "af1bcb4c38b86bbe1a973aedcba2ca72b03485d2c5877457192878beeb5989a2" },
  ],
  hook_logs: [
    { backend: "mlx", tier: "1b", path: "hooks/mlx-starvector-1b.log", sha256: "4c06aa10dfa65cf575e741df0632baa03970a277485eb93ca69dca3070d3828b" },
    { backend: "mlx", tier: "8b", path: "hooks/mlx-starvector-8b.log", sha256: "4c2e426c53ac0728fa20765dbedbac1440766acc4b52670acdf0e7c574bb2da3" },
    { backend: "candle-cuda", tier: "1b", path: "hooks/candle-cuda-starvector-1b.log", sha256: "c55e5ed6d44c670ccdb06315106e136e769b67ce16de3a43a1260c7f75e47d03" },
    { backend: "candle-cuda", tier: "8b", path: "hooks/candle-cuda-starvector-8b.log", sha256: "0f34fe938c1b9491a8380b5b27583735546b1afd39f1c21abdaedb09550bddc4" },
  ],
});
export const LPIPS_LINEAR_SHA256 = "df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0";
export const ALEXNET_SHA256 = "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02";

export function terminalSourceRowRecord(row) {
  return JSON.stringify({ dataset: row.dataset, revision: row.revision, row_index: row.row_index, filename: row.filename, svg_sha256: row.svg_sha256 });
}

export function serializeTerminalSourceRows(rows) {
  return `${rows.map(terminalSourceRowRecord).join("\n")}\n`;
}

export function terminalSourceRowsSha256(rows) {
  return sha(serializeTerminalSourceRows(rows));
}

const fail = (message) => { throw new Error(`starvector terminal campaign: ${message}`); };

export function validateMetricsLock(lock) {
  if (lock?.schema_version !== 1 || lock?.canvas?.width !== 512 || lock.canvas.height !== 512 || lock.canvas.background !== "white" || lock.canvas.colorspace !== "srgb8") fail("metric canvas must be fixed 512x512 white sRGB8");
  const s = lock.ssim;
  if (s?.implementation !== "skimage.metrics.structural_similarity" || s.data_range !== 255 || s.channel_axis !== 2 || s.gaussian_weights !== true || s.sigma !== 1.5 || s.use_sample_covariance !== false) fail("SSIM identity must be exact scikit-image RGB settings");
  const l = lock.lpips;
  if (l?.implementation !== "richzhang/lpips" || l.constructor !== "LPIPS(net='alex', version='0.1')" || l.eval_mode !== true || l.input_rgb_range !== "[-1,1]" || l.linear_weights_sha256 !== LPIPS_LINEAR_SHA256 || l.alexnet_weights_sha256 !== ALEXNET_SHA256) fail("LPIPS identity or weights are not pinned");
  if (JSON.stringify(lock.required_packages) !== JSON.stringify([{ name: "numpy", version: "2.2.6" }, { name: "scikit-image", version: "0.25.2" }, { name: "lpips", version: "0.1.4" }, { name: "torch", version: "2.7.0" }, { name: "torchvision", version: "0.22.0" }, { name: "Pillow", version: "11.3.0" }, { name: "open-clip-torch", version: "3.1.0" }])) fail("metric package closure changed");
  if (lock.clip?.implementation !== "mlfoundations/open_clip" || lock.clip.package !== "open-clip-torch" || lock.clip.model !== "ViT-B-32" || lock.clip.checkpoint_must_be_local !== true || lock.clip.input !== "512px preview resized by locked OpenCLIP preprocess" || lock.clip.normalization !== "OpenCLIP ViT-B-32 default") fail("local OpenCLIP metric identity changed");
  return lock;
}

export function validatePlan(plan) {
  if (plan?.schema_version !== 1 || plan?.inference_contract?.revision !== INFERENCE_REVISION || plan.inference_contract.repository !== "SceneWorks/inference") fail("inference contract identity changed");
  if (JSON.stringify(plan.inference_preflight) !== JSON.stringify(INFERENCE_PREFLIGHT)) fail("inference preflight provenance changed");
  if (JSON.stringify(plan.tuples) !== JSON.stringify(TUPLES)) fail("tuples must remain MLX 1B, MLX 8B, CUDA 1B, CUDA 8B in order");
  if (JSON.stringify(plan.counts) !== JSON.stringify(COUNTS)) fail("terminal corpus counts changed");
  if (plan.model_snapshot_revisions?.["starvector-1b"] !== "380ab95d25a8e9ab1dc825debe238b4953ae13b9" || plan.model_snapshot_revisions?.["starvector-8b"] !== "518beea8dcb5f7a37c5911e92d1d62a76beee7f9") fail("StarVector snapshot revision changed");
  if (plan.metrics_lock !== "release/starvector-terminal-metrics-lock-v1.json") fail("metric lock path changed");
  for (const key of ["dispatch_only", "no_job_time_downloads", "single_permanent_pin_run", "upload_on_failure", "fail_closed"]) if (plan.policy?.[key] !== true) fail(`policy ${key}`);
  return plan;
}

export async function readPlanAndLock(planPath) {
  const plan = validatePlan(JSON.parse(await readFile(planPath, "utf8")));
  const lockPath = new URL(`../${plan.metrics_lock}`, import.meta.url);
  const lockBytes = await readFile(lockPath);
  return { plan, lock: validateMetricsLock(JSON.parse(lockBytes)), metrics_lock_sha256: sha(lockBytes) };
}

if (isExecutedModule(import.meta.url)) {
  try {
    const { plan, metrics_lock_sha256 } = await readPlanAndLock(process.argv[2] ?? "release/starvector-terminal-campaign-v1.json");
    console.log(JSON.stringify({ inference_revision: plan.inference_contract.revision, tuples: plan.tuples, metrics_lock_sha256 }));
  } catch (error) {
    console.error(error.message);
    process.exitCode = 1;
  }
}
