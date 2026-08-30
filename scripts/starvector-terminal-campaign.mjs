#!/usr/bin/env node
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";

const sha = (value) => createHash("sha256").update(value).digest("hex");
export const TUPLES = ["mlx:1b", "mlx:8b", "candle-cuda:1b", "candle-cuda:8b"];
export const COUNTS = { image_quality: 120, deterministic_parity: 20, hostile_sanitizer: 200, prompt_composition: 60 };
export const INFERENCE_REVISION = "65778fb790fa631597fd2739921a669b275d4429";
export const LPIPS_LINEAR_SHA256 = "df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0";
export const ALEXNET_SHA256 = "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02";

const fail = (message) => { throw new Error(`starvector terminal campaign: ${message}`); };

export function validateMetricsLock(lock) {
  if (lock?.schema_version !== 1 || lock?.canvas?.width !== 512 || lock.canvas.height !== 512 || lock.canvas.background !== "white" || lock.canvas.colorspace !== "srgb8") fail("metric canvas must be fixed 512x512 white sRGB8");
  const s = lock.ssim;
  if (s?.implementation !== "skimage.metrics.structural_similarity" || s.data_range !== 255 || s.channel_axis !== 2 || s.gaussian_weights !== true || s.sigma !== 1.5 || s.use_sample_covariance !== false) fail("SSIM identity must be exact scikit-image RGB settings");
  const l = lock.lpips;
  if (l?.implementation !== "richzhang/lpips" || l.constructor !== "LPIPS(net='alex', version='0.1')" || l.eval_mode !== true || l.input_rgb_range !== "[-1,1]" || l.linear_weights_sha256 !== LPIPS_LINEAR_SHA256 || l.alexnet_weights_sha256 !== ALEXNET_SHA256) fail("LPIPS identity or weights are not pinned");
  if (JSON.stringify(lock.required_packages) !== JSON.stringify(["scikit-image", "lpips", "torch", "torchvision"])) fail("metric package closure changed");
  return lock;
}

export function validatePlan(plan) {
  if (plan?.schema_version !== 1 || plan?.inference_contract?.revision !== INFERENCE_REVISION || plan.inference_contract.repository !== "SceneWorks/inference") fail("inference contract identity changed");
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

if (import.meta.url === `file://${process.argv[1]}`) {
  try {
    const { plan, metrics_lock_sha256 } = await readPlanAndLock(process.argv[2] ?? "release/starvector-terminal-campaign-v1.json");
    console.log(JSON.stringify({ inference_revision: plan.inference_contract.revision, tuples: plan.tuples, metrics_lock_sha256 }));
  } catch (error) {
    console.error(error.message);
    process.exitCode = 1;
  }
}
