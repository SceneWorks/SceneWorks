import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
const fail=(m)=>{throw new Error(`starvector terminal campaign: ${m}`)};
export function validatePlan(plan) {
  if (plan.schema_version!==1 || plan.inference_contract?.revision!=="65778fb790fa631597fd2739921a669b275d4429") fail("immutable inference contract missing");
  if (JSON.stringify(plan.tuples)!==JSON.stringify(["mlx:1b","mlx:8b","candle-cuda:1b","candle-cuda:8b"])) fail("tuple order must be strictly serial");
  for (const [key,count] of Object.entries({image_quality:120,deterministic_parity:20,hostile_sanitizer:200,prompt_composition:60})) if(plan.counts?.[key]!==count) fail(`count ${key}`);
  if (plan.metrics?.canvas!=="512x512 white sRGB8" || !plan.metrics?.ssim?.includes("channel_axis=2") || !plan.metrics?.lpips?.includes("net=alex") || plan.metrics.lpips_linear_sha256!=="df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0" || plan.metrics.alexnet_sha256!=="7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02") fail("metric identity");
  for(const key of ["dispatch_only","no_job_time_downloads","single_permanent_pin_run","upload_on_failure","fail_closed"]) if(plan.policy?.[key]!==true) fail(`policy ${key}`);
  return createHash("sha256").update(JSON.stringify(plan)).digest("hex");
}
if(import.meta.url===`file://${process.argv[1]}`){try{console.log(validatePlan(JSON.parse(readFileSync(process.argv[2],"utf8"))))}catch(e){console.error(e.message);process.exitCode=1}}
