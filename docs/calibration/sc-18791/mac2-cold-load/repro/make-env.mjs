import { readFileSync, writeFileSync } from "node:fs";
import {
  expandPlan, selectPlanProviders, prepareLtx25CaptureArtifacts, ltx25ProviderEnvironment,
} from "/Volumes/Data/calibration/sc-18791/diagnostic/SceneWorks/scripts/memory-calibration-harness.mjs";

const SNAP = "/Volumes/Data/huggingface/hub/models--SceneWorks--ltx-2.5-mlx/snapshots/081658ce6886cacba20817ce0359bbefef706ff2";
const plan = JSON.parse(readFileSync("/Volumes/Data/calibration/sc-18791/diagnostic/SceneWorks/config/memory-calibration-plan.json", "utf8"));
const cases = expandPlan({ ...plan, providers: selectPlanProviders(plan, { model: "ltx_2_5" }) })
  .filter((c) => c.backend === "mlx");
const target = cases.find((c) =>
  c.target.tier === "q4" && c.target.transformerVariant === "dev" && c.target.decoder === "conv" &&
  c.target.geometry.width === 512 && c.target.geometry.height === 768 && c.target.geometry.frames === 145 &&
  c.loadShape === "eager_materialization" && (c.target.overlay ?? "none") === "none");
if (!target) throw new Error("case not found");
const prepared = await prepareLtx25CaptureArtifacts(SNAP, cases);
const env = ltx25ProviderEnvironment(prepared, target, {});
writeFileSync("/Volumes/Data/calibration/sc-18791/diagnostic/repro/case-env.sh",
  Object.entries(env).map(([k, v]) => `export ${k}=${JSON.stringify(String(v))}`).join("\n") + "\n");
console.log("wrote case-env.sh:", Object.keys(env).join(","));
