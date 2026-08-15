#!/usr/bin/env node
/**
 * Generate the frozen SC-18946 LTX capture plans. The campaign is deliberately split so
 * single-pass fit/held-out records can be promoted without pooling the rung-2 selector or
 * unscored host-risk probes into a regression.
 */
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
export const STORY = "sc-18946";
export const FINGERPRINT = "sc-19109-ltx-2-3-mlx-memory-ladder-v1";
export const INFERENCE_REVISION = "0e4adc6f05884e6891e29dfecd32ee695efe8b18";
export const SEED = 18946;
// Frozen serialized order: establish the smallest tier first at each rung-2 boundary, then advance
// q4 -> q8 -> bf16 before moving from f305 to f449. The runbook still selects one provider by name
// at a time, but the checked-in plan must preserve that reviewed operator order too.
export const TIERS = Object.freeze(["q4", "q8", "bf16"]);
export const RUNG2_PARAMETERS = Object.freeze({ decodeTileEdge: 384, decodeOverlap: 64 });
const PROVIDER = "ltx_2_3";

const singlePassRows = Object.freeze([
  // The original six fit roles, reseeded under the permanent provider identity.
  [768, 512, 121, 30, "fit", "original_fit"],
  [768, 512, 241, 30, "fit", "original_fit"],
  [768, 512, 361, 30, "fit", "original_fit"],
  [1280, 704, 121, 30, "fit", "original_fit"],
  [1280, 704, 145, 30, "fit", "original_fit"],
  [1280, 704, 177, 30, "fit", "original_fit"],
  // Deep conditioning-bound anchors.
  [768, 512, 97, 30, "fit", "deep_conditioning"],
  [640, 640, 97, 30, "fit", "deep_conditioning"],
  [1280, 704, 97, 30, "fit", "deep_conditioning"],
  // Predeclared brackets around the three historical q8 geometry flip bands.
  [768, 512, 257, 30, "fit", "flip_band"],
  [768, 512, 265, 30, "fit", "flip_band"],
  [640, 640, 249, 30, "fit", "flip_band"],
  [640, 640, 257, 30, "fit", "flip_band"],
  [1280, 704, 113, 30, "fit", "flip_band"],
  // Identical held-out transfer matrix in every numeric tier.
  [640, 640, 177, 30, "held_out", "coefficient_transfer"],
  [512, 768, 361, 30, "held_out", "coefficient_transfer"],
  [704, 1280, 177, 30, "held_out", "coefficient_transfer"],
  [768, 512, 241, 24, "held_out_fps_probe", "coefficient_transfer"],
]);

const rung2Rows = Object.freeze([
  [1280, 704, 305, 30, "rung2_boundary", "rung2_first"],
  [1280, 704, 449, 30, "rung2_boundary", "rung2_first"],
]);

const hostRiskRows = Object.freeze([
  [768, 512, 449, 30, "planned_host_risk", "single_pass_host_risk"],
  [1280, 704, 241, 30, "planned_host_risk", "single_pass_host_risk"],
  [1280, 704, 297, 30, "planned_host_risk", "single_pass_write_cap"],
]);

function writableFrameCap(width, height) {
  return Math.floor(2_147_483_647 / (8 * width * height));
}

function providerRow(tier, [width, height, frames, fps, role, campaignPhase], rung) {
  const bounded = rung === "bounded_decode";
  const parameters = bounded ? RUNG2_PARAMETERS : {};
  const outputVoxels = width * height * frames;
  const latentTokens = ((frames - 1) / 8 + 1) * (width / 32) * (height / 32);
  const row = {
    name: `mlx-ltx-2-3-${tier}-${width}x${height}-f${frames}-fps${fps}-${rung}`,
    _role: role,
    _campaignPhase: campaignPhase,
    _coverageExpectation: "captured_or_attempted_or_arithmetic_unmeasurable",
    _latentTokens: latentTokens,
    _writableFrameCap: writableFrameCap(width, height),
    _outputVoxels: outputVoxels,
    evidenceScope: "authoritative",
    backend: "mlx",
    loadShape: "eager_materialization",
    target: {
      provider: PROVIDER,
      modelId: PROVIDER,
      tier,
      mode: "text_to_video",
      overlay: "none",
      geometry: { width, height, batch: 1, frames },
    },
    rung,
    engagedRungs: bounded
      ? ["resident", "staged_residency", "bounded_decode"]
      : ["resident", "staged_residency"],
    calibrationFingerprint: FINGERPRINT,
    fixture: `ltx-2-3-mlx-${tier}-${width}x${height}-f${frames}-fps${fps}-seed${SEED}`,
    cases: [{ parameters, expectedResult: "passed" }],
  };
  if (bounded) {
    row._predictedDecodeBytes =
      3_300_000_000 + 40 * outputVoxels + 300 * 384 * 384 * 96;
    row._predictedDecodeFormula =
      "3.3e9 + 40*(width*height*frames) + 300*(384*384*96) bytes";
  }
  return row;
}

function plan(comment, rows, rung, { rowMajor = false } = {}) {
  return {
    _comment: comment,
    story: STORY,
    campaignIdentity: {
      inferenceRevision: INFERENCE_REVISION,
      calibrationFingerprint: FINGERPRINT,
      seed: SEED,
    },
    providers: rowMajor
      ? rows.flatMap((row) => TIERS.map((tier) => providerRow(tier, row, rung)))
      : TIERS.flatMap((tier) => rows.map((row) => providerRow(tier, row, rung))),
  };
}

export function buildPlans() {
  return {
    "ltx-mlx-single-pass-sweep.json": plan(
      "SC-18946 fit and held-out matrix. Only staged-residency/single-pass rows live here so every record is eligible for selector-scoped fitting. Roles are frozen before capture; q4, q8 and bf16 use the identical geometry and held-out split.",
      singlePassRows,
      "staged_residency",
    ),
    "ltx-mlx-rung2-sweep.json": plan(
      "SC-18946 rung-2-first matrix. Every q4/q8/bf16 row explicitly selects decodeTileEdge=384 and decodeOverlap=64 at 1280x704 f305/f449. These records are ingested as canonical evidence but never pooled into the single-pass temporal fit.",
      rung2Rows,
      "bounded_decode",
      { rowMajor: true },
    ),
    "ltx-mlx-host-risk-probes.json": {
      ...plan(
        "SC-18946 predeclared single-pass host-risk probes. Their coverage state must distinguish captured, attempted/aborted, arithmetic-unmeasurable, and never planned. This unscored bundle is not a curve source.",
        hostRiskRows,
        "staged_residency",
      ),
      providers: [
        ...TIERS.flatMap((tier) => hostRiskRows.map((row) => providerRow(tier, row, "staged_residency"))),
        providerRow("bf16", [768, 512, 145, 24, "reproduction_probe", "sc18809_reproduction"], "staged_residency"),
      ],
    },
  };
}

async function main() {
  const check = process.argv.includes("--check");
  const outputDir = path.join(ROOT, "docs/calibration/sc-18946");
  if (!check) await mkdir(outputDir, { recursive: true });
  const plans = buildPlans();
  const stale = [];
  for (const [name, value] of Object.entries(plans)) {
    const file = path.join(outputDir, name);
    const bytes = `${JSON.stringify(value, null, 2)}\n`;
    if (check) {
      const existing = await readFile(file, "utf8").catch(() => null);
      if (existing !== bytes) stale.push(path.relative(ROOT, file));
    } else {
      await writeFile(file, bytes);
      process.stdout.write(`wrote ${path.relative(ROOT, file)}\n`);
    }
  }
  if (stale.length > 0) throw new Error(`stale SC-18946 plan(s): ${stale.join(", ")}`);
  if (check) process.stdout.write("SC-18946 plans are current\n");
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((error) => {
    process.stderr.write(`${error.stack ?? error}\n`);
    process.exitCode = 1;
  });
}
