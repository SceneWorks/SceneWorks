// Verify all five canonical shipping `sga.capabilities/2` route-profile exports
// (`ios-media`, `ios-llm-only`, `candle-cpu`, `candle-cuda`, `macos-mlx`) without compiling or
// loading an engine or scraping Rust source files (`SW-6983`, `REG-3`, `M8`).
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

export const SHIPPING_PROFILES = [
  { file: "ios-media.json", featureSet: "ios-media", platform: "ios", backend: "mlx" },
  { file: "ios-llm-only.json", featureSet: "ios-llm-only", platform: "ios", backend: "mlx" },
  { file: "candle-cpu.json", featureSet: "candle-cpu", platform: "cpu", backend: "candle" },
  { file: "candle-cuda.json", featureSet: "candle-cuda", platform: "cuda", backend: "candle" },
  { file: "macos-mlx.json", featureSet: "macos-mlx", platform: "macos", backend: "mlx" },
];

export function verifyGenerationRouteProfiles(exportsByFeatureSet) {
  for (const spec of SHIPPING_PROFILES) {
    const doc = exportsByFeatureSet[spec.featureSet];
    assert.ok(doc && typeof doc === "object", `missing sga.capabilities/2 export: ${spec.featureSet}`);
    assert.equal(doc.schema_version, "sga.capabilities/2", `${spec.featureSet}: wrong schema_version`);
    assert.equal(doc.feature_set, spec.featureSet, `${spec.featureSet}: wrong feature_set`);
    assert.equal(doc.platform, spec.platform, `${spec.featureSet}: wrong platform`);
    assert.equal(doc.backend, spec.backend, `${spec.featureSet}: wrong backend`);
    assert.equal("generatedFrom" in doc, false, `${spec.featureSet}: must be pin-free (M8)`);
  }

  const iosMedia = exportsByFeatureSet["ios-media"];
  const iosLlmOnly = exportsByFeatureSet["ios-llm-only"];
  const candleCpu = exportsByFeatureSet["candle-cpu"];
  const candleCuda = exportsByFeatureSet["candle-cuda"];
  const macosMlx = exportsByFeatureSet["macos-mlx"];

  // iOS media profile has the curated compact image subset; iOS LLM-only has zero media routes.
  const iosMediaRoutes = Object.keys(iosMedia.routes).sort();
  assert.equal(iosMediaRoutes.length, 7, "ios-media must expose exactly 7 media routes");
  assert.ok(!iosMediaRoutes.some((id) => id.startsWith("flux1")), "ios-media must exclude flux1");
  assert.equal(Object.keys(iosLlmOnly.routes).length, 0, "ios-llm-only must expose 0 media routes");
  assert.equal(
    Object.keys(iosLlmOnly.transforms?.image ?? {}).length,
    0,
    "ios-llm-only must expose 0 image transforms",
  );
  assert.deepEqual(
    Object.keys(iosMedia.llm_models).sort(),
    Object.keys(iosLlmOnly.llm_models).sort(),
    "ios-media and ios-llm-only must share the exact iOS LLM model surface",
  );

  // StarVector Modality::Vector routes on desktop macOS + CUDA.
  for (const [label, doc] of [
    ["macos-mlx", macosMlx],
    ["candle-cuda", candleCuda],
  ]) {
    for (const id of ["starvector_1b", "starvector_8b"]) {
      const route = doc.routes[id];
      assert.ok(route, `${label} missing ${id} route`);
      assert.equal(route.modality, "vector", `${label}/${id} must have modality=vector`);
      assert.ok(route.modes.includes("image_to_svg"), `${label}/${id} must advertise image_to_svg`);
    }
  }

  // Candle CUDA registers all 4 Qwen-Image-Edit aliases + nvfp4 advanced tier; CPU omits nvfp4.
  for (const alias of [
    "qwen_image_edit",
    "qwen_image_edit_2509",
    "qwen_image_edit_2511",
    "qwen_image_edit_2511_lightning",
  ]) {
    assert.ok(candleCuda.routes[alias], `candle-cuda missing Qwen edit alias ${alias}`);
  }
  assert.ok(
    candleCuda.advanced_quant_tiers.includes("nvfp4"),
    "candle-cuda must declare nvfp4 advanced quant tier",
  );
  assert.deepEqual(candleCpu.advanced_quant_tiers, [], "candle-cpu must not declare CUDA quant tiers");
  assert.ok(Object.keys(candleCpu.routes).length > 50, "candle-cpu must carry full Candle media catalog");
  assert.ok(Object.keys(candleCpu.audio_routes).length > 10, "candle-cpu must carry Candle audio catalog");
  assert.ok(Object.keys(candleCpu.llm_models).includes("candle-llama"), "candle-cpu must carry candle-llama");

  return {
    schemaVersion: "sga.capabilities/2",
    profiles: SHIPPING_PROFILES.map((s) => s.featureSet),
    iosMediaRoutes: iosMediaRoutes.length,
    iosLlmOnlyRoutes: Object.keys(iosLlmOnly.routes).length,
    candleCpuRoutes: Object.keys(candleCpu.routes).length,
    candleCudaRoutes: Object.keys(candleCuda.routes).length,
    macosMlxRoutes: Object.keys(macosMlx.routes).length,
  };
}

export function loadShippingProfiles(
  sgaDir = fileURLToPath(new URL("../config/engine-capabilities/sga", import.meta.url)),
) {
  return Object.fromEntries(
    SHIPPING_PROFILES.map((spec) => [
      spec.featureSet,
      JSON.parse(readFileSync(path.join(sgaDir, spec.file), "utf8")),
    ]),
  );
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  const sgaDir =
    process.argv[2] ?? fileURLToPath(new URL("../config/engine-capabilities/sga", import.meta.url));
  const summary = verifyGenerationRouteProfiles(loadShippingProfiles(sgaDir));
  console.log(JSON.stringify(summary, null, 2));
}
