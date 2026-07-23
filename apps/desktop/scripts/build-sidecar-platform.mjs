import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

export function shouldBuildCandle(platform, desktopCandle) {
  if (platform === "linux") return true;
  return platform === "win32" && desktopCandle !== "0";
}

export function sidecarBuildPlan(platform, env = {}) {
  const candle = shouldBuildCandle(
    platform,
    env.SCENEWORKS_DESKTOP_CANDLE,
  );
  if (!candle) {
    return {
      candle,
      npmScript: "api:build:embedded",
      env: { VITE_API_BASE_URL: "" },
    };
  }

  const computeCap = env.CUDA_COMPUTE_CAP || "80";
  return {
    candle,
    npmScript: "api:build:embedded:candle",
    computeCap,
    env: { VITE_API_BASE_URL: "", CUDA_COMPUTE_CAP: computeCap },
  };
}

function onnxruntimePlaceholder(platform) {
  if (platform === "linux") {
    return (
      "onnxruntime is not bundled on Linux yet; install it on the host until " +
      "Linux runtime provisioning is implemented (sc-10376).\n"
    );
  }
  return (
    "onnxruntime is bundled on macOS (CoreML) only; the Windows candle build " +
    "downloads the CUDA onnxruntime on first run into " +
    "%APPDATA%\\SceneWorks\\gpu-runtime (cuda_provision.rs), not into this " +
    "resource dir (sc-3487 / sc-5496).\n"
  );
}

function ffmpegPlaceholder(platform) {
  if (platform === "linux") {
    return (
      "Static ffmpeg is not bundled on Linux yet; install ffmpeg on PATH until " +
      "Linux runtime provisioning is implemented (sc-10376).\n"
    );
  }
  return "Static ffmpeg is bundled on macOS only (sc-3767); Windows uses PATH ffmpeg.\n";
}

export function stageNonMacResourcePlaceholders(
  platform,
  { onnxruntimeDir, ffmpegDir },
) {
  const onnxruntimeReadme = join(onnxruntimeDir, "README.txt");
  const ffmpegReadme = join(ffmpegDir, "README.txt");
  mkdirSync(onnxruntimeDir, { recursive: true });
  mkdirSync(ffmpegDir, { recursive: true });
  writeFileSync(onnxruntimeReadme, onnxruntimePlaceholder(platform));
  writeFileSync(ffmpegReadme, ffmpegPlaceholder(platform));
  return { onnxruntimeReadme, ffmpegReadme };
}
