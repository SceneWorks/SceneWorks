export function shouldBuildCandle(
  platform,
  desktopCandle = process.env.SCENEWORKS_DESKTOP_CANDLE,
) {
  if (platform === "linux") return true;
  return platform === "win32" && desktopCandle !== "0";
}

export function onnxruntimePlaceholder(platform) {
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

export function ffmpegPlaceholder(platform) {
  if (platform === "linux") {
    return (
      "Static ffmpeg is not bundled on Linux yet; install ffmpeg on PATH until " +
      "Linux runtime provisioning is implemented (sc-10376).\n"
    );
  }
  return "Static ffmpeg is bundled on macOS only (sc-3767); Windows uses PATH ffmpeg.\n";
}
