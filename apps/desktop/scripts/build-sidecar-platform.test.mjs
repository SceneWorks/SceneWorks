import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  ffmpegPlaceholder,
  onnxruntimePlaceholder,
  shouldBuildCandle,
} from "./build-sidecar-platform.mjs";

const scriptsDir = dirname(fileURLToPath(import.meta.url));

test("Linux always selects the embedded candle API build", () => {
  assert.equal(shouldBuildCandle("linux"), true);
  assert.equal(shouldBuildCandle("linux", "0"), true);
});

test("Windows preserves its candle opt-out", () => {
  assert.equal(shouldBuildCandle("win32"), true);
  assert.equal(shouldBuildCandle("win32", "1"), true);
  assert.equal(shouldBuildCandle("win32", "0"), false);
});

test("macOS preserves its non-candle MLX build", () => {
  assert.equal(shouldBuildCandle("darwin"), false);
});

test("Linux resource placeholders satisfy both Tauri globs", () => {
  const ort = onnxruntimePlaceholder("linux");
  const ffmpeg = ffmpegPlaceholder("linux");

  assert.match(ort, /not bundled on Linux yet/);
  assert.match(ort, /sc-10376/);
  assert.match(ffmpeg, /not bundled on Linux yet/);
  assert.match(ffmpeg, /PATH/);
  assert.match(ffmpeg, /sc-10376/);
});

test("build-sidecar stages files for every configured resource glob", () => {
  const config = JSON.parse(
    readFileSync(join(scriptsDir, "..", "tauri.conf.json"), "utf8"),
  );
  assert.ok(config.bundle.resources.includes("onnxruntime/**/*"));
  assert.ok(config.bundle.resources.includes("ffmpeg/**/*"));

  const source = readFileSync(join(scriptsDir, "build-sidecar.mjs"), "utf8");
  assert.match(source, /onnxruntimePlaceholder\(process\.platform\)/);
  assert.match(source, /ffmpegPlaceholder\(process\.platform\)/);
});

test("Windows placeholder guidance remains Windows-specific", () => {
  assert.match(onnxruntimePlaceholder("win32"), /%APPDATA%/);
  assert.match(ffmpegPlaceholder("win32"), /Windows uses PATH ffmpeg/);
});
