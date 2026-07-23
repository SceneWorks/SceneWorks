import assert from "node:assert/strict";
import {
  mkdtempSync,
  readFileSync,
  readdirSync,
  rmSync,
} from "node:fs";
import os from "node:os";
import { dirname, join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  sidecarBuildPlan,
  stageNonMacResourcePlaceholders,
} from "./build-sidecar-platform.mjs";

const scriptsDir = dirname(fileURLToPath(import.meta.url));

test("Linux plans the embedded candle API build with compute capability 80", () => {
  assert.deepEqual(sidecarBuildPlan("linux"), {
    candle: true,
    npmScript: "api:build:embedded:candle",
    computeCap: "80",
    env: { VITE_API_BASE_URL: "", CUDA_COMPUTE_CAP: "80" },
  });
  assert.deepEqual(sidecarBuildPlan("linux", { SCENEWORKS_DESKTOP_CANDLE: "0" }), {
    candle: true,
    npmScript: "api:build:embedded:candle",
    computeCap: "80",
    env: { VITE_API_BASE_URL: "", CUDA_COMPUTE_CAP: "80" },
  });
  assert.equal(
    sidecarBuildPlan("linux", { CUDA_COMPUTE_CAP: "90" }).computeCap,
    "90",
  );
});

test("Windows preserves its candle opt-out", () => {
  assert.equal(sidecarBuildPlan("win32").candle, true);
  assert.equal(
    sidecarBuildPlan("win32", { SCENEWORKS_DESKTOP_CANDLE: "1" }).candle,
    true,
  );
  assert.deepEqual(
    sidecarBuildPlan("win32", { SCENEWORKS_DESKTOP_CANDLE: "0" }),
    {
      candle: false,
      npmScript: "api:build:embedded",
      env: { VITE_API_BASE_URL: "" },
    },
  );
});

test("macOS preserves its non-candle MLX build", () => {
  assert.deepEqual(sidecarBuildPlan("darwin"), {
    candle: false,
    npmScript: "api:build:embedded",
    env: { VITE_API_BASE_URL: "" },
  });
});

test("Linux stages real files for every configured resource glob", (t) => {
  const tempDir = mkdtempSync(join(os.tmpdir(), "sceneworks-sidecar-test-"));
  t.after(() => rmSync(tempDir, { recursive: true }));
  const onnxruntimeDir = join(tempDir, "onnxruntime");
  const ffmpegDir = join(tempDir, "ffmpeg");
  const staged = stageNonMacResourcePlaceholders("linux", {
    onnxruntimeDir,
    ffmpegDir,
  });
  const config = JSON.parse(
    readFileSync(join(scriptsDir, "..", "tauri.conf.json"), "utf8"),
  );
  for (const glob of ["onnxruntime/**/*", "ffmpeg/**/*"]) {
    assert.ok(config.bundle.resources.includes(glob));
    const resourceDir = join(tempDir, glob.split("/")[0]);
    assert.ok(readdirSync(resourceDir).length > 0, `${glob} must match a file`);
  }
  assert.match(readFileSync(staged.onnxruntimeReadme, "utf8"), /sc-10376/);
  assert.match(readFileSync(staged.ffmpegReadme, "utf8"), /PATH.*sc-10376/);
});

test("Windows stages its existing platform-specific placeholder guidance", (t) => {
  const tempDir = mkdtempSync(join(os.tmpdir(), "sceneworks-sidecar-test-"));
  t.after(() => rmSync(tempDir, { recursive: true }));
  const staged = stageNonMacResourcePlaceholders("win32", {
    onnxruntimeDir: join(tempDir, "onnxruntime"),
    ffmpegDir: join(tempDir, "ffmpeg"),
  });
  assert.match(readFileSync(staged.onnxruntimeReadme, "utf8"), /%APPDATA%/);
  assert.match(
    readFileSync(staged.ffmpegReadme, "utf8"),
    /Windows uses PATH ffmpeg/,
  );
});
