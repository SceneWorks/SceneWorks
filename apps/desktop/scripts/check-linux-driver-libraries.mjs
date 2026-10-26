#!/usr/bin/env node
// Artifact regression check: the host supplies NVIDIA's driver; CUDA toolkit
// libraries (libcudart, cuBLAS, cuDNN, NVRTC, etc.) may remain in the AppImage.
import { spawnSync } from "node:child_process";
import { existsSync, mkdtempSync, readdirSync, rmSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, relative, resolve } from "node:path";
import { pathToFileURL } from "node:url";

export function isHostNvidiaDriverLibrary(name) {
  return (
    /^lib(?:cuda|cudadebugger|nvcuvid|nvoptix)\.so(?:\.|$)/.test(name) ||
    /^libnvidia-[^.]+\.so(?:\.|$)/.test(name) ||
    /^lib(?:EGL|GLX|GLESv[12])_nvidia\.so(?:\.|$)/.test(name)
  );
}

export function inspectAppDir(directory) {
  const root = resolve(directory);
  if (!statSync(join(root, "usr/bin/sceneworks-api")).isFile()) {
    throw new Error("AppDir must contain usr/bin/sceneworks-api");
  }
  const forbidden = [];
  const walk = (dir) => {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      const path = join(dir, entry.name);
      // Check names before following anything: even a dangling driver symlink
      // must fail. Never traverse symlink directories outside the artifact.
      if (isHostNvidiaDriverLibrary(entry.name)) {
        forbidden.push(relative(root, path).replaceAll("\\", "/"));
      }
      if (entry.isDirectory()) walk(path);
    }
  };
  walk(root);
  return { passed: forbidden.length === 0, forbidden: forbidden.sort() };
}

export function inspectAppImage(artifact) {
  if (process.platform !== "linux") {
    throw new Error("--appimage requires Linux; use --appdir for an extracted artifact elsewhere");
  }
  const path = resolve(artifact);
  if (!statSync(path).isFile()) throw new Error(`Not an AppImage file: ${path}`);
  const scratch = mkdtempSync(join(tmpdir(), "sceneworks-driver-check-"));
  try {
    // Runtime extraction works without FUSE. Discard its file-list stdout.
    const extracted = spawnSync(path, ["--appimage-extract"], {
      cwd: scratch,
      stdio: ["ignore", "ignore", "pipe"],
      encoding: "utf8",
      timeout: 120_000,
    });
    if (extracted.error || extracted.status !== 0) {
      throw new Error(`AppImage extraction failed: ${extracted.error?.message ?? extracted.stderr}`);
    }
    return inspectAppDir(join(scratch, "squashfs-root"));
  } finally {
    rmSync(scratch, { recursive: true, force: true });
  }
}

function main(args) {
  if (args.length !== 2 || !["--appdir", "--appimage"].includes(args[0])) {
    throw new Error("Usage: node check-linux-driver-libraries.mjs --appdir <AppDir> | --appimage <file>");
  }
  if (!existsSync(args[1])) throw new Error(`Artifact does not exist: ${args[1]}`);
  const report = args[0] === "--appimage" ? inspectAppImage(args[1]) : inspectAppDir(args[1]);
  console.log(JSON.stringify(report, null, 2));
  return report.passed ? 0 : 1;
}

if (process.argv[1] && pathToFileURL(resolve(process.argv[1])).href === import.meta.url) {
  try {
    process.exitCode = main(process.argv.slice(2));
  } catch (error) {
    console.error(error.message);
    process.exitCode = 2;
  }
}
