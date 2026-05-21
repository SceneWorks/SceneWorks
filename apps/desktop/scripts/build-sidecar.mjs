#!/usr/bin/env node
// Builds the sceneworks-rust-api binary (with the embedded web UI) and stages it
// as a Tauri sidecar named for the host target triple. Wired as the
// tauri.conf.json `beforeBuildCommand` so `tauri build` is self-contained.
import { execFileSync } from "node:child_process";
import { copyFileSync, mkdirSync, chmodSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import process from "node:process";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const desktopDir = resolve(scriptDir, ".."); // apps/desktop
const repoRoot = resolve(desktopDir, "..", ".."); // repository root
const npmCmd = process.platform === "win32" ? "npm.cmd" : "npm";

function run(cmd, args) {
  console.log(`> ${cmd} ${args.join(" ")}`);
  execFileSync(cmd, args, { stdio: "inherit", cwd: repoRoot });
}

// Host target triple, e.g. aarch64-apple-darwin or x86_64-pc-windows-msvc.
const triple = execFileSync("rustc", ["-vV"], { encoding: "utf8" }).match(
  /host:\s*(\S+)/,
)?.[1];
if (!triple) {
  console.error("build-sidecar: could not determine host target triple");
  process.exit(1);
}
const exe = triple.includes("windows") ? ".exe" : "";

// Build the web bundle + API binary with the embedded UI (single source of
// truth for the embedded build).
run(npmCmd, ["run", "api:build:embedded"]);

const src = join(repoRoot, "target", "release", `sceneworks-rust-api${exe}`);
const outDir = join(desktopDir, "binaries");
mkdirSync(outDir, { recursive: true });
const dest = join(outDir, `sceneworks-api-${triple}${exe}`);
copyFileSync(src, dest);
if (!exe) {
  chmodSync(dest, 0o755);
}
console.log(`build-sidecar: staged ${dest}`);
