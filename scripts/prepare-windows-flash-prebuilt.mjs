#!/usr/bin/env node
// Preserve the frozen Cargo selector used by the matched-resident controller.
// Candle build jobs call this entry point to select that cache and fetch FlashAttention.
import { execFileSync } from "node:child_process";
import { appendFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { selectWindowsCargoCache } from "./select-windows-cargo-cache.mjs";
import { candleFlashSourceSha256, fetchPrebuilt } from "./fetch-flash-prebuilt.mjs";

async function main() {
  const cacheOnly = process.argv.length === 3 && process.argv[2] === "--cache-only";
  if (process.argv.length > 2 && !cacheOnly) throw new Error("usage: prepare-windows-flash-prebuilt.mjs [--cache-only]");
  if (process.platform !== "win32") throw new Error("Windows Cargo cache selection requires Windows");
  const selector = path.join(path.dirname(fileURLToPath(import.meta.url)), "select-windows-cargo-cache.mjs");
  execFileSync(process.execPath, [selector], { stdio: "inherit" });
  if (cacheOnly) return;

  const selected = selectWindowsCargoCache({
    runnerName: process.env.RUNNER_NAME,
    cargoHome: process.env.CARGO_HOME,
    runnerToolCache: process.env.RUNNER_TOOL_CACHE,
  });
  process.env.CARGO_HOME = selected.cache;
  const sourceSha256 = await candleFlashSourceSha256(process.cwd());
  const directory = await fetchPrebuilt({
    sourceSha256,
    target: "x86_64-pc-windows-msvc",
    cacheRoot: path.join(selected.cache, "flash-attn-prebuilt"),
  });
  if (directory) {
    if (!process.env.GITHUB_ENV) throw new Error("GITHUB_ENV is required");
    await appendFile(process.env.GITHUB_ENV, `CANDLE_FLASH_ATTN_PREBUILT_DIR=${directory}\n`);
  }
  console.log(JSON.stringify({ kind: "windows-flash-prebuilt", cache: selected.cache, directory }));
}

main().catch(error => { console.error(`Windows FlashAttention preparation failed: ${error.message}`); process.exitCode = 1; });
