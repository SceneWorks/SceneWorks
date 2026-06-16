#!/usr/bin/env node
// Notarize + staple the macOS DMG. Tauri notarizes and staples the inner .app but
// leaves the wrapping .dmg without its own notarization ticket, so a freshly
// downloaded DMG can't be Gatekeeper-verified offline. This submits the DMG
// container itself to the notary and staples the returned ticket to it.
//
// Used by `npm run release:mac` (build then staple) and by the CI release
// workflow (.github/workflows/release.yml), so stapling has a single source of
// truth. No-op off macOS and when notary credentials are absent, so it never
// slows or breaks ordinary/dev builds (mirrors the env-gated codesign in
// build-sidecar.mjs).
//
// Credentials = App Store Connect API key (the same env Tauri uses to notarize):
//   APPLE_API_ISSUER     issuer UUID
//   APPLE_API_KEY        key id
//   APPLE_API_KEY_PATH   path to the .p8 key file
import { execFileSync } from "node:child_process";
import { existsSync, readdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import process from "node:process";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const desktopDir = resolve(scriptDir, ".."); // apps/desktop
const repoRoot = resolve(desktopDir, "..", ".."); // repository root

if (process.platform !== "darwin") {
  console.log("staple-dmg: not macOS — nothing to do");
  process.exit(0);
}

const { APPLE_API_ISSUER, APPLE_API_KEY, APPLE_API_KEY_PATH } = process.env;
if (!APPLE_API_ISSUER || !APPLE_API_KEY || !APPLE_API_KEY_PATH) {
  console.log(
    "staple-dmg: notary credentials not set " +
      "(APPLE_API_ISSUER / APPLE_API_KEY / APPLE_API_KEY_PATH) — skipping",
  );
  process.exit(0);
}

// Locate the built DMG(s) under the Tauri bundle dir (normally one per build).
const dmgDir = join(repoRoot, "target", "release", "bundle", "dmg");
const dmgs = existsSync(dmgDir)
  ? readdirSync(dmgDir).filter((f) => f.endsWith(".dmg"))
  : [];
if (dmgs.length === 0) {
  console.error(
    `staple-dmg: no .dmg found in ${dmgDir} — run a release build first`,
  );
  process.exit(1);
}
if (dmgs.length > 1) {
  console.warn(
    `staple-dmg: multiple DMGs present, stapling all: ${dmgs.join(", ")}`,
  );
}

function xcrun(args) {
  console.log(`> xcrun ${args.join(" ")}`);
  execFileSync("xcrun", args, { stdio: "inherit" });
}

for (const name of dmgs) {
  const dmg = join(dmgDir, name);
  xcrun([
    "notarytool",
    "submit",
    dmg,
    "--key",
    APPLE_API_KEY_PATH,
    "--key-id",
    APPLE_API_KEY,
    "--issuer",
    APPLE_API_ISSUER,
    "--wait",
  ]);
  xcrun(["stapler", "staple", dmg]);
  xcrun(["stapler", "validate", dmg]);
  console.log(`staple-dmg: stapled ${dmg}`);
}
