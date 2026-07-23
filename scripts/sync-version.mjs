#!/usr/bin/env node
// Propagate the version to every file Tauri + npm read, so one `npm version` at
// the repo root bumps the whole product atomically. Wired as the root
// package.json `version` lifecycle script: npm bumps the root package.json first,
// then runs this (before the version commit), so the synced files land in the
// SAME commit + tag — the git tag can never drift from the shipped version.
//
//   npm version 0.3.0        # bump everything -> one commit "0.3.0" -> tag v0.3.0
//   git push --follow-tags   # triggers .github/workflows/release.yml
//
// tauri.conf.json's `version` names the bundled .app / DMG (the artifact-critical
// field). The sub-package versions are bumped with `npm version` so each
// package.json AND its package-lock.json stay in lockstep — a mismatch can fail
// the `npm ci` the release workflow runs.
import { execFileSync } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");

// `npm version` already wrote the new version into the root package.json; mirror it.
const version = JSON.parse(
  readFileSync(join(repoRoot, "package.json"), "utf8"),
).version;
if (!version) {
  console.error("sync-version: no version in root package.json");
  process.exit(1);
}

// Sub-packages with lockfiles: use `npm version` so package.json AND
// package-lock.json move together. --allow-same-version makes an already-aligned
// package a no-op (idempotent re-runs).
for (const dir of ["apps/desktop", "apps/web"]) {
  execFileSync(
    "npm",
    ["version", version, "--no-git-tag-version", "--allow-same-version"],
    { cwd: join(repoRoot, dir), stdio: "inherit" },
  );
}

// tauri.conf.json has no lockfile — surgical replace of the first `"version": "…"`
// to preserve exact formatting (no JSON reparse/reformat churn).
const conf = join(repoRoot, "apps", "desktop", "tauri.conf.json");
const before = readFileSync(conf, "utf8");
const tauriVersionKey = /"version":\s*"[^"]*"/;
if (!tauriVersionKey.test(before)) {
  console.error('sync-version: no "version" field found in tauri.conf.json');
  process.exit(1);
}
// Distinguish "field missing" (fatal, above) from "already at target" (idempotent
// no-op): only write when the value actually changes so re-runs don't error or churn.
const after = before.replace(tauriVersionKey, `"version": "${version}"`);
if (after !== before) {
  writeFileSync(conf, after);
}

// Root Cargo.toml [workspace.package] version — the single source every local
// crate inherits via `version.workspace = true`. It feeds logs, /health payloads,
// and the project-file appVersion, so it must track the product version too
// (sc-13613: it had drifted to 0.2.0 while the product shipped 0.8.0). Rewrite
// ONLY the [workspace.package] table's `version` key: bound the edit to that
// section so we never touch a member-crate `version.workspace = true` line, a
// [workspace.dependencies] pin, or the [patch] rev. Idempotent — an already
// aligned table is left untouched (no spurious write on re-runs).
const cargoToml = join(repoRoot, "Cargo.toml");
const cargoBefore = readFileSync(cargoToml, "utf8");
const cargoHeader = "[workspace.package]";
const cargoSectionStart = cargoBefore.indexOf(cargoHeader);
if (cargoSectionStart === -1) {
  console.error("sync-version: no [workspace.package] table found in Cargo.toml");
  process.exit(1);
}
const cargoAfterHeader = cargoSectionStart + cargoHeader.length;
const cargoNextHeaderRel = cargoBefore.slice(cargoAfterHeader).search(/\n\[/);
const cargoSectionEnd =
  cargoNextHeaderRel === -1 ? cargoBefore.length : cargoAfterHeader + cargoNextHeaderRel;
const cargoSection = cargoBefore.slice(cargoSectionStart, cargoSectionEnd);
const cargoVersionKey = /^version\s*=\s*"[^"]*"/m;
if (!cargoVersionKey.test(cargoSection)) {
  console.error("sync-version: no version key in Cargo.toml [workspace.package]");
  process.exit(1);
}
const cargoPatchedSection = cargoSection.replace(cargoVersionKey, `version = "${version}"`);
if (cargoPatchedSection !== cargoSection) {
  writeFileSync(
    cargoToml,
    cargoBefore.slice(0, cargoSectionStart) +
      cargoPatchedSection +
      cargoBefore.slice(cargoSectionEnd),
  );

  // Cargo.lock records each workspace member's resolved version (they inherit
  // `version.workspace = true`), so the rewrite above leaves the lockfile stale.
  // A stale lock hard-fails the `cargo fetch --locked` that ~8 CI lanes run
  // (parity / release / candle / windows), which would defeat the atomic-bump
  // guarantee on the very first real `npm version`. Refresh ONLY the workspace
  // members, offline (no registry access, no external-dep churn), in this same
  // run so the bump commit carries a consistent lock. Gated on the version
  // actually changing: an already-aligned re-run does no cargo work and needs no
  // toolchain (idempotency is preserved on cargo-less machines).
  try {
    execFileSync("cargo", ["update", "--workspace", "--offline"], {
      cwd: repoRoot,
      stdio: "inherit",
    });
  } catch (error) {
    if (error?.code === "ENOENT") {
      console.error(
        "sync-version: bumped Cargo.toml but `cargo` is not on PATH, so Cargo.lock " +
          "is now STALE. Install the Rust toolchain and run " +
          "`cargo update --workspace --offline` (then commit Cargo.lock) — otherwise " +
          "CI's `cargo fetch --locked` will fail on the version skew.",
      );
    } else {
      console.error(
        `sync-version: \`cargo update --workspace --offline\` failed to refresh ` +
          `Cargo.lock: ${error?.message ?? error}`,
      );
    }
    process.exit(1);
  }
}

console.log(
  `sync-version: apps/desktop + apps/web + tauri.conf.json + Cargo.toml (+ Cargo.lock) synced to ${version}`,
);
