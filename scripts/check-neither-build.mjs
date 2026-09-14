#!/usr/bin/env node
// sc-10463: reproduce the CI "parity" lane's Rust build locally — Linux with default features and
// NO `backend-candle` (the "neither backend" config) — so the cfg/dead-code trap that only surfaces
// there is caught before pushing, not after a full CI round-trip.
//
// The trap: `sceneworks-worker`'s generation harness (`src/image_jobs/base.rs` and the code it pulls
// in) is `include!`d only under
//     any(target_os = "macos", all(not(target_os = "macos"), feature = "backend-candle"))
// so on the neither build (`not(macos)` + candle off, which is the default) that whole harness is
// absent. Anything used *only* by it — a `use`, a struct, a helper fn — is then dead, and
// `cargo clippy -- -D warnings` fails ("never constructed" / "unused import"). This has bitten
// sc-10404 (`PhaseTimer`) and sc-8390 (`run_blocking_with_heartbeat`). See CONTRIBUTING.md
// ("The base.rs / candle cfg rule") for the fix pattern.
//
// Reproduction is host-dependent because the gate keys off `target_os`, which a native build cannot
// change:
//   * Non-macOS host (Linux, Windows): `target_os != "macos"` already, and candle is off by default,
//     so the plain default-feature clippy below IS the neither build. Run it natively — no Docker,
//     no extra toolchain.
//   * macOS host: `target_os` is pinned to "macos", so a native clippy ALWAYS compiles `base.rs` and
//     can never see the trap. Run the identical clippy inside a Linux `rust` container instead.
//
// Either way the lint is `cargo clippy -p sceneworks-worker -p sceneworks-rust-api --all-targets --
// -D warnings` — the parity lane's clippy, scoped to the two crates that carry the gated code
// (`sceneworks-worker` holds the whole trap class; `sceneworks-rust-api` is included as belt-and-
// suspenders since it links the same contract types).
//
// DISK (macOS/Docker path only): the target volume is keyed by checkout, so every worktree that has
// ever run this check owns a 30-42 GB Docker volume, and nothing prunes them — including the
// `sceneworks-neither-target` from before the keying and the volumes of worktrees that are gone.
// Each run prints the volume it is about to use; `--prune` removes every OTHER
// `sceneworks-neither-target*` volume (never this checkout's, never the shared toolchain/registry/
// git caches) and exits:
//
//   node scripts/check-neither-build.mjs --prune        # reclaim, then exit
//   node scripts/check-neither-build.mjs --prune --run  # reclaim, then run the check
//
// A volume in use by a running container is reported as skipped rather than force-removed.

import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { fileURLToPath } from "node:url";
import path from "node:path";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

// The target volume is keyed by the checkout it builds, because EVERY checkout mounts at the same
// container path `/workspace`. A single shared target volume therefore makes cargo's fingerprints
// collide across worktrees: whichever checkout built last leaves artifacts the next one reuses, and
// the check reports errors from source it is not looking at ("no field `x` on type `y`" for a field
// that is plainly there). The toolchain, registry and VCS volumes stay shared — those are keyed by
// content, not by checkout, and they are the expensive ones.
const TARGET_VOLUME = `sceneworks-neither-target-${createHash("sha256")
  .update(repoRoot)
  .digest("hex")
  .slice(0, 12)}`;

// The parity lane's clippy, scoped to the crates that carry the macOS/candle-gated code.
const CLIPPY_ARGS = [
  "clippy",
  "-p",
  "sceneworks-worker",
  "-p",
  "sceneworks-rust-api",
  "--all-targets",
  "--",
  "-D",
  "warnings",
];

// Official Debian-based Rust image (buildpack-deps lineage → ships gcc/pkg-config/libssl the native
// worker deps need). It uses the *minimal* rustup profile, so clippy must be added in-container
// (CI does the same via `components: clippy`). Overridable for a pinned/mirrored tag.
const RUST_IMAGE = process.env.SCENEWORKS_NEITHER_IMAGE || "rust:bookworm";

function hasDocker() {
  const res = spawnSync("docker", ["--version"], { stdio: "ignore" });
  return !res.error && res.status === 0;
}

function runNative() {
  console.log(
    `[neither] host is ${process.platform} (not macOS): the default-feature build already excludes\n` +
      "          base.rs, so this native clippy IS the parity 'neither' build.\n" +
      `          cargo ${CLIPPY_ARGS.join(" ")}\n`,
  );
  const res = spawnSync("cargo", CLIPPY_ARGS, { stdio: "inherit", cwd: repoRoot });
  if (res.error) {
    if (res.error.code === "ENOENT") {
      console.error("[neither] cargo not found on PATH. Install Rust (https://rustup.rs) and retry.");
      return 1;
    }
    throw res.error;
  }
  return res.status ?? 1;
}

function runDocker() {
  if (!hasDocker()) {
    console.error(
      "[neither] macOS host: a native clippy always compiles the macOS-only base.rs, so it CANNOT\n" +
        "          reproduce the Linux 'neither' build. This check needs Docker here.\n" +
        "          • Install Docker Desktop, or\n" +
        "          • run this on a Linux or Windows box, where `npm run rust:check` already is the\n" +
        "            neither build.\n" +
        "          (SCENEWORKS_NEITHER_IMAGE overrides the container image.)",
    );
    return 1;
  }
  // Named volumes prepopulate from the image on first use and then cache the toolchain, crate
  // registry, GIT DEPENDENCY DATABASE, and target dir across runs, so only the first invocation is
  // cold.
  //
  // `/usr/local/cargo/git` is not an optional extra here. Every one of this workspace's inference
  // dependencies is a `git = "…/SceneWorks/inference"` pin, and that repository's bare database is
  // ~900 MB. Without this volume it landed in the container's throwaway writable layer, so EVERY
  // run re-cloned the whole thing over the network before compiling a line — observed at >45
  // minutes and still fetching, which is long enough that the check reads as hung and gets skipped.
  // Cached, the clone is paid once.
  const dockerArgs = [
    "run",
    "--rm",
    "-v",
    `${repoRoot}:/workspace`,
    "-w",
    "/workspace",
    "-v",
    "sceneworks-neither-rustup:/usr/local/rustup",
    "-v",
    "sceneworks-neither-registry:/usr/local/cargo/registry",
    "-v",
    "sceneworks-neither-git:/usr/local/cargo/git",
    "-v",
    `${TARGET_VOLUME}:/workspace/target`,
    RUST_IMAGE,
    "bash",
    "-euc",
    // rust-toolchain.toml pins a concrete version; add clippy to it (no-op if already present) then lint.
    `rustup component add clippy && exec cargo ${CLIPPY_ARGS.join(" ")}`,
  ];
  console.log(
    `[neither] macOS host: reproducing the Linux 'neither' build in ${RUST_IMAGE}.\n` +
      `          target volume: ${TARGET_VOLUME} (this checkout's; 30-42 GB once warm —\n` +
      `          'node scripts/check-neither-build.mjs --prune' removes the others)\n` +
      `          docker ${dockerArgs.join(" ")}\n`,
  );
  const res = spawnSync("docker", dockerArgs, { stdio: "inherit", cwd: repoRoot });
  if (res.error) throw res.error;
  return res.status ?? 1;
}

/// Remove every `sceneworks-neither-target*` volume EXCEPT this checkout's: the orphans of deleted
/// worktrees, and the unkeyed `sceneworks-neither-target` from before the per-checkout keying. The
/// shared toolchain/registry/git volumes are never touched — they are keyed by content, they are
/// small next to a target dir, and re-cloning the ~900 MB inference database is the slow path this
/// script exists to avoid.
function pruneTargetVolumes() {
  if (!hasDocker()) {
    console.error("[neither] --prune needs Docker (nothing to prune without it).");
    return 1;
  }
  const listed = spawnSync("docker", ["volume", "ls", "--format", "{{.Name}}"], {
    encoding: "utf8",
  });
  if (listed.error || listed.status !== 0) {
    console.error(`[neither] could not list Docker volumes: ${listed.stderr || listed.error}`);
    return 1;
  }
  const orphans = listed.stdout
    .split("\n")
    .map((name) => name.trim())
    .filter((name) => name.startsWith("sceneworks-neither-target") && name !== TARGET_VOLUME);
  console.log(`[neither] keeping this checkout's volume: ${TARGET_VOLUME}`);
  if (orphans.length === 0) {
    console.log("[neither] no other sceneworks-neither-target* volumes to prune.");
    return 0;
  }
  for (const volume of orphans) {
    const removed = spawnSync("docker", ["volume", "rm", volume], { encoding: "utf8" });
    if (removed.status === 0) {
      console.log(`[neither] removed ${volume}`);
    } else {
      // In use by a running container, or already gone. Never force and never fail the prune: a
      // volume a live build is writing to is not this script's to destroy, and leaving it is the
      // correct outcome rather than an error.
      console.log(`[neither] skipped ${volume}: ${(removed.stderr || "").trim()}`);
    }
  }
  return 0;
}

if (process.argv.includes("--help") || process.argv.includes("-h")) {
  console.log(
    "Reproduce the CI 'parity' (Linux, no backend-candle) Rust clippy locally.\n\n" +
      "  node scripts/check-neither-build.mjs          # or: npm run rust:check:neither\n" +
      "  node scripts/check-neither-build.mjs --prune  # remove OTHER checkouts' target volumes\n" +
      "  node scripts/check-neither-build.mjs --prune --run   # prune, then run the check\n\n" +
      "Non-macOS hosts run it natively; macOS hosts run it in a Linux Docker container, whose\n" +
      "per-checkout target volume (printed on every run) is 30-42 GB once warm.\n" +
      "Env: SCENEWORKS_NEITHER_IMAGE overrides the container image (default rust:bookworm).",
  );
  process.exit(0);
}

if (process.argv.includes("--prune")) {
  const pruned = pruneTargetVolumes();
  if (pruned !== 0 || !process.argv.includes("--run")) {
    process.exit(pruned);
  }
}

process.exit(process.platform === "darwin" ? runDocker() : runNative());
