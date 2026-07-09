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

import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

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
  // registry, and target dir across runs, so only the first invocation is cold.
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
    "sceneworks-neither-target:/workspace/target",
    RUST_IMAGE,
    "bash",
    "-euc",
    // rust-toolchain.toml pins `stable`; add clippy to it (no-op if already present) then lint.
    `rustup component add clippy && exec cargo ${CLIPPY_ARGS.join(" ")}`,
  ];
  console.log(
    `[neither] macOS host: reproducing the Linux 'neither' build in ${RUST_IMAGE}.\n` +
      `          docker ${dockerArgs.join(" ")}\n`,
  );
  const res = spawnSync("docker", dockerArgs, { stdio: "inherit", cwd: repoRoot });
  if (res.error) throw res.error;
  return res.status ?? 1;
}

if (process.argv.includes("--help") || process.argv.includes("-h")) {
  console.log(
    "Reproduce the CI 'parity' (Linux, no backend-candle) Rust clippy locally.\n\n" +
      "  node scripts/check-neither-build.mjs        # or: npm run rust:check:neither\n\n" +
      "Non-macOS hosts run it natively; macOS hosts run it in a Linux Docker container.\n" +
      "Env: SCENEWORKS_NEITHER_IMAGE overrides the container image (default rust:bookworm).",
  );
  process.exit(0);
}

process.exit(process.platform === "darwin" ? runDocker() : runNative());
