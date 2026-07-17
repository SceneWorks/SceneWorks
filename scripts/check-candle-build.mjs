#!/usr/bin/env node
// sc-12356: reproduce the CI `windows-candle` lane's Rust typecheck locally — the
// `all(not(target_os = "macos"), feature = "backend-candle")` configuration — so a candle-only break
// is caught before pushing instead of after a full self-hosted-runner round-trip.
//
// The sibling of `check-neither-build.mjs` (sc-10463), which covers the OTHER lane a macOS host
// cannot see. Between them, all three configurations are reachable locally:
//
//     npm run rust:check          macOS / mlx          (native here)
//     npm run rust:check:neither  Linux, candle OFF    (the parity lane)
//     npm run rust:check:candle   Linux, candle ON     (this script — the windows-candle lane)
//
// ## Why this lane needs its own script
//
// Everything under `#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]` — `vram_gate`,
// `krea_control_fit`, the candle-control image modules, `generate_candle_video` and its Mochi fit
// gates — is compiled by NEITHER a macOS `npm run rust:check` (`target_os` is pinned to `macos`, so
// the cfg is false whatever features you pass) NOR `rust:check:neither` (candle off). Its only
// compiler was `windows-candle.yml`. That lane carries the most cfg-gated code in the repo, so it is
// the one where a local check pays for itself the most — and it was the one with none.
//
// ## Why a plain `cargo clippy --target x86_64-unknown-linux-gnu --features backend-candle` fails
//
// It dies in DEPENDENCY BUILD SCRIPTS, long before typechecking, on a box with no CUDA toolkit:
//   1. `cudarc`         → panics "`nvcc --version` failed".
//   2. `candle-kernels` → `ComputeCapDetectionFailed` (it shells `nvidia-smi`).
//   3. `candle-kernels` → shells `nvcc --ptx` per kernel, then `include_str!`s the `.ptx` it emitted.
//
// `cargo clippy`/`check` NEVER LINK and never run a kernel, so a fake toolkit satisfies all three and
// the typecheck itself is entirely real. This script generates that stub, points `PATH` at it, and
// sets `CUDA_COMPUTE_CAP`. A box WITH a real CUDA toolkit uses it instead — the stub is a fallback,
// not a default.
//
// ## What this does NOT do — read this before trusting a green run
//
// It TYPECHECKS the candle lane. It cannot RUN the candle tests: `cargo test` links, which needs a
// real libcuda. So candle-gated `#[test]`s still execute for the first time on CI, and their
// fixtures/thresholds must be verified by other means (arithmetic, or by exercising the shared
// non-gated helper on a Mac). sc-12306 shipped a candle test whose fixture asserted the exact
// opposite of its intent — it compiled cleanly, so this script would have said nothing.
//
// Green here means "it compiles under the cfg CI compiles it under", nothing more. That is exactly
// the class that used to cost a round-trip (E0425 from a cfg-gated symbol, dead code under
// `-D warnings`), which is why it is worth having.

import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

// The cfg keys off `target_os`, which a native build cannot change — so a macOS host MUST cross to a
// non-macOS target. Linux/Windows hosts already satisfy `not(target_os = "macos")` natively, and
// building for the host avoids needing a cross std at all.
const CROSS_TARGET = process.env.SCENEWORKS_CANDLE_TARGET || "x86_64-unknown-linux-gnu";
const needsCross = process.platform === "darwin";

// The version string `cudarc`'s build.rs parses. It reads stdout LINE INDEX 3, splits on ", " and
// takes [1], then splits on " " and takes [1] — so line 4 must end up as `release 12.9` → `12.9`,
// and 12.9 must be in its `SUPPORTED_CUDA_VERSIONS` table. Keep those in lockstep if cudarc bumps.
const STUB_CUDA_VERSION = "12.9";

// candle-kernels shells `nvidia-smi` to detect the compute cap; with no GPU it fails and tells you to
// set this. Any supported cap works — nothing is executed, only typechecked.
const DEFAULT_COMPUTE_CAP = "90";

/** The CI lane's own commands, mirrored. `sceneworks-worker` has no `default` feature, so
 *  `--features backend-candle` is the whole feature set, exactly as windows-candle.yml passes it. */
const STEPS = [
  {
    label: "clippy (candle worker)",
    args: [
      "clippy",
      "-p",
      "sceneworks-worker",
      "--features",
      "backend-candle",
      "--all-targets",
      "--",
      "-D",
      "warnings",
    ],
  },
  {
    label: "check (candle sidecar: rust-api)",
    // `embed-web` is deliberately absent: it gates `WebAssets` via rust-embed's `debug-embed`, which
    // embeds at COMPILE time even under `cargo check`, and this lane never builds `apps/web/dist`
    // → `error[E0599]: no function get`. CI omits it for the same reason.
    args: ["check", "-p", "sceneworks-rust-api", "--features", "backend-candle"],
  },
];

function run(cmd, args, env) {
  return spawnSync(cmd, args, { cwd: repoRoot, env, encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] });
}

function hasRealNvcc() {
  const res = spawnSync("nvcc", ["--version"], { stdio: "ignore" });
  return !res.error && res.status === 0;
}

function targetInstalled(target) {
  const res = spawnSync("rustup", ["target", "list", "--installed"], { encoding: "utf8" });
  if (res.error || res.status !== 0) return null; // no rustup → can't tell; let cargo speak.
  return res.stdout.split("\n").some((line) => line.trim() === target);
}

/** Write a fake `nvcc` that satisfies the three build scripts above. Only the two behaviours they
 *  actually use are implemented; everything else exits 0 having done nothing, which is correct
 *  because nothing downstream reads it under `cargo check`. */
function writeStubNvcc() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "sceneworks-candle-stub-"));
  const nvcc = path.join(dir, "nvcc");
  fs.writeFileSync(
    nvcc,
    `#!/bin/sh
# Generated by scripts/check-candle-build.mjs (sc-12356). NOT a CUDA compiler: it exists so
# cargo check/clippy can TYPECHECK the candle cfg on a box with no CUDA toolkit. cargo check never
# links and never runs a kernel, so empty .ptx stand-ins are sufficient. CI compiles for real.
for a in "$@"; do
  [ "$a" = "--version" ] && {
    printf 'nvcc: NVIDIA (R) Cuda compiler driver\\n'
    printf 'Copyright (c) 2005-2025 NVIDIA Corporation\\n'
    printf 'Built on stub\\n'
    printf 'Cuda compilation tools, release ${STUB_CUDA_VERSION}, V${STUB_CUDA_VERSION}.0\\n'
    printf 'Build stub\\n'
    exit 0; }
done
# cudaforge invokes: nvcc <gencode> --ptx --default-stream per-thread --output-directory DIR ... X.cu
outdir=""; outfile=""; src=""; prev=""
for a in "$@"; do
  case "$prev" in
    --output-directory) outdir="$a" ;;
    -o) outfile="$a" ;;
  esac
  case "$a" in *.cu) src="$a" ;; esac
  prev="$a"
done
if [ -n "$outfile" ]; then
  mkdir -p "$(dirname "$outfile")" 2>/dev/null; : > "$outfile"
elif [ -n "$outdir" ] && [ -n "$src" ]; then
  base=$(basename "$src" .cu); mkdir -p "$outdir" 2>/dev/null
  : > "$outdir/$base.ptx"; : > "$outdir/$base.o"
fi
exit 0
`,
    { mode: 0o755 },
  );
  return dir;
}

/** The gotcha this function exists for: if a candle-kernels build script ever "succeeded" without
 *  emitting its .ptx (e.g. an earlier run with no stub, or a half-written one), cargo CACHES that
 *  success and replays `couldn't read .../affine.ptx` forever — without re-invoking nvcc. The stub
 *  then looks broken when it is fine. Blow the cached build dirs away and the next run re-runs it. */
function clearCandleKernelsCache() {
  const roots = [path.join(repoRoot, "target"), path.join(repoRoot, "target", CROSS_TARGET)];
  let cleared = 0;
  for (const root of roots) {
    const buildDir = path.join(root, "debug", "build");
    if (!fs.existsSync(buildDir)) continue;
    for (const entry of fs.readdirSync(buildDir)) {
      if (!entry.startsWith("candle-kernels-")) continue;
      fs.rmSync(path.join(buildDir, entry), { recursive: true, force: true });
      cleared += 1;
    }
  }
  return cleared;
}

const STALE_PTX = /couldn't read .*\.ptx/;

function main() {
  if (process.argv.includes("--help") || process.argv.includes("-h")) {
    console.log(
      "Reproduce the CI 'windows-candle' (not-macOS + backend-candle) Rust typecheck locally.\n\n" +
        "  node scripts/check-candle-build.mjs      # or: npm run rust:check:candle\n\n" +
        "macOS hosts cross-compile to a Linux target (the cfg keys off target_os, which a native\n" +
        "build cannot change); Linux/Windows hosts build for the host. With no CUDA toolkit a stub\n" +
        "nvcc is generated — cargo check never links, so the typecheck is still real.\n\n" +
        "This TYPECHECKS only; it cannot RUN candle tests (cargo test links, needing real libcuda).\n\n" +
        "  --allow-skip   exit 0 (with guidance) instead of 1 when the toolchain for this host is\n" +
        "                 missing. For the pre-push hook: a dev who never opted into the cross\n" +
        "                 target should not be blocked from pushing by a check they cannot run.\n\n" +
        "Env: SCENEWORKS_CANDLE_TARGET overrides the cross target (default x86_64-unknown-linux-gnu).",
    );
    return 0;
  }

  // A missing toolchain is a "cannot run here", not "your code is broken" — the caller decides
  // whether that blocks. Interactive runs fail loudly (you asked for the check); the pre-push hook
  // passes --allow-skip so it never turns an un-opted-in dev's push into an error.
  const unavailable = process.argv.includes("--allow-skip") ? 0 : 1;
  const env = { ...process.env };
  const targetArgs = [];

  if (needsCross) {
    const installed = targetInstalled(CROSS_TARGET);
    if (installed === false) {
      const say = unavailable === 0 ? console.log : console.error;
      say(
        `[candle] SKIPPED: the cfg keys off target_os, so a macOS host must cross-compile — but the\n` +
          `         ${CROSS_TARGET} target is not installed. Add it once with:\n\n` +
          `             rustup target add ${CROSS_TARGET}\n\n` +
          `         (SCENEWORKS_CANDLE_TARGET overrides the target.)`,
      );
      return unavailable;
    }
    targetArgs.push("--target", CROSS_TARGET);
    console.log(
      `[candle] macOS host: cross-compiling to ${CROSS_TARGET} — a native build pins\n` +
        `         target_os="macos", so it can never compile this cfg.`,
    );
  } else {
    console.log(
      `[candle] host is ${process.platform} (not macOS): not(target_os = "macos") already holds, so\n` +
        `         this builds for the host.`,
    );
  }

  if (hasRealNvcc()) {
    console.log("[candle] using the real nvcc found on PATH.");
  } else if (process.platform === "win32") {
    const say = unavailable === 0 ? console.log : console.error;
    say(
      "[candle] SKIPPED: no nvcc on PATH. The stub is a POSIX shell script and won't run here; a\n" +
        "         Windows box running this lane is expected to have the real CUDA Toolkit (as the CI\n" +
        "         runner does).",
    );
    return unavailable;
  } else {
    const stubDir = writeStubNvcc();
    env.PATH = `${stubDir}${path.delimiter}${env.PATH}`;
    if (!env.CUDA_COMPUTE_CAP) env.CUDA_COMPUTE_CAP = DEFAULT_COMPUTE_CAP;
    console.log(
      `[candle] no CUDA toolkit here — generated a stub nvcc in ${stubDir}\n` +
        `         (CUDA_COMPUTE_CAP=${env.CUDA_COMPUTE_CAP}). cargo check never links, so the\n` +
        `         typecheck is real; only the kernel objects are fake.`,
    );
  }

  for (const step of STEPS) {
    const args = [step.args[0], ...targetArgs, ...step.args.slice(1)];
    console.log(`\n[candle] ${step.label}: cargo ${args.join(" ")}`);
    let res = run("cargo", args, env);
    if (res.error) {
      if (res.error.code === "ENOENT") {
        console.error("[candle] cargo not found on PATH. Install Rust (https://rustup.rs) and retry.");
        return 1;
      }
      throw res.error;
    }

    // The cached-build-script trap: retry ONCE after clearing, rather than making every run pay for
    // a candle-kernels rebuild it almost never needs.
    if (res.status !== 0 && STALE_PTX.test(res.stderr || "")) {
      const cleared = clearCandleKernelsCache();
      console.log(
        `[candle] stale candle-kernels build cache detected (missing .ptx replayed from a cached\n` +
          `         build-script success). Cleared ${cleared} dir(s); retrying once.`,
      );
      res = run("cargo", args, env);
      if (res.error) throw res.error;
    }

    process.stdout.write(res.stdout || "");
    process.stderr.write(res.stderr || "");
    if (res.status !== 0) {
      console.error(`\n[candle] FAILED: ${step.label}`);
      return res.status ?? 1;
    }
  }

  console.log(
    "\n[candle] OK — the candle cfg typechecks, incl. test targets, under -D warnings.\n" +
      "         Reminder: candle #[test]s still first RUN on the windows-candle CI lane.",
  );
  return 0;
}

process.exit(main());
