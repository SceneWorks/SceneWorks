#!/usr/bin/env node
// Builds the sceneworks-rust-api binary (with the embedded web UI) and stages it
// as a Tauri sidecar named for the host target triple. Wired as the
// tauri.conf.json `beforeBuildCommand` so `tauri build` is self-contained.
import { execFileSync } from "node:child_process";
import {
  copyFileSync,
  mkdirSync,
  chmodSync,
  writeFileSync,
  readFileSync,
  existsSync,
  readdirSync,
  statSync,
} from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import process from "node:process";
import os from "node:os";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const desktopDir = resolve(scriptDir, ".."); // apps/desktop
const repoRoot = resolve(desktopDir, "..", ".."); // repository root
const npmCmd = process.platform === "win32" ? "npm.cmd" : "npm";

function run(cmd, args, extraEnv = {}) {
  console.log(`> ${cmd} ${args.join(" ")}`);
  // execFileSync (argv), NOT a joined execSync shell string (F-127, sc-8929): a
  // `PYTHON` path containing spaces (e.g. "C:\\Program Files\\Python\\python.exe")
  // would break when re-split by the shell. Each arg is passed as a discrete argv
  // entry, so no quoting is needed and spaces are safe. (`shell:true` is deliberately
  // NOT used — under it Node concatenates args WITHOUT escaping (DEP0190), which would
  // reintroduce the very space-splitting this fixes.)
  const opts = {
    stdio: "inherit",
    cwd: repoRoot,
    env: { ...process.env, ...extraEnv },
  };
  // Windows can't exec a batch shim (npm.cmd) directly post-CVE-2024-27980; route it
  // through cmd.exe with each token discretely quoted so spaces stay intact. Real
  // executables (python.exe, rustc.exe) spawn directly via argv on every platform.
  if (process.platform === "win32" && /\.(cmd|bat)$/i.test(cmd)) {
    const quote = (s) => (/[\s"]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s);
    const line = [cmd, ...args].map(quote).join(" ");
    execFileSync(process.env.ComSpec || "cmd.exe", ["/d", "/s", "/c", line], opts);
    return;
  }
  execFileSync(cmd, args, opts);
}

// Locate the mlx.metallib produced by the api build's pmetal-mlx-sys compile
// (sc-10349). Prefer the build tree — the exact, freshly-built artifact matching
// the api binary we just staged. There can be several pmetal-mlx-sys-<hash> build
// dirs (feature/profile variants), so pick the newest that actually holds the file.
// Fall back to ~/.cache/pmetal/lib/mlx.metallib, which mlx-sys build.rs refreshes
// (newest-wins) after every build. Returns null if neither exists (caller errors).
function findBuiltMetallib() {
  const buildRoot = join(repoRoot, "target", "release", "build");
  const candidates = [];
  if (existsSync(buildRoot)) {
    for (const entry of readdirSync(buildRoot)) {
      if (!entry.startsWith("pmetal-mlx-sys-")) continue;
      const lib = join(buildRoot, entry, "out", "build", "lib", "mlx.metallib");
      if (existsSync(lib)) candidates.push([lib, statSync(lib).mtimeMs]);
    }
  }
  if (candidates.length) {
    candidates.sort((a, b) => b[1] - a[1]);
    return candidates[0][0];
  }
  const cached = join(os.homedir(), ".cache", "pmetal", "lib", "mlx.metallib");
  return existsSync(cached) ? cached : null;
}

// Packaging-time regression guard for the quant/moe kernel fatbin (sc-7544 / sc-13510).
// candle-kernels' GGUF quant/moe kernels are a static `libmoe.a` of SASS (no PTX): built
// un-patched at the cap=80 baseline it holds an sm_80-only cubin, and on Blackwell every
// quantized matmul silently returns zeros (models render solid black — nothing else fails).
// The multi-arch fatbin comes from the root Cargo.toml [patch] onto the inference repo's
// vendored candle-kernels; that patch was silently lost once already (sc-13510), so trusting
// the build graph is not enough — inspect the artifact we are about to package. Requires
// `cuobjdump`, which ships in the same CUDA toolkit bin dir as the nvcc this build just ran.
function verifyCandleQuantFatbin(computeCap) {
  const buildRoot = join(repoRoot, "target", "release", "build");
  const candidates = [];
  if (existsSync(buildRoot)) {
    for (const entry of readdirSync(buildRoot)) {
      if (!entry.startsWith("candle-kernels-")) continue;
      const lib = join(buildRoot, entry, "out", "libmoe.a");
      if (existsSync(lib)) candidates.push([lib, statSync(lib).mtimeMs]);
    }
  }
  if (!candidates.length) {
    console.error(
      "build-sidecar: candle build left no candle-kernels-*/out/libmoe.a under " +
        `${buildRoot} — cannot verify the quant kernel fatbin`,
    );
    process.exit(1);
  }
  candidates.sort((a, b) => b[1] - a[1]);
  const lib = candidates[0][0];
  const dump = (flag) => {
    try {
      return execFileSync("cuobjdump", [flag, lib], { encoding: "utf8" });
    } catch (err) {
      console.error(
        `build-sidecar: cuobjdump ${flag} failed (${err?.message ?? err}) — it ships in the ` +
          `same CUDA toolkit bin dir as the nvcc this build just used; ensure that dir is on PATH`,
      );
      process.exit(1);
    }
  };
  const elf = dump("--list-elf");
  const ptx = dump("--list-ptx");
  // The vendored build.rs always adds sm_90 + sm_120 SASS and compute_120 PTX on top of
  // cudaforge's one CUDA_COMPUTE_CAP-derived baseline arch (sm_80 at the packaging default).
  // cuobjdump lists cubins as `libmoe.N.sm_XX.cubin`; the compute_120 PTX entry is listed
  // as `libmoe.N.sm_120.ptx` (cuobjdump names PTX by target arch, not virtual arch).
  const required = [...new Set([`sm_${computeCap}`, "sm_90", "sm_120"])];
  const missing = required.filter((arch) => !elf.includes(`${arch}.cubin`));
  const problems = [];
  if (missing.length) {
    problems.push(`libmoe.a lacks ${missing.join(", ")} SASS (has:\n${elf.trim()})`);
  }
  if (!ptx.includes("sm_120.ptx")) {
    problems.push(`libmoe.a lacks compute_120 PTX (forward-JIT for post-Blackwell archs)`);
  }
  if (problems.length) {
    console.error(
      `build-sidecar: ${lib} is not the multi-arch quant fatbin — quantized models would ` +
        `silently render black on unsupported GPU architectures (sc-7544 / sc-13510):\n` +
        problems.map((p) => `  - ${p}`).join("\n") +
        `\nThe root Cargo.toml [patch] onto the inference repo's vendored candle-kernels is ` +
        `probably not in effect; see crates/sceneworks-worker/tests/candle_kernels_patch_guard.rs.`,
    );
    process.exit(1);
  }
  console.log(
    `build-sidecar: quant kernel fatbin verified (${required.join("+")} SASS + compute_120 PTX) at ${lib}`,
  );
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

// Sign a nested Mach-O for notarization. Tauri signs the .app and the externalBin
// sidecar (sceneworks-api), but NOT the extra binaries we drop into the
// bundle's Resources/ (the static ffmpeg, the onnxruntime dylib). Apple's notary
// service rejects any nested binary that lacks a Developer ID signature, a secure
// timestamp, or (for executables) hardened runtime — so sign them inside-out here,
// before Tauri seals the bundle. No-op unless an identity is configured (the same
// identity Tauri uses for the .app), so plain dev builds are unchanged. The
// identity comes from the APPLE_SIGNING_IDENTITY env var (CI/headless) OR, as a
// fallback, bundle.macOS.signingIdentity in tauri.conf.json — because
// beforeBuildCommand runs before Tauri signs and does NOT inherit the conf value
// as an env var, so a local `tauri build` that sets the identity only in the conf
// would otherwise skip pre-signing and fail notarization on the nested binaries.
// execFileSync (not the shell `run` above) so the identity's spaces/parens in
// "Developer ID Application: Name (TEAMID)" don't need quoting.
function readConfSigningIdentity() {
  try {
    const conf = JSON.parse(
      readFileSync(join(desktopDir, "tauri.conf.json"), "utf8"),
    );
    return conf?.bundle?.macOS?.signingIdentity || "";
  } catch {
    return "";
  }
}
const signingIdentity =
  process.env.APPLE_SIGNING_IDENTITY || readConfSigningIdentity();
function codesignForNotarization(file) {
  if (!signingIdentity || !triple.includes("apple-darwin")) return;
  console.log(`> codesign --force --options runtime --timestamp "${file}"`);
  execFileSync(
    "codesign",
    ["--force", "--sign", signingIdentity, "--options", "runtime", "--timestamp", file],
    { stdio: "inherit" },
  );
  console.log(`build-sidecar: codesigned ${file} for notarization`);
}

// Build the web bundle + API binary with the embedded UI (single source of
// truth for the embedded build). Empty VITE_API_BASE_URL makes the embedded UI
// talk to its own origin (the API serves it), so it works on the dynamic port
// with no CORS.
//
// Candle (Windows/CUDA) backend — the DEFAULT for the Windows desktop (sc-5559 /
// sc-5563): compile the sidecar with `--features embed-web,backend-candle` so the
// desktop's Rust worker runs candle. Off-Mac is CUDA-only (product decision: no
// CPU/AMD, and the Python torch worker was retired in Phase 7), so a plain Windows
// build is a GPU-less shell with no inference backend at all — never what we ship.
// Building it therefore REQUIRES the CUDA Toolkit 12.9 + VS2022 BuildTools MSVC
// 14.44 on PATH (run from its vcvars64 — CUDA 12.9 rejects VS2026's 14.51); the
// candle build aborts with a clear error otherwise. Opt OUT with
// SCENEWORKS_DESKTOP_CANDLE=0 only for a deliberately GPU-less compile/packaging
// check on a box without the CUDA toolkit (e.g. a fast windows-latest CI lane).
// macOS is unaffected — it bakes MLX into the api binary and never builds candle.
const candle =
  process.platform === "win32" && process.env.SCENEWORKS_DESKTOP_CANDLE !== "0";
if (candle) {
  // CUDA_COMPUTE_CAP=80 builds `compute_80` PTX the driver JITs forward to sm_120
  // (Blackwell) — one binary covers Ampere→Blackwell (per sc-3676). That claim holds
  // for candle-kernels' dense build_ptx() kernels only: the GGUF quant/moe kernels are
  // a static SASS `libmoe.a` with NO PTX, which at cap=80 alone is Ampere-only and
  // silently zeros every quantized matmul on Blackwell (sc-7544). Multi-arch coverage
  // for those comes from the root Cargo.toml [patch] onto the inference repo's
  // vendored candle-kernels (sm_80+sm_90+sm_120 SASS + compute_120 PTX), verified
  // below after the build (sc-13510 — the patch already dropped out silently once).
  // Honor an explicit override (e.g. a native-cap dev build; the vendored extra gencodes
  // are unconditional, so the quant fatbin stays multi-arch at any cap) if the env set it.
  const candleEnv = { VITE_API_BASE_URL: "" };
  if (!process.env.CUDA_COMPUTE_CAP) {
    candleEnv.CUDA_COMPUTE_CAP = "80";
  }
  console.log(
    `build-sidecar: candle backend ON (CUDA_COMPUTE_CAP=${process.env.CUDA_COMPUTE_CAP ?? "80"})`,
  );
  // candle-kernels compiles its CUDA kernels with `cudaforge`, which fans out one
  // nvcc per .cu over a rayon pool sized to ~half the CPU count. On a high-core
  // Windows box (e.g. 48 logical CPUs → ~24 parallel nvcc, each spawning
  // cl/cicc/ptxas) that process-spawn storm — concurrent with the rest of the
  // cargo build — INTERMITTENTLY gets an nvcc child killed with NO output. It
  // surfaces as `CompilationFailed { path: "...affine.cu", message: "nvcc error:\n\n" }`
  // (empty on both streams) — a teardown symptom, NOT a real compile error: every
  // kernel compiles cleanly for compute_80 AND sm_120a in isolation, and the build
  // succeeds at low concurrency (CI's 4-core runner, cache hits). Two mitigations:
  //   1) cap cudaforge's nvcc fan-out (CUDAFORGE_THREADS) so the storm is bounded
  //      regardless of core count (honored by cudaforge's ParallelConfig).
  //   2) if it still fails, retry ONCE fully serial (CUDAFORGE_THREADS=1): the
  //      serial recompile only redoes the kernels the killed run left missing, and
  //      it both clears the spawn contention AND — were the failure ever a genuine
  //      per-file nvcc error — surfaces the true (non-empty) message instead of the
  //      swallowed empty one. NOTE: CUDAFORGE_THREADS is not in candle-kernels'
  //      rerun-if-env-changed set, so toggling it never forces a needless rebuild.
  const nvccCap = String(Math.min(8, Math.max(1, os.cpus().length)));
  try {
    run(npmCmd, ["run", "api:build:embedded:candle"], {
      ...candleEnv,
      CUDAFORGE_THREADS: nvccCap,
    });
  } catch (err) {
    console.warn(
      `build-sidecar: candle build failed (${err?.message ?? err}); retrying once ` +
        `with serial CUDA kernel compilation (CUDAFORGE_THREADS=1) to clear nvcc ` +
        `spawn contention / surface the real nvcc error`,
    );
    run(npmCmd, ["run", "api:build:embedded:candle"], {
      ...candleEnv,
      CUDAFORGE_THREADS: "1",
    });
  }
  verifyCandleQuantFatbin(process.env.CUDA_COMPUTE_CAP || candleEnv.CUDA_COMPUTE_CAP);
} else {
  run(npmCmd, ["run", "api:build:embedded"], { VITE_API_BASE_URL: "" });
}

const src = join(repoRoot, "target", "release", `sceneworks-rust-api${exe}`);
const outDir = join(desktopDir, "binaries");
mkdirSync(outDir, { recursive: true });
const dest = join(outDir, `sceneworks-api-${triple}${exe}`);
copyFileSync(src, dest);
if (!exe) {
  chmodSync(dest, 0o755);
}
console.log(`build-sidecar: staged ${dest}`);

// The Rust DWPose detector (sc-3487) dlopens onnxruntime at runtime via
// ORT_DYLIB_PATH (set in setup.rs), bundled as a Tauri resource
// (tauri.conf.json `resources` -> `onnxruntime/**/*`) so a packaged, Python-free
// Mac can still detect poses. The `onnxruntime` dir must exist on EVERY platform —
// Tauri errors on a resource glob that matches no files. Only macOS stages the
// real CoreML dylib (pose detection on the Rust worker is macOS-only); other
// platforms ship a placeholder so the glob matches and the build succeeds.
//
// Windows note: the CUDA-enabled onnxruntime-gpu DLLs are NO LONGER bundled here.
// Together with the CUDA runtime + cuDNN they exceed NSIS's ~2 GB datablock limit
// (`makensis` "mmapping datablock" error), so the Windows candle build downloads them
// on first run into %APPDATA%\SceneWorks\gpu-runtime instead (apps/desktop/src/
// cuda_provision.rs), pointed at by setup.rs. Windows therefore ships only the
// placeholder below — the glob still matches and the install stays small.
const ortDir = join(desktopDir, "onnxruntime");
mkdirSync(ortDir, { recursive: true });
if (triple.includes("apple-darwin")) {
  const dylibDest = join(ortDir, "libonnxruntime.dylib");
  const py = process.env.PYTHON || "python3";
  run(py, ["apps/desktop/scripts/stage-onnxruntime.py", dylibDest]);
  console.log(`build-sidecar: staged ${dylibDest}`);
  codesignForNotarization(dylibDest);
  // onnxruntime is MIT — ship its license text + notice next to the dylib so the
  // MIT "include the copyright + permission notice" requirement is satisfied at
  // the distribution level (mirrors the ffmpeg GPLv3 §6 staging below). Source of
  // truth: apps/desktop/licenses/onnxruntime/ (tracked); also surfaced in the
  // in-app About → Licenses screen (sc-3778).
  for (const name of ["LICENSE", "NOTICE.txt"]) {
    copyFileSync(
      join(desktopDir, "licenses", "onnxruntime", name),
      join(ortDir, name),
    );
  }
  console.log(`build-sidecar: staged onnxruntime MIT license + notice`);
} else {
  writeFileSync(
    join(ortDir, "README.txt"),
    "onnxruntime is bundled on macOS (CoreML) only; the Windows candle build downloads the CUDA onnxruntime on first run into %APPDATA%\\SceneWorks\\gpu-runtime (cuda_provision.rs), not into this resource dir (sc-3487 / sc-5496).\n",
  );
  console.log(`build-sidecar: ${ortDir} placeholder (no bundled onnxruntime)`);
}

// The Rust worker shells out to ffmpeg (frame sampling, frame extract, timeline
// export, video-gen audio mux) via SCENEWORKS_FFMPEG (set in setup.rs). The
// desktop ships no system ffmpeg, and epic 3482 strips the Python venv it used to
// borrow imageio-ffmpeg from — so we bundle a static ffmpeg as a Tauri resource
// (tauri.conf.json `resources` -> `ffmpeg/**/*`). Like the onnxruntime dir above,
// the `ffmpeg` dir must exist on EVERY platform (Tauri errors on an empty glob);
// only macOS stages the real binary (Windows/Linux desktop + server/Docker use
// PATH ffmpeg), other platforms ship a placeholder. GPLv3 — see
// docs/sc-3767/ffmpeg-bundling.md.
const ffmpegDir = join(desktopDir, "ffmpeg");
mkdirSync(ffmpegDir, { recursive: true });
if (triple.includes("apple-darwin")) {
  const ffmpegDest = join(ffmpegDir, "ffmpeg");
  const py = process.env.PYTHON || "python3";
  run(py, ["apps/desktop/scripts/stage-ffmpeg.py", ffmpegDest]);
  console.log(`build-sidecar: staged ${ffmpegDest}`);
  codesignForNotarization(ffmpegDest);
  // The bundled ffmpeg is GPLv3 — ship its license text + written source offer
  // next to the binary so the distribution satisfies GPLv3 §6 (sc-3767). Source
  // of truth: apps/desktop/licenses/ffmpeg/ (tracked).
  for (const name of ["COPYING.GPLv3", "NOTICE.txt"]) {
    copyFileSync(
      join(desktopDir, "licenses", "ffmpeg", name),
      join(ffmpegDir, name),
    );
  }
  console.log(`build-sidecar: staged ffmpeg GPLv3 license + written offer`);
} else {
  writeFileSync(
    join(ffmpegDir, "README.txt"),
    "Static ffmpeg is bundled on macOS only (sc-3767); Windows/Linux use PATH ffmpeg.\n",
  );
  console.log(`build-sidecar: ${ffmpegDir} placeholder (non-macOS, PATH ffmpeg)`);
}

// The in-process MLX worker (macOS) loads MLX's compiled Metal shader library
// (mlx.metallib, ~158 MB) at runtime — it is NOT embedded in the api binary. The
// pmetal-mlx-rs fork's resolver (sc-7898) looks for it via PMETAL_METALLIB_PATH,
// then a path into the *build machine's* target dir baked into the binary, then
// ~/.cache/pmetal — NONE of which exist on a clean end-user Mac (the cache is only
// populated as a side effect of a local `cargo build`). A packaged app that doesn't
// ship the file therefore fails on first MLX use with "Failed to load the default
// metallib. library not found" (sc-10349). Bundle it as a Tauri resource; setup.rs
// points the worker at it via PMETAL_METALLIB_PATH (mirrors the ffmpeg/onnxruntime
// staging above). A .metallib is NOT a Mach-O — it's sealed by the .app signature
// as a plain resource, so unlike the ffmpeg/onnxruntime binaries it needs no
// separate codesign. macOS-only; other platforms use candle and ship no metallib
// (placeholder so the `mlx/**/*` resource glob still matches — Tauri errors on an
// empty glob).
const mlxDir = join(desktopDir, "mlx");
mkdirSync(mlxDir, { recursive: true });
if (triple.includes("apple-darwin")) {
  const metallibSrc = findBuiltMetallib();
  if (!metallibSrc) {
    console.error(
      "build-sidecar: could not locate mlx.metallib under " +
        `${join(repoRoot, "target", "release", "build")}/pmetal-mlx-sys-*/out/build/lib/ ` +
        "or ~/.cache/pmetal/lib/ — a macOS build without it ships an app that fails on " +
        "first MLX use (sc-10349). The api build above compiles pmetal-mlx-sys, which " +
        "produces it, so this usually means that build did not run or was redirected.",
    );
    process.exit(1);
  }
  const metallibDest = join(mlxDir, "mlx.metallib");
  copyFileSync(metallibSrc, metallibDest);
  console.log(`build-sidecar: staged ${metallibDest} (from ${metallibSrc})`);
} else {
  writeFileSync(
    join(mlxDir, "README.txt"),
    "mlx.metallib is bundled on macOS only (the in-process MLX worker). Windows/Linux use the candle backend and ship no MLX shader library (sc-10349).\n",
  );
  console.log(`build-sidecar: ${mlxDir} placeholder (non-macOS, no MLX metallib)`);
}

// The candle (Windows/CUDA) worker links cudarc with dynamic-linking, which
// LoadLibrary's the CUDA runtime redist DLLs by name at runtime, and the worker's
// `ort` paths dlopen a CUDA-enabled onnxruntime. These DLLs are NO LONGER bundled:
// the full CUDA runtime + cuDNN + onnxruntime-gpu set is ~2.7 GB, which exceeds NSIS's
// ~2 GB datablock limit (`makensis` "mmapping datablock" error). The Windows candle
// build now downloads them on first run from pinned PyPI wheels into
// %APPDATA%\SceneWorks\gpu-runtime\{cuda,onnxruntime} (apps/desktop/src/
// cuda_provision.rs); setup.rs resolves the candle worker's PATH + ORT env from there.
// The `cuda` resource dir is no longer produced at all (it's dropped from
// tauri.conf.json `bundle.resources`), so there is nothing to stage here.
