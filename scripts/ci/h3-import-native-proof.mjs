// Explicit synthetic model-level validation only. Ordinary CI may import the pure checks below,
// but must never call runProof. No model download or production model/cache path is used.
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { lockedInferenceRevision } from "../select-windows-cargo-cache.mjs";

export function proofSelection(backend) {
  if (!["mlx", "cuda"].includes(backend)) throw new Error("select mlx or cuda explicitly");
  return ["pruned", "w4a8"].map(format => `minimax_h3_import_fixture::${format}_import_native_${backend}_projection_smoke`);
}

export function verifyProofResult(text, name, status) {
  if (status !== 0 || !text.includes(name)
      || !/^running 1 test$/m.test(text)
      || !/^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; \d+ filtered out;/m.test(text)) {
    throw new Error(`${name}: must execute exactly one passing test, zero failed/ignored; no skip-as-pass`);
  }
}

export function requireManualSelection(event, selection, backend) {
  proofSelection(backend);
  if (event !== "workflow_dispatch" || ![backend, "both"].includes(selection)) {
    throw new Error("H3 model proofs require workflow_dispatch and explicit backend selection");
  }
}

function checked(command, args, options = {}) {
  const result = spawnSync(command, args, { encoding: "utf8", maxBuffer: 64 * 1024 * 1024, ...options });
  if (result.error || result.status !== 0) throw new Error(`${command} failed: ${result.error ?? result.stderr}`);
  return result.stdout;
}

export function runProof(backend, evidence) {
  requireManualSelection(process.env.GITHUB_EVENT_NAME, process.env.H3_IMPORT_BACKEND, backend);
  if (process.platform !== (backend === "mlx" ? "darwin" : "win32")) throw new Error("native backend/host mismatch");
  if (backend === "mlx" && process.env.RUNNER_NAME !== "nax-macos-2") throw new Error("MLX proof is Mac2-only");
  if (backend === "cuda" && process.env.CUDA_VISIBLE_DEVICES !== "0") throw new Error("manual CUDA proof must retain device 0");
  if (!path.isAbsolute(evidence)) throw new Error("evidence directory must be absolute and new");
  mkdirSync(evidence); // refuse accidental reuse of an earlier receipt/log set
  const start = Date.now();
  const fixture = "crates/sceneworks-worker/src/minimax_h3_import_fixture.rs";
  const inferenceSha = lockedInferenceRevision(readFileSync("Cargo.lock", "utf8"));
  const receipt = {
    schema: 1, sourceSha: checked("git", ["rev-parse", "HEAD"]).trim(), inferenceSha,
    event: process.env.GITHUB_EVENT_NAME, selection: process.env.H3_IMPORT_BACKEND,
    runner: process.env.RUNNER_NAME, backend, device: backend === "cuda" ? "CUDA_VISIBLE_DEVICES=0" : "Mac2 Metal",
    fixtureSource: fixture, fixtureSha256: createHash("sha256").update(readFileSync(fixture)).digest("hex"),
    configuration: { syntheticOnly: true, downloads: false, layers: 1, refinerLayers: 1, hiddenSize: 256,
      attentionHeads: 2, headDim: 128, ffnDim: 256, timeEmbedDim: 64, tableGrid: 1025, tableRank: 8,
      testThreads: 1, wallSecondsPerTest: 180, mlxTreeCapGiB: backend === "mlx" ? 8 : null },
    startedAt: new Date(start).toISOString(), status: "running", cases: [],
  };
  const save = () => writeFileSync(path.join(evidence, "receipt.json"), `${JSON.stringify(receipt, null, 2)}\n`);
  save();
  try {
    if (receipt.sourceSha !== process.env.GITHUB_SHA) throw new Error("checkout must equal the dispatched source SHA");
    if (backend === "mlx") {
      const total = Number(checked("sysctl", ["-n", "hw.memsize"]).trim());
      const pressure = checked("/usr/bin/memory_pressure", ["-Q"]);
      writeFileSync(path.join(evidence, "host.txt"), `${checked("system_profiler", ["SPHardwareDataType", "SPDisplaysDataType"])}\n${pressure}`);
      const free = Number(pressure.match(/free percentage:\s*(\d+)%/)?.[1]);
      if (!Number.isFinite(free) || total * free / 100 < 18 * 2 ** 30) throw new Error("Mac2 needs 8 GiB cap + 10 GiB free headroom");
    } else {
      writeFileSync(path.join(evidence, "host.txt"), checked("nvidia-smi", ["--query-gpu=index,uuid,name,driver_version", "--format=csv"]));
    }
    for (const dir of ["hf", "data", "tmp"]) mkdirSync(path.join(evidence, dir));
    const env = { ...process.env, CARGO_NET_OFFLINE: "true", HF_HUB_OFFLINE: "1", TRANSFORMERS_OFFLINE: "1",
      HF_HOME: path.join(evidence, "hf"), HF_HUB_CACHE: path.join(evidence, "hf", "hub"),
      HUGGINGFACE_HUB_CACHE: path.join(evidence, "hf", "hub"), TRANSFORMERS_CACHE: path.join(evidence, "hf", "transformers"),
      SCENEWORKS_DATA_DIR: path.join(evidence, "data"), TMPDIR: path.join(evidence, "tmp"),
      TMP: path.join(evidence, "tmp"), TEMP: path.join(evidence, "tmp") };
    const args = ["test", "--offline", "--locked", "-p", "sceneworks-worker", "--lib", "--no-run", "--message-format=json"];
    if (backend === "cuda") args.push("--features", "backend-candle");
    const buildStart = Date.now();
    const build = spawnSync("cargo", args, { env, encoding: "utf8", maxBuffer: 64 * 1024 * 1024 });
    writeFileSync(path.join(evidence, "build.jsonl"), build.stdout ?? "");
    writeFileSync(path.join(evidence, "build.log"), build.stderr ?? "");
    receipt.build = { command: ["cargo", ...args], status: build.status, durationMs: Date.now() - buildStart };
    if (build.error || build.status !== 0) throw new Error(`proof build failed: ${build.error ?? build.status}`);
    const binaries = (build.stdout ?? "").split("\n").filter(Boolean).map(line => JSON.parse(line))
      .filter(item => item.reason === "compiler-artifact" && item.target?.name === "sceneworks_worker" && item.profile?.test && item.executable)
      .map(item => item.executable);
    if (binaries.length !== 1) throw new Error(`expected one worker libtest executable, got ${binaries.length}`);
    for (const name of proofSelection(backend)) {
      const log = path.join(evidence, `${name.split("::").at(-1)}.log`);
      const testArgs = [name, "--exact", "--ignored", "--nocapture", "--test-threads=1"];
      const testStart = Date.now();
      const result = backend === "mlx"
        ? spawnSync("python3.12", ["scripts/ci/iris-mlx-real-weights/tree_guard.py", "8", "180", log, "--", binaries[0], ...testArgs], { env, encoding: "utf8" })
        : spawnSync(binaries[0], testArgs, { env, encoding: "utf8", timeout: 180_000, killSignal: "SIGTERM" });
      const text = backend === "mlx" ? readFileSync(log, "utf8") : `${result.stdout ?? ""}${result.stderr ?? ""}`;
      if (backend === "cuda") writeFileSync(log, text);
      else writeFileSync(`${log}.guard.txt`, `${result.stdout ?? ""}${result.stderr ?? ""}`);
      receipt.cases.push({ name, command: [binaries[0], ...testArgs], exitCode: result.status, durationMs: Date.now() - testStart, log: path.basename(log) });
      save();
      process.stdout.write(text);
      if (result.error) throw result.error;
      verifyProofResult(text, name, result.status);
    }
    receipt.status = "passed";
  } catch (error) {
    receipt.status = "failed";
    receipt.error = String(error);
    throw error;
  } finally {
    receipt.completedAt = new Date().toISOString();
    receipt.durationMs = Date.now() - start;
    save();
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  runProof(process.argv[2], process.argv[3]);
}
