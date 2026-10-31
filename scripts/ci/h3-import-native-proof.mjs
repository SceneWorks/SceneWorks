// Explicit synthetic model-level validation only. Ordinary CI may import the pure checks below,
// but must never call runProof. No model download or production model/cache path is used.
import { spawn, spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { closeSync, mkdirSync, openSync, readFileSync, writeFileSync } from "node:fs";
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

export function requireMetalHeadroom(observation) {
  const { totalBytes, freePercent } = observation;
  if (!Number.isFinite(totalBytes) || totalBytes <= 0 || !Number.isFinite(freePercent)
      || freePercent < 0 || freePercent > 100 || totalBytes * freePercent / 100 < 18 * 2 ** 30) {
    throw new Error("Mac2 needs 8 GiB cap + 10 GiB free headroom immediately before each case");
  }
}

function metalMemoryObservation() {
  const totalBytes = Number(checked("sysctl", ["-n", "hw.memsize"]).trim());
  const pressure = checked("/usr/bin/memory_pressure", ["-Q"]);
  const freePercent = Number(pressure.match(/free percentage:\s*(\d+)%/)?.[1]);
  return { observedAt: new Date().toISOString(), totalBytes, freePercent,
    availableBytes: totalBytes * freePercent / 100, requiredBytes: 18 * 2 ** 30, pressure };
}

export function observeMetalCase(name, receipt, save, observe = metalMemoryObservation) {
  const observation = { name, ...observe() };
  (receipt.memoryObservations ??= []).push(observation);
  save(); // retain the actual pre-case observation even when it refuses execution
  requireMetalHeadroom(observation);
  return observation;
}

// Physical GPU 0 is the owner's display/work GPU; the self-hosted CUDA runners own physical GPU 1.
// Manual H3 proofs therefore pin the PCI-bus-ordered physical device 1 (CUDA ordinal 0 inside the
// process, because CUDA_VISIBLE_DEVICES narrows visibility to that one card). SCENEWORKS_GPU_ID is
// the worker's physical (nvidia-smi, PCI-ordered) index, so it must name the same card.
export const MANUAL_CUDA_PHYSICAL_GPU = "1";

export function cudaDeviceEvidence(env, csv) {
  if (env.CUDA_DEVICE_ORDER !== "PCI_BUS_ID" || env.CUDA_VISIBLE_DEVICES !== MANUAL_CUDA_PHYSICAL_GPU) {
    throw new Error(`manual CUDA proof requires PCI_BUS_ID ordering and visible physical device ${MANUAL_CUDA_PHYSICAL_GPU}`);
  }
  if (env.SCENEWORKS_GPU_ID !== undefined && env.SCENEWORKS_GPU_ID !== "" && env.SCENEWORKS_GPU_ID !== MANUAL_CUDA_PHYSICAL_GPU) {
    throw new Error(`manual CUDA proof requires SCENEWORKS_GPU_ID ${MANUAL_CUDA_PHYSICAL_GPU} to match the visible physical device`);
  }
  const devices = csv.trim().split(/\r?\n/).filter(Boolean).map(line => {
    const [index, uuid, pciBusId, name, driverVersion] = line.split(",").map(value => value.trim());
    const bus = /^([0-9a-f]+):([0-9a-f]{2}):([0-9a-f]{2})\.([0-7])$/i.exec(pciBusId ?? "");
    if (!/^\d+$/.test(index) || !uuid?.startsWith("GPU-") || !bus || !name || !driverVersion) {
      throw new Error("invalid physical CUDA device identity in nvidia-smi evidence");
    }
    return { nvmlIndex: Number(index), uuid, pciBusId, name, driverVersion,
      busOrder: bus.slice(1).map(part => Number.parseInt(part, 16)) };
  });
  devices.sort((a, b) => {
    for (let i = 0; i < a.busOrder.length; i++) if (a.busOrder[i] !== b.busOrder[i]) return a.busOrder[i] - b.busOrder[i];
    return 0;
  });
  const physical = Number(MANUAL_CUDA_PHYSICAL_GPU);
  if (devices.length <= physical) throw new Error(`missing physical CUDA device ${physical} identity`);
  const { busOrder: _order, ...selectedPhysicalDevice } = devices[physical];
  return { cudaDeviceOrder: env.CUDA_DEVICE_ORDER, cudaVisibleDevices: env.CUDA_VISIBLE_DEVICES,
    selectedPhysicalDevice, enumeration: `PCI_BUS_ID ascending; physical device ${physical} (CUDA visible ordinal 0)` };
}

/// Build-only process supervision: stdout/stderr go directly to durable files, not a Node
/// buffer. The receipt is saved BEFORE launch. A bounded timeout/interruption records failure
/// and terminates only this compiler process tree; no native model runs through this helper.
export async function streamProofBuild(command, args, { env, evidence, receipt, save,
  timeoutMs = 40 * 60 * 1000, graceMs = 10_000 }) {
  const start = Date.now();
  receipt.build = { command: [command, ...args], state: "running", startedAt: new Date(start).toISOString(),
    timeoutMs, stdout: "build.jsonl", stderr: "build.log" };
  save();
  const stdout = openSync(path.join(evidence, "build.jsonl"), "w");
  const stderr = openSync(path.join(evidence, "build.log"), "w");
  let child;
  try { child = spawn(command, args, { env, stdio: ["ignore", stdout, stderr], detached: process.platform !== "win32" }); }
  finally { closeSync(stdout); closeSync(stderr); }
  let reason;
  let escalation;
  const terminate = signal => {
    if (!child.pid) return;
    try {
      if (process.platform === "win32") spawnSync("taskkill", ["/PID", String(child.pid), "/T", ...(signal === "SIGKILL" ? ["/F"] : [])], { stdio: "ignore", timeout: 5000 });
      else process.kill(-child.pid, signal);
    } catch (error) { if (error.code !== "ESRCH") receipt.build.terminationError = String(error); }
  };
  const stop = why => {
    if (reason) return;
    reason = why;
    receipt.build.state = "terminating";
    receipt.build.termination = why;
    receipt.build.terminationRequestedAt = new Date().toISOString();
    save();
    terminate("SIGTERM");
    escalation = setTimeout(() => terminate("SIGKILL"), graceMs);
  };
  const onTerm = () => stop("SIGTERM");
  const onInt = () => stop("SIGINT");
  process.on("SIGTERM", onTerm);
  process.on("SIGINT", onInt);
  const timer = setTimeout(() => stop("timeout"), timeoutMs);
  try {
    const result = await new Promise(resolve => {
      child.once("error", error => resolve({ status: null, error }));
      child.once("close", (status, signal) => resolve({ status, signal }));
    });
    receipt.build = { ...receipt.build, state: reason ? "interrupted" : result.status === 0 ? "passed" : "failed",
      status: result.status, signal: result.signal ?? null, error: result.error ? String(result.error) : null,
      durationMs: Date.now() - start, completedAt: new Date().toISOString() };
    save();
    if (reason || result.error || result.status !== 0) throw new Error(`proof build failed: ${reason ?? result.error ?? result.status}`);
    return readFileSync(path.join(evidence, "build.jsonl"), "utf8");
  } finally {
    clearTimeout(timer);
    clearTimeout(escalation);
    process.off("SIGTERM", onTerm);
    process.off("SIGINT", onInt);
  }
}

export async function runProof(backend, evidence) {
  requireManualSelection(process.env.GITHUB_EVENT_NAME, process.env.H3_IMPORT_BACKEND, backend);
  if (process.platform !== (backend === "mlx" ? "darwin" : "win32")) throw new Error("native backend/host mismatch");
  if (backend === "mlx" && process.env.RUNNER_NAME !== "nax-macos-2") throw new Error("MLX proof is Mac2-only");
  if (backend === "cuda" && (process.env.CUDA_VISIBLE_DEVICES !== MANUAL_CUDA_PHYSICAL_GPU || process.env.CUDA_DEVICE_ORDER !== "PCI_BUS_ID")) throw new Error(`manual CUDA proof must retain PCI_BUS_ID ordering and physical device ${MANUAL_CUDA_PHYSICAL_GPU}`);
  if (!path.isAbsolute(evidence)) throw new Error("evidence directory must be absolute and new");
  mkdirSync(evidence); // refuse accidental reuse of an earlier receipt/log set
  const start = Date.now();
  const fixture = "crates/sceneworks-worker/src/minimax_h3_import_fixture.rs";
  const inferenceSha = lockedInferenceRevision(readFileSync("Cargo.lock", "utf8"));
  const receipt = {
    schema: 1, sourceSha: checked("git", ["rev-parse", "HEAD"]).trim(), inferenceSha,
    event: process.env.GITHUB_EVENT_NAME, selection: process.env.H3_IMPORT_BACKEND,
    runner: process.env.RUNNER_NAME, backend, device: backend === "cuda" ? `CUDA_VISIBLE_DEVICES=${MANUAL_CUDA_PHYSICAL_GPU}` : "Mac2 Metal",
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
      receipt.initialMemory = metalMemoryObservation();
      writeFileSync(path.join(evidence, "host.txt"), `${checked("system_profiler", ["SPHardwareDataType", "SPDisplaysDataType"])}\n${receipt.initialMemory.pressure}`);
      save();
      requireMetalHeadroom(receipt.initialMemory);
    } else {
      const inventory = checked("nvidia-smi", ["--query-gpu=index,uuid,pci.bus_id,name,driver_version", "--format=csv,noheader,nounits"]);
      writeFileSync(path.join(evidence, "host.txt"), inventory);
      receipt.cuda = cudaDeviceEvidence(process.env, inventory);
      save();
    }
    for (const dir of ["hf", "data", "tmp"]) mkdirSync(path.join(evidence, dir));
    const env = { ...process.env, CARGO_NET_OFFLINE: "true", HF_HUB_OFFLINE: "1", TRANSFORMERS_OFFLINE: "1",
      HF_HOME: path.join(evidence, "hf"), HF_HUB_CACHE: path.join(evidence, "hf", "hub"),
      HUGGINGFACE_HUB_CACHE: path.join(evidence, "hf", "hub"), TRANSFORMERS_CACHE: path.join(evidence, "hf", "transformers"),
      SCENEWORKS_DATA_DIR: path.join(evidence, "data"), TMPDIR: path.join(evidence, "tmp"),
      TMP: path.join(evidence, "tmp"), TEMP: path.join(evidence, "tmp") };
    const args = ["test", "--offline", "--locked", "-p", "sceneworks-worker", "--lib", "--no-run", "--message-format=json"];
    if (backend === "cuda") args.push("--features", "backend-candle");
    const buildOutput = await streamProofBuild("cargo", args, { env, evidence, receipt, save });
    const binaries = buildOutput.split("\n").filter(Boolean).map(line => JSON.parse(line))
      .filter(item => item.reason === "compiler-artifact" && item.target?.name === "sceneworks_worker" && item.profile?.test && item.executable)
      .map(item => item.executable);
    if (binaries.length !== 1) throw new Error(`expected one worker libtest executable, got ${binaries.length}`);
    for (const name of proofSelection(backend)) {
      const log = path.join(evidence, `${name.split("::").at(-1)}.log`);
      const testArgs = [name, "--exact", "--ignored", "--nocapture", "--test-threads=1"];
      const testStart = Date.now();
      if (backend === "mlx") observeMetalCase(name, receipt, save);
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
  await runProof(process.argv[2], process.argv[3]);
}
