// Cooperative execution ownership only: not memory admission, measurement currency, or a
// universal scheduler. Unwrapped user apps/processes still require operator coordination.
import { spawn, spawnSync } from "node:child_process";
import { fstatSync, statSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

export const METAL_LOCK_PROTOCOL = "sceneworks-host-metal-flock-v1";
export const METAL_LOCK_PATH = "/Users/Shared/SceneWorks/gpu-locks/apple-metal.lock";
export const METAL_LOCK_FD = "SCENEWORKS_METAL_EXECUTION_LOCK_FD";
const helper = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../ci/metal_execution_lock.py");

export function metalExecutionWaitSeconds(env = process.env, existingBudgetSeconds = 1800) {
  // This bounds acquisition WAIT only; it is not a queue-size/submission/model-runtime cap.
  const value = Number(env.SCENEWORKS_METAL_EXECUTION_WAIT_SECONDS ?? existingBudgetSeconds);
  if (!Number.isFinite(value) || value <= 0) throw new Error("Metal acquisition wait must be finite and positive");
  return value;
}

function withDescriptor(options, fd, environment) {
  const original = options.stdio ?? "pipe";
  const stdio = Array.isArray(original) ? [...original] : [original, original, original];
  while (stdio.length < 3) stdio.push("pipe");
  const childFd = stdio.length;
  stdio.push(fd);
  return { ...options, stdio, env: { ...environment, [METAL_LOCK_FD]: String(childFd) } };
}

// Dependency injection is for CPU-only tests. Approved entrypoints use the fixed default
// instance below; there is no CLI/environment namespace override or private-path fallback.
export function createMetalExecutionEntry({
  platform = process.platform, environment = process.env, argv = process.argv,
  python = "python3.12", helperPath = helper, canonicalPath = METAL_LOCK_PATH,
  spawnImpl = spawn, spawnSyncImpl = spawnSync,
  fstat = (fd) => fstatSync(fd, { bigint: true }), stat = (file) => statSync(file, { bigint: true }),
  processObject = process,
} = {}) {
  let verified = null;
  const descriptor = () => {
    const raw = environment[METAL_LOCK_FD];
    if (raw === undefined) return null;
    if (!/^[0-9]+$/.test(raw)) throw new Error("invalid inherited Metal lease descriptor");
    const fd = Number(raw);
    if (!Number.isSafeInteger(fd) || fd < 3) throw new Error("invalid inherited Metal lease descriptor");
    const opened = fstat(fd), named = stat(canonicalPath);
    if (!opened.isFile() || BigInt(opened.nlink) !== 1n || opened.dev !== named.dev || opened.ino !== named.ino) {
      throw new Error("inherited Metal lease lost its canonical inode");
    }
    return { fd, device: opened.dev, inode: opened.ino };
  };
  const validate = (inherited, wait) => {
    // A real kernel-flock check in the identical helper, not trust in an environment marker.
    const options = withDescriptor({ stdio: ["ignore", "ignore", "pipe"], encoding: "utf8", maxBuffer: 65536 }, inherited.fd, environment);
    const result = spawnSyncImpl(python, ["-I", helperPath, "--wait-seconds", String(wait), "--", python, "-I", "-c", "pass"], options);
    if (result.error || result.status !== 0) throw new Error(`inherited Metal execution lease rejected: ${result.error?.message ?? result.stderr ?? result.signal}`);
    const line = String(result.stderr).split(/\r?\n/).find((value) => value.startsWith("METAL_EXECUTION_LEASE "));
    const body = line?.slice("METAL_EXECUTION_LEASE ".length);
    const record = body ? JSON.parse(body) : null;
    const device = body?.match(/"device"\s*:\s*([0-9]+)/)?.[1];
    const inode = body?.match(/"inode"\s*:\s*([0-9]+)/)?.[1];
    if (record?.protocol !== METAL_LOCK_PROTOCOL || record.path !== canonicalPath || record.borrowed !== true
        || device === undefined || inode === undefined
        || BigInt(device) !== BigInt(inherited.device) || BigInt(inode) !== BigInt(inherited.inode)) {
      throw new Error("inherited Metal execution lease has no matching kernel ownership record");
    }
    verified = inherited;
  };
  return {
    async enter(waitSeconds) {
      if (platform !== "darwin") return false;
      const wait = metalExecutionWaitSeconds(environment, waitSeconds);
      const inherited = descriptor();
      if (inherited === null && verified !== null) throw new Error("verified Metal lease descriptor was lost");
      if (inherited !== null) {
        validate(inherited, wait);
        return false;
      }
      const child = spawnImpl(python, ["-I", helperPath, "--wait-seconds", String(wait), "--", ...argv], { env: environment, stdio: "inherit" });
      const forward = (signal) => {
        // Only the owned non-GPU lease supervisor receives this signal; it defers until its tree
        // drains. Never signal/kill an unrelated app or forward directly to a model process.
        try { child.kill(signal); } catch (error) { if (error.code !== "ESRCH") throw error; }
      };
      const interrupt = () => forward("SIGINT"), terminate = () => forward("SIGTERM");
      processObject.on("SIGINT", interrupt); processObject.on("SIGTERM", terminate);
      try {
        processObject.exitCode = await new Promise((resolve, reject) => {
          child.once("error", reject);
          child.once("close", (code, signal) => signal ? reject(new Error(`Metal lease supervisor ended by ${signal}`)) : resolve(code ?? 1));
        });
      } finally {
        processObject.removeListener("SIGINT", interrupt); processObject.removeListener("SIGTERM", terminate);
      }
      return true;
    },
    childOptions(options = {}) {
      if (platform !== "darwin") return options;
      if (environment[METAL_LOCK_FD] === undefined) {
        if (verified !== null) throw new Error("verified Metal lease descriptor was lost");
        return options;
      }
      const current = descriptor();
      if (verified !== null && (current.fd !== verified.fd || current.device !== verified.device || current.inode !== verified.inode)) {
        throw new Error("Metal child launch requires a verified inherited lease");
      }
      // The same descriptor number/inode can have been reopened onto another OFD.
      // Reprove actual kernel ownership at every handoff; never unlock/close the borrower.
      validate(current, metalExecutionWaitSeconds(environment));
      return withDescriptor(options, current.fd, options.env ?? environment);
    },
  };
}

const entry = createMetalExecutionEntry();
export const enterMacMetalExecution = (waitSeconds) => {
  if (process.platform === "darwin" && process.env.SCENEWORKS_MEMORY_WATCHDOG_SOCKET && process.env[METAL_LOCK_FD] === undefined) {
    throw new Error("acquire the Metal execution lease outside the physical watchdog before launching a Metal model");
  }
  return entry.enter(waitSeconds);
};
export function metalExecutionCommand(command, args, waitSeconds, platform = process.platform, env = process.env) {
  if (platform !== "darwin") return { command, args };
  return { command: "python3.12", args: ["-I", helper, "--wait-seconds", String(metalExecutionWaitSeconds(env, waitSeconds)), "--", command, ...args] };
}
export const metalChildOptions = (options) => entry.childOptions(options);
export const isDirectModule = (url) => process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(url);
