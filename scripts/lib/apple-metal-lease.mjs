// Narrow parent -> owned-child FD handoff. No service API, PID substitution, broker, or model
// option change. The frozen Python protocol acquires the SAME open-file-description as Node.
import { AsyncLocalStorage } from "node:async_hooks";
import { randomInt } from "node:crypto";
import { spawn, spawnSync } from "node:child_process";
import { constants, openSync, closeSync, lstatSync, fstatSync } from "node:fs";
import { performance } from "node:perf_hooks";
import { fileURLToPath } from "node:url";
import { METAL_LOCK_FD, METAL_LOCK_PATH, METAL_LOCK_PROTOCOL, metalExecutionWaitSeconds, metalChildOptions } from "./apple-metal-execution-lock.mjs";

const bridgePath = fileURLToPath(new URL("../ci/metal_execution_node_bridge.py", import.meta.url));
const scope = new AsyncLocalStorage();
// A seek changes the OFD offset, not file length/content. Keep its positive generation cookie
// within signed 32-bit file offsets: Linux filesystems may reject the old 48-bit seek range.
// It is not an authorization token; canonical inode + kernel flock remain mandatory, and a
// newly reopened foreign description has offset zero, never this positive cookie.
const OFD_MARKER_MAX = 0x7fffffff;
const noopLease = Object.freeze({ childOptions: (options) => options, close: async () => {}, record: () => null });

function descriptorOptions(fd, options, environment) {
  const original = options.stdio ?? "pipe";
  const stdio = Array.isArray(original) ? [...original] : [original, original, original];
  while (stdio.length < 3) stdio.push("pipe");
  const childFd = stdio.length;
  stdio.push(fd);
  return { ...options, stdio, env: { ...environment, [METAL_LOCK_FD]: String(childFd) } };
}
function identity(fd, canonicalPath, fstat, lstat) {
  const opened = fstat(fd), named = lstat(canonicalPath);
  if (!opened.isFile() || !named.isFile() || BigInt(opened.nlink) !== 1n || BigInt(named.nlink) !== 1n
      || opened.dev !== named.dev || opened.ino !== named.ino) throw new Error("Metal descriptor lost its unique canonical inode");
  return { device: BigInt(opened.dev), inode: BigInt(opened.ino) };
}
function rawInteger(body, key) {
  const value = body.match(new RegExp(`"${key}"\\s*:\\s*([0-9]+)`))?.[1];
  if (value === undefined) throw new Error(`Metal bridge omitted ${key}`);
  return BigInt(value);
}

// Injected dependencies are CPU-test seams, never CLI/env namespace overrides or fallbacks.
export function createMacMetalLeaseFactory({
  platform = process.platform, environment = process.env, python = "python3.12",
  bridge = bridgePath, canonicalPath = METAL_LOCK_PATH,
  spawnImpl = spawn, spawnSyncImpl = spawnSync, open = openSync, close = closeSync,
  lstat = (file) => lstatSync(file, { bigint: true }), fstat = (fd) => fstatSync(fd, { bigint: true }),
  now = () => performance.now(), marker = () => randomInt(1, OFD_MARKER_MAX + 1),
} = {}) {
  async function invoke(mode, waitSeconds, fd = null, offset = null) {
    const args = ["-I", bridge, "--mode", mode, "--wait-seconds", String(waitSeconds)];
    if (offset !== null) args.push("--marker-offset", String(offset));
    let options = { env: environment, stdio: ["ignore", "pipe", "pipe"] };
    if (fd !== null) options = descriptorOptions(fd, options, environment);
    const child = spawnImpl(python, args, options);
    let stdout = "", stderr = "";
    return await new Promise((resolve, reject) => {
      let exceeded = false;
      const collect = (kind) => (chunk) => {
        if (exceeded) return; // keep draining without unbounded buffering or any PID signal
        if (kind === "out") stdout += chunk; else stderr += chunk;
        if (stdout.length + stderr.length > 65536) exceeded = true;
      };
      child.stdout.on("data", collect("out")); child.stderr.on("data", collect("err"));
      child.once("error", reject);
      child.once("close", (code, signal) => code === 0 && !signal && !exceeded ? resolve(stdout)
        : reject(new Error(`Metal acquisition bridge failed: ${stderr.trim() || signal || code}`)));
    });
  }
  return async function acquire(waitSeconds) {
    if (platform !== "darwin") return noopLease;
    const wait = metalExecutionWaitSeconds(environment, waitSeconds), deadline = now() + wait * 1000;
    const remaining = () => { const seconds = (deadline - now()) / 1000; if (!(seconds > 0)) throw new Error("Metal acquisition wait expired; no model launched"); return seconds; };
    const inherited = environment[METAL_LOCK_FD];
    let fd, owned = false, captured, offset = null, marked = false;
    const inspectOffset = (mark = false) => {
      const args = ["-I", bridge, "--mode", "inspect", "--wait-seconds", "1"];
      if (mark) args.push("--marker-offset", String(offset));
      const options = descriptorOptions(fd, { stdio: ["ignore", "pipe", "pipe"], encoding: "utf8", timeout: 5000 }, environment);
      const result = spawnSyncImpl(python, args, options);
      let observed = null, parseError = null;
      try { observed = JSON.parse(result.stdout).offset; } catch (error) { parseError = error.message; }
      if (result.error || result.status !== 0 || result.signal || observed !== offset || parseError) {
        const detail = { phase: mark ? "mark" : "verify", expectedOffset: offset, observedOffset: observed,
          status: result.status, signal: result.signal, errorCode: result.error?.code ?? null,
          error: String(result.error?.message ?? "").slice(0, 256), parseError,
          stderr: String(result.stderr ?? "").slice(0, 512), stdout: String(result.stdout ?? "").slice(0, 256) };
        throw new Error(`Metal open-file-description was lost/reused; refusing to close or launch; inspect=${JSON.stringify(detail)}`);
      }
    };
    if (inherited !== undefined) {
      if (!/^[0-9]+$/.test(inherited) || !Number.isSafeInteger(Number(inherited)) || Number(inherited) < 3) throw new Error("invalid inherited Metal descriptor");
      fd = Number(inherited);
    } else {
      try { lstat(canonicalPath); } catch (error) {
        if (error.code !== "ENOENT") throw error;
        // Only the unchanged helper creates/validates the fixed namespace; never a private file.
        await invoke("prepare", remaining());
      }
      fd = open(canonicalPath, constants.O_RDWR | (constants.O_NOFOLLOW ?? 0));
      owned = true;
    }
    try {
      captured = identity(fd, canonicalPath, fstat, lstat);
      offset = owned ? marker() : null;
      if (owned && (!Number.isSafeInteger(offset) || offset <= 0 || offset > OFD_MARKER_MAX)) throw new Error("invalid portable owned-description marker");
      // Mark synchronously before the first asynchronous acquisition yield. Even an acquisition
      // failure must not close a foreign FD reopened on the same canonical inode.
      if (owned) { inspectOffset(true); marked = true; }
      const ownershipRecord = (output) => {
        const line = output.split(/\r?\n/).find((value) => value.startsWith("METAL_EXECUTION_LEASE "));
        const body = line?.slice("METAL_EXECUTION_LEASE ".length), record = body ? JSON.parse(body) : null;
        if (record?.protocol !== METAL_LOCK_PROTOCOL || record.path !== canonicalPath || record.borrowed !== true
            || rawInteger(body, "device") !== captured.device || rawInteger(body, "inode") !== captured.inode) throw new Error("Metal acquisition has no matching kernel-flock proof");
        return record;
      };
      const output = await invoke("acquire", remaining(), fd, offset);
      const record = ownershipRecord(output);
      let released = false;
      const assertDescription = () => {
        if (released) throw new Error("Metal lease scope already closed");
        const current = identity(fd, canonicalPath, fstat, lstat);
        if (current.device !== captured.device || current.inode !== captured.inode) throw new Error("Metal descriptor was replaced/reused");
        if (owned) inspectOffset();
        else {
          // Borrowers must not alter the owner's offset cookie. A fresh duplicate must
          // prove the actual kernel lock each time, not merely the same inode/FD number.
          const options = descriptorOptions(fd, { stdio: ["ignore", "pipe", "pipe"], encoding: "utf8", timeout: 5000, maxBuffer: 65536 }, environment);
          const result = spawnSyncImpl(python, ["-I", bridge, "--mode", "acquire", "--wait-seconds", "1"], options);
          if (result.error || result.status !== 0 || result.signal) throw new Error(`borrowed Metal kernel ownership was lost: ${String(result.error?.message ?? result.stderr ?? result.signal).slice(0, 512)}`);
          ownershipRecord(result.stdout);
        }
      };
      return {
        childOptions(options) { assertDescription(); return descriptorOptions(fd, options, options.env ?? environment); },
        record() { return { ...record, device: captured.device.toString(), inode: captured.inode.toString(), metadata_helper_pid: record.pid, pid: process.pid, borrowed: !owned, phase: released ? "Node lease scope ended; no post-scope ownership claim" : "Node parent holds actual kernel lease; only child fd inheritance extends ownership" }; },
        async close() {
          if (released) return;
          try { assertDescription(); } catch (error) { released = true; throw error; }
          released = true;
          // NEVER LOCK_UN: the actual worker/cooperating grandchildren may still own copies.
          if (owned) close(fd);
        },
      };
    } catch (error) {
      if (owned) {
        // Acquisition failed before a child could inherit; still refuse to close a reused inode.
        try {
          const current = identity(fd, canonicalPath, fstat, lstat);
          if (captured && marked && current.device === captured.device && current.inode === captured.inode) { inspectOffset(); close(fd); }
        } catch {}
      }
      throw error;
    }
  };
}

const acquire = createMacMetalLeaseFactory();
export const scopedMetalChildOptions = (options) => scope.getStore()?.childOptions(options) ?? options;
export const metalLeaseChildOptions = (options) => scope.getStore()?.childOptions(options) ?? metalChildOptions(options);
export const withMacMetalExecutionLease = (waitSeconds, work) => withMacMetalExecutionLeaseUsing(acquire, waitSeconds, work);
// Internal CPU dependency-injection seam. Production CLI still uses only the fixed canonical
// factory above; this adds no environment/path switch or alternative production lock namespace.
export async function withMacMetalExecutionLeaseUsing(acquireLease, waitSeconds, work) {
  const active = scope.getStore();
  if (active) return work(active); // exact inherited description, no self-conflicting reopen
  const lease = await acquireLease(waitSeconds);
  let result, primary = null;
  try { result = await scope.run(lease, () => work(lease)); } catch (error) { primary = error; }
  try { await lease.close(); } catch (error) {
    if (primary) primary.message += `; Metal lease close also failed: ${error.message}`;
    else primary = error;
  }
  if (primary) throw primary;
  return result;
}
