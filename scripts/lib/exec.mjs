import { execFile as callbackExecFile } from "node:child_process";
import { promisify } from "node:util";
import { execFileWithFds } from "./exec-file-with-fds.mjs";

const execute = promisify(callbackExecFile);
export const DEFAULT_EXEC_TIMEOUT_MS = 5 * 60 * 1000;

/** Bounded buffered commands, preserving execFile's argv/env/cwd/encoding, AbortSignal,
 * killSignal, exit errors (including stdout/stderr), and promise.child streaming access.
 * Long-lived supervised services keep their spawn/lifecycle owners. Explicit FD maps
 * use the existing descriptor-aware adapter; ordinary calls stay native execFile.
 * Callers may set larger finite bounds for large output or longer metadata commands.
 */
export function execFileBounded(file, args, options = {}, execFileImpl = execute) {
  const bounded = { timeout: DEFAULT_EXEC_TIMEOUT_MS, maxBuffer: 1024 * 1024, ...options };
  if (!Number.isFinite(bounded.timeout) || bounded.timeout <= 0) throw new RangeError("positive finite exec timeout required");
  if (!Number.isFinite(bounded.maxBuffer) || bounded.maxBuffer < 0) throw new RangeError("finite nonnegative exec maxBuffer required");
  if (Array.isArray(bounded.stdio) && bounded.stdio.length > 3) {
    return execFileWithFds(file, args, bounded, execFileImpl);
  }
  return execFileImpl(file, args, bounded);
}
