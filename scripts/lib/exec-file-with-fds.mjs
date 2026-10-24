// Node execFile deliberately discards options.stdio. Use spawn ONLY when a cooperating
// descriptor must survive, retaining execFile's existing timeout/maxBuffer/killSignal semantics.
import { execFile as callbackExecFile, spawn } from "node:child_process";
import { promisify } from "node:util";
const originalExecFile = promisify(callbackExecFile);

export async function execFileWithFds(file, args, options = {}, fallback = originalExecFile) {
  if (!Array.isArray(options.stdio) || options.stdio.length <= 3) return fallback(file, args, options);
  const maxBuffer = options.maxBuffer ?? 1024 * 1024;
  if (!(maxBuffer >= 0)) throw new RangeError("invalid maxBuffer");
  const child = spawn(file, args, options); // same argv/env/cwd/timeout/signal, plus actual FD map
  const chunks = { stdout: [], stderr: [] }, bytes = { stdout: 0, stderr: 0 };
  let failure = null;
  const collect = (stream) => (value) => {
    const buffer = Buffer.isBuffer(value) ? value : Buffer.from(value);
    const available = Math.max(0, maxBuffer - bytes[stream]);
    if (available > 0) chunks[stream].push(buffer.subarray(0, available));
    bytes[stream] += buffer.length;
    if (bytes[stream] > maxBuffer && failure === null) {
      failure = new RangeError(`${stream} maxBuffer length exceeded`);
      failure.code = "ERR_CHILD_PROCESS_STDIO_MAXBUFFER";
      // This is the existing execFile output-limit behavior, not a new model/resource limit.
      child.kill(options.killSignal ?? "SIGTERM");
    }
  };
  child.stdout?.on("data", collect("stdout")); child.stderr?.on("data", collect("stderr"));
  const result = await new Promise((resolve) => {
    child.once("error", (error) => { failure ??= error; });
    child.once("close", (code, signal) => resolve({ code, signal }));
  });
  const encoding = options.encoding ?? "utf8";
  const decode = (parts) => { const buffer = Buffer.concat(parts); return encoding === "buffer" ? buffer : buffer.toString(encoding); };
  const stdout = decode(chunks.stdout), stderr = decode(chunks.stderr);
  if (failure || result.code !== 0 || result.signal) {
    const error = failure ?? new Error(`Command failed: ${file}\n${stderr}`);
    error.stdout = stdout; error.stderr = stderr; error.signal = result.signal;
    error.code ??= result.code; error.killed = child.killed;
    throw error;
  }
  return { stdout, stderr };
}
