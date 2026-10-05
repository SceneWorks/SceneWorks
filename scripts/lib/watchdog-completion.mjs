// Completion-only lifecycle client. The command stays alive until the guard has committed
// its final group/host sample. This protocol grants no allocation or memory-policy exemption.
import { createConnection } from "node:net";
import { open, readdir } from "node:fs/promises";
import { createHash } from "node:crypto";
import path from "node:path";

export async function syncEvidence(directory) {
  const hash = createHash("sha256");
  async function visit(dir) {
    for (const entry of (await readdir(dir, { withFileTypes: true })).sort((a, b) => a.name.localeCompare(b.name))) {
      const file = path.join(dir, entry.name);
      if (entry.isDirectory()) await visit(file);
      else if (entry.isFile()) {
        const handle = await open(file, "r");
        try {
          await handle.sync();
          hash.update(JSON.stringify(path.relative(directory, file)));
          const fileHash = createHash("sha256");
          const buffer = Buffer.alloc(1024 * 1024);
          for (;;) {
            const { bytesRead } = await handle.read(buffer, 0, buffer.length, null);
            if (!bytesRead) break;
            fileHash.update(buffer.subarray(0, bytesRead));
          }
          hash.update(fileHash.digest());
        } finally { await handle.close(); }
      } else throw new Error(`completion evidence must be regular files: ${file}`);
    }
    const handle = await open(dir, "r");
    try { await handle.sync(); } finally { await handle.close(); }
  }
  await visit(directory);
  // Persist the evidence directory's own entry as well as the entries beneath it.
  const parent = await open(path.dirname(directory), "r");
  try { await parent.sync(); } finally { await parent.close(); }
  return hash.digest("hex");
}

export async function connectWatchdogCompletion({
  socketPath = process.env.SCENEWORKS_MEMORY_WATCHDOG_SOCKET,
  timeoutMs = 120_000, onFailure = () => {},
} = {}) {
  if (!socketPath) return { complete: async () => {}, close() {} };
  const socket = createConnection(socketPath);
  let state = "hello", nonce, buffer = "", failure, resolveReady, rejectReady, resolveDone, rejectDone;
  const ready = new Promise((resolve, reject) => { resolveReady = resolve; rejectReady = reject; });
  const fail = (error) => {
    if (failure || state === "closed") return;
    failure = error;
    rejectReady(error);
    rejectDone?.(error);
    socket.destroy();
    onFailure(error);
  };
  socket.setEncoding("utf8");
  socket.setTimeout(timeoutMs, () => fail(new Error("watchdog completion channel timed out")));
  socket.on("error", fail);
  socket.on("close", () => {
    if (state !== "released" && state !== "closed") fail(new Error("watchdog completion channel closed before acknowledgement"));
  });
  socket.on("data", (chunk) => {
    buffer += chunk;
    if (buffer.length > 8192) return fail(new Error("watchdog completion message exceeded size bound"));
    while (buffer.includes("\n")) {
      const end = buffer.indexOf("\n");
      const line = buffer.slice(0, end);
      buffer = buffer.slice(end + 1);
      try {
        if (state === "hello") {
          const hello = JSON.parse(line);
          if (hello.protocol !== "sceneworks-memory-watchdog-completion-v1" || !/^[a-f0-9]{64}$/.test(hello.nonce)) {
            throw new Error("invalid watchdog completion greeting");
          }
          nonce = hello.nonce;
          state = "go";
          socket.write(`ACK ${nonce}\n`);
        } else if (state === "go" && line === `GO ${nonce}`) {
          state = "running";
          resolveReady();
        } else if (["running", "syncing", "done"].includes(state) && line === `PING ${nonce}`) {
          // A heartbeat keeps the channel timeout bounded while the driver runs or fsyncs.
        } else if (state === "done" && line === `BYE ${nonce}`) {
          state = "released";
          socket.end();
          resolveDone();
        } else throw new Error(`unexpected watchdog completion message in ${state}`);
      } catch (error) { fail(error); return; }
    }
  });
  await ready;
  return {
    async complete(evidenceDirectory) {
      if (failure) throw failure;
      if (state !== "running") throw new Error("completion may only be requested once after GO");
      state = "syncing";
      let digest;
      try { digest = await syncEvidence(evidenceDirectory); }
      catch (error) { fail(error); throw error; }
      if (failure) throw failure;
      const done = new Promise((resolve, reject) => { resolveDone = resolve; rejectDone = reject; });
      state = "done";
      socket.write(`DONE ${nonce} ${digest}\n`);
      await done;
    },
    close() { state = "closed"; socket.destroy(); },
  };
}
