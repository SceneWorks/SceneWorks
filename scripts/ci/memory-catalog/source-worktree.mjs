#!/usr/bin/env node
// Persistent object cache, private capture HEAD. Never checkout/reset/clean the cache or use FETCH_HEAD.
import { execFileSync, spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdtemp, mkdir, readFile, writeFile, realpath, rm, rmdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const git = (root, args) => execFileSync("git", ["-C", root, ...args], { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] }).trim();
const revision = (value) => { if (!/^[0-9a-f]{40}$/.test(value)) throw new Error("exact 40-hex source pin required"); return value; };
// Git derives its common-directory worktree ID from the checkout's basename.
// Derive a different basename for each owned directory, even across temp roots.
const ownedLeaf = (owner) => `source-${createHash("sha256").update(owner).digest("hex").slice(0, 32)}`;
// Git scans all worktree registrations while adding/removing one. Serialize only this cache's
// short metadata transactions, never the capture/model lifetime or another object cache.
// A persistent regular file names an OS-owned lock. It is never unlinked or replaced: the
// kernel releases ownership when the helper exits, including after its parent's stdin closes.
// Python is already a prerequisite of the native campaign lanes (stdlib only, no packages).
const metadataLocker = `
import errno, os, sys, time
with open(sys.argv[1], "a+b", buffering=0) as handle:
    if os.name == "nt":
        import msvcrt
        if os.fstat(handle.fileno()).st_size == 0: handle.write(b"\\0")
        def lock():
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        def lock(): fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    deadline = time.monotonic() + 10
    while True:
        try:
            lock()
            break
        except OSError as error:
            if error.errno not in (errno.EACCES, errno.EAGAIN): raise
            if time.monotonic() >= deadline: raise TimeoutError("source-cache metadata transaction timed out")
            time.sleep(0.01)
    print("locked", flush=True)
    sys.stdin.buffer.read()
`;
export async function withSourceMetadata(cache, operation) {
  const common = await realpath(git(cache, ["rev-parse", "--path-format=absolute", "--git-common-dir"]));
  const lock = path.join(common, "sceneworks-source-metadata.kernel-lock");
  const child = spawn(process.platform === "win32" ? "python" : "python3", ["-I", "-c", metadataLocker, lock], { stdio: ["pipe", "pipe", "pipe"] });
  // Acquisition errors (including a missing Python) are reported through the helper result.
  child.stdin.on("error", () => {});
  let stderr = "", ready = "";
  child.stderr.setEncoding("utf8").on("data", (data) => { stderr += data; });
  const ended = new Promise((resolve, reject) => { child.once("error", reject); child.once("close", (code, signal) => resolve({ code, signal })); });
  // Attach before awaiting readiness, so spawn failures never become unhandled rejections.
  ended.catch(() => {});
  try {
    await Promise.race([
      new Promise((resolve) => child.stdout.setEncoding("utf8").on("data", (data) => { ready += data; if (ready === "locked\n" || ready === "locked\r\n") resolve(); })),
      ended.then(({ code, signal }) => { throw new Error(`source-cache metadata lock failed (${code ?? signal}): ${stderr}`); }),
    ]);
    return await operation();
  } finally {
    child.stdin.end();
    const { code, signal } = await ended;
    if (code !== 0 || signal) throw new Error(`source-cache metadata lock failed (${code ?? signal}): ${stderr}`);
  }
}
export async function prepareSourceWorktree(cache, pin, temporaryRoot) {
  revision(pin);
  cache = await realpath(cache);
  return withSourceMetadata(cache, async () => {
    git(cache, ["ls-tree", "-r", pin]);
    // Named exact-SHA refs also keep Cargo's file:// transport independent of the cache HEAD.
    const ref = `refs/sceneworks/source-pins/${pin}`;
    try { git(cache, ["update-ref", ref, pin]); }
    catch (error) { if (git(cache, ["rev-parse", "--verify", ref]) !== pin) throw error; }
    await mkdir(temporaryRoot, { recursive: true });
    const owner = await mkdtemp(path.join(await realpath(temporaryRoot), "memory-catalog-source-"));
    const source = path.join(owner, ownedLeaf(owner));
    try {
      git(cache, ["worktree", "add", "--detach", source, pin]);
      const head = git(source, ["rev-parse", "HEAD"]), tree = git(source, ["rev-parse", "HEAD^{tree}"]);
      if (head !== pin || git(source, ["status", "--porcelain"])) throw new Error("private capture source is not clean at the exact pin");
      await writeFile(path.join(owner, "owner.json"), JSON.stringify({ version: 2, cache, source, pin, tree }) + "\n", { flag: "wx" });
      return source;
    } catch (error) {
      // No force/recursive deletion. If Git registered a partial worktree, preserve it for diagnosis.
      try { await rmdir(owner); } catch {}
      throw error;
    }
  });
}
export async function cleanupSourceWorktree(source) {
  source = await realpath(source);
  const owner = path.dirname(source), record = JSON.parse(await readFile(path.join(owner, "owner.json"), "utf8"));
  const ownedName = record.version === 1 ? "source" : record.version === 2 ? ownedLeaf(owner) : null;
  if (!ownedName || record.source !== source || path.basename(source) !== ownedName
      || !path.basename(owner).startsWith("memory-catalog-source-")) throw new Error("not an owned campaign source worktree");
  revision(record.pin);
  const common = await realpath(git(source, ["rev-parse", "--path-format=absolute", "--git-common-dir"]));
  const expected = await realpath(git(record.cache, ["rev-parse", "--path-format=absolute", "--git-common-dir"]));
  if (common !== expected) throw new Error("owned source identity changed; refusing cleanup");
  return withSourceMetadata(record.cache, async () => {
    const registered = git(record.cache, ["worktree", "list", "--porcelain", "-z"]).split("\0")
      .filter((field) => field.startsWith("worktree ")).some((field) => path.resolve(field.slice(9)) === source);
    if (common !== expected || !registered || git(source, ["rev-parse", "HEAD"]) !== record.pin
        || git(source, ["rev-parse", "HEAD^{tree}"]) !== record.tree || git(source, ["status", "--porcelain"])) {
      throw new Error("owned source identity changed; refusing cleanup");
    }
    // Caller invokes only after the owned walk and model descendants finish. Cancellation retains
    // the checkout rather than guessing whether a surviving child still reads it.
    git(record.cache, ["worktree", "remove", source]);
    await rm(path.join(owner, "owner.json"));
    await rmdir(owner);
  });
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const [command, ...args] = process.argv.slice(2);
  (async () => {
    if (command === "prepare" && args.length === 3) console.log(await prepareSourceWorktree(...args));
    else if (command === "cleanup" && args.length === 1) await cleanupSourceWorktree(args[0]);
    else throw new Error("expected prepare CACHE EXACT_PIN TEMP_ROOT or cleanup OWNED_SOURCE");
  })().catch((error) => { console.error(error.message); process.exitCode = 1; });
}
