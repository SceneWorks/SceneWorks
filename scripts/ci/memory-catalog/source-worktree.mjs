#!/usr/bin/env node
// Persistent object cache, private capture HEAD. Never checkout/reset/clean the cache or use FETCH_HEAD.
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdtemp, mkdir, readFile, writeFile, realpath, rm, rmdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { setTimeout as delay } from "node:timers/promises";
const git = (root, args) => execFileSync("git", ["-C", root, ...args], { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] }).trim();
const revision = (value) => { if (!/^[0-9a-f]{40}$/.test(value)) throw new Error("exact 40-hex source pin required"); return value; };
// Git derives its common-directory worktree ID from the checkout's basename.
// Derive a different basename for each owned directory, even across temp roots.
const ownedLeaf = (owner) => `source-${createHash("sha256").update(owner).digest("hex").slice(0, 32)}`;
// Git scans all worktree registrations while adding/removing one. Serialize only this cache's
// short metadata transactions, never the capture/model lifetime or another object cache.
export async function withSourceMetadata(cache, operation) {
  const common = await realpath(git(cache, ["rev-parse", "--path-format=absolute", "--git-common-dir"]));
  const lock = path.join(common, "sceneworks-source-metadata.lock");
  const deadline = performance.now() + 10_000;
  for (;;) {
    try { await mkdir(lock, { mode: 0o700 }); break; }
    catch (error) {
      if (error.code !== "EEXIST") throw error;
      if (performance.now() >= deadline) throw new Error(`source-cache metadata transaction timed out; retained owner lock: ${lock}`);
      await delay(10);
    }
  }
  try { return await operation(); }
  finally { await rmdir(lock); }
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
