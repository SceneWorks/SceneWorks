#!/usr/bin/env node
// Persistent object cache, private capture HEAD. Never checkout/reset/clean the cache or use FETCH_HEAD.
import { execFileSync } from "node:child_process";
import { mkdtemp, mkdir, readFile, writeFile, realpath, rm, rmdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
const git = (root, args) => execFileSync("git", ["-C", root, ...args], { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] }).trim();
const revision = (value) => { if (!/^[0-9a-f]{40}$/.test(value)) throw new Error("exact 40-hex source pin required"); return value; };
export async function prepareSourceWorktree(cache, pin, temporaryRoot) {
  revision(pin);
  cache = await realpath(cache);
  git(cache, ["ls-tree", "-r", pin]);
  // Named exact-SHA refs also keep Cargo's file:// transport independent of the cache HEAD.
  const ref = `refs/sceneworks/source-pins/${pin}`;
  try { git(cache, ["update-ref", ref, pin]); }
  catch (error) { if (git(cache, ["rev-parse", "--verify", ref]) !== pin) throw error; }
  await mkdir(temporaryRoot, { recursive: true });
  const owner = await mkdtemp(path.join(await realpath(temporaryRoot), "memory-catalog-source-"));
  const source = path.join(owner, "source");
  try {
    git(cache, ["worktree", "add", "--detach", source, pin]);
    const head = git(source, ["rev-parse", "HEAD"]), tree = git(source, ["rev-parse", "HEAD^{tree}"]);
    if (head !== pin || git(source, ["status", "--porcelain"])) throw new Error("private capture source is not clean at the exact pin");
    await writeFile(path.join(owner, "owner.json"), JSON.stringify({ version: 1, cache, source, pin, tree }) + "\n", { flag: "wx" });
    return source;
  } catch (error) {
    // No force/recursive deletion. If Git registered a partial worktree, preserve it for diagnosis.
    try { await rmdir(owner); } catch {}
    throw error;
  }
}
export async function cleanupSourceWorktree(source) {
  source = await realpath(source);
  const owner = path.dirname(source), record = JSON.parse(await readFile(path.join(owner, "owner.json"), "utf8"));
  if (record.version !== 1 || record.source !== source || path.basename(source) !== "source"
      || !path.basename(owner).startsWith("memory-catalog-source-")) throw new Error("not an owned campaign source worktree");
  revision(record.pin);
  const common = await realpath(git(source, ["rev-parse", "--path-format=absolute", "--git-common-dir"]));
  const expected = await realpath(git(record.cache, ["rev-parse", "--path-format=absolute", "--git-common-dir"]));
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
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const [command, ...args] = process.argv.slice(2);
  (async () => {
    if (command === "prepare" && args.length === 3) console.log(await prepareSourceWorktree(...args));
    else if (command === "cleanup" && args.length === 1) await cleanupSourceWorktree(args[0]);
    else throw new Error("expected prepare CACHE EXACT_PIN TEMP_ROOT or cleanup OWNED_SOURCE");
  })().catch((error) => { console.error(error.message); process.exitCode = 1; });
}
