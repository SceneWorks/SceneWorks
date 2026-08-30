import assert from "node:assert/strict";
import { execFileSync, spawn } from "node:child_process";
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";

const helperTarget = path.join(process.cwd(), "target", "starvector-terminal-lease-test");
const helper = path.join(helperTarget, "debug", "starvector_terminal_lease");
function buildHelper() {
  // Keep the test independent of a prior developer/CI build and execute the
  // exact current-tree fs2 source which the workflow builds for every tuple.
  execFileSync("cargo", ["build", "--locked", "-p", "sceneworks-worker", "--bin", "starvector_terminal_lease"], { cwd: process.cwd(), env: { ...process.env, CARGO_TARGET_DIR: helperTarget }, stdio: "pipe" });
}
const waitFor = (child, text) => new Promise((resolve, reject) => {
  let stderr = ""; child.stderr.setEncoding("utf8"); child.stderr.on("data", (chunk) => { stderr += chunk; });
  child.stdout.setEncoding("utf8"); child.stdout.once("data", (chunk) => String(chunk).trim() === text ? resolve() : reject(new Error(`${chunk}; ${stderr}`)));
  child.once("error", reject); child.once("exit", (code) => reject(new Error(`lease helper exited ${code}: ${stderr}`)));
});
const exit = (child) => new Promise((resolve) => child.once("exit", (code) => resolve(code)));

test("compiled fs2 helper holds an OS advisory lock and rejects a second holder", async () => {
  buildHelper();
  const directory = await mkdtemp(path.join(tmpdir(), "starvector-fs2-"));
  const lock = path.join(directory, "shared.lock");
  const first = spawn(helper, ["hold", lock, '{"owner":"first"}'], { stdio: ["pipe", "pipe", "pipe"] });
  await waitFor(first, "locked");
  const second = spawn(helper, ["hold", lock, '{"owner":"second"}'], { stdio: ["pipe", "pipe", "pipe"] });
  let secondError = ""; second.stderr.setEncoding("utf8"); second.stderr.on("data", (chunk) => { secondError += chunk; });
  assert.equal(await exit(second), 2); assert.match(secondError, /advisory lease is held; never auto-break/);
  first.stdin.end(); assert.equal(await exit(first), 0);
});
