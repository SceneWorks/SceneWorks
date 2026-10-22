#!/usr/bin/env node
// Test-child isolation only. Never install this environment into a runner, user profile or model job.
import { lstatSync, mkdirSync, mkdtempSync, realpathSync } from "node:fs";
import { spawnSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const requireFact = (value, message) => { if (!value) throw new Error(`unit HF sandbox: ${message}`); };
const within = (root, child) => { const relative = path.relative(root, child); return relative === "" || (!relative.startsWith(`..${path.sep}`) && relative !== ".." && !path.isAbsolute(relative)); };
export function createUnitHfSandbox(checkout, env = process.env) {
  requireFact(path.isAbsolute(env.RUNNER_TEMP ?? ""), "absolute runner temporary directory required");
  const temporary = path.resolve(env.RUNNER_TEMP), realTemporary = realpathSync(temporary);
  requireFact(lstatSync(temporary).isDirectory() && !lstatSync(temporary).isSymbolicLink() && path.relative(temporary, realTemporary) === "", "runner temporary path must not contain symlinks");
  const source = realpathSync(checkout);
  requireFact(!within(source, realTemporary), "runner temporary directory must be outside checkout");
  requireFact(/^[1-9][0-9]*$/.test(env.GITHUB_RUN_ID ?? "") && /^[1-9][0-9]*$/.test(env.GITHUB_RUN_ATTEMPT ?? "") && /^[A-Za-z][A-Za-z0-9_-]{0,80}$/.test(env.GITHUB_JOB ?? ""), "exact run, attempt and job identity required");
  const directory = mkdtempSync(path.join(realTemporary, `unit-hf-${env.GITHUB_RUN_ID}-${env.GITHUB_RUN_ATTEMPT}-${env.GITHUB_JOB}-`));
  const home = path.join(directory, "home"), hub = path.join(home, "hub");
  mkdirSync(home); mkdirSync(hub);
  for (const value of [directory, home, hub]) requireFact(!lstatSync(value).isSymbolicLink() && path.relative(value, realpathSync(value)) === "" && within(realTemporary, value) && !within(source, value), "sandbox must remain owned, outside checkout and free of symlinks");
  return { directory, env: { HF_HOME: home, HF_HUB_CACHE: hub, HUGGINGFACE_HUB_CACHE: hub } };
}

export function runUnitCommand(args, env = process.env, checkout = process.cwd()) {
  requireFact(args.length > 1 && args[0] === "--", "expected -- followed by a test command");
  const sandbox = createUnitHfSandbox(checkout, env);
  const result = spawnSync(args[1], args.slice(2), { cwd: checkout, env: { ...env, ...sandbox.env }, stdio: "inherit" });
  if (result.error) throw result.error;
  return result;
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    const result = runUnitCommand(process.argv.slice(2));
    if (result.signal) process.kill(process.pid, result.signal); // Mirror a completed child's signal; never signal a child or a process group.
    else process.exitCode = result.status ?? 1;
  }
  catch (error) { console.error(error.message); process.exitCode = 1; }
}
