#!/usr/bin/env node

import { constants, accessSync, lstatSync, readdirSync, realpathSync, rmdirSync, statSync, unlinkSync } from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { spawnSync } from "node:child_process";

const DEFAULT_PROBE_TIMEOUT_MS = 15_000;
const PROBE = `import json,platform,struct,sys;print(json.dumps({"executable":sys.executable,"base_executable":getattr(sys,"_base_executable",None),"prefix":sys.prefix,"base_prefix":sys.base_prefix,"version":[sys.version_info.major,sys.version_info.minor,sys.version_info.micro],"implementation":platform.python_implementation(),"architecture":platform.machine(),"pointer_bits":struct.calcsize("P")*8}))`;

function safeAbsolutePath(value) {
  return typeof value === "string" && path.isAbsolute(value) && !/[\0\r\n"]/.test(value);
}

function executableRealpath(value) {
  if (!safeAbsolutePath(value)) return null;
  try {
    const resolved = realpathSync(value);
    if (!statSync(resolved).isFile()) return null;
    accessSync(resolved, constants.X_OK);
    return resolved;
  } catch {
    return null;
  }
}

function parseIdentity(stdout) {
  let identity;
  try {
    identity = JSON.parse(stdout);
  } catch {
    return null;
  }
  if (!identity || typeof identity !== "object" || !Array.isArray(identity.version) || identity.version.length !== 3 || !identity.version.every(Number.isInteger)) {
    return null;
  }
  return identity;
}

export function probeStarVectorMacPython(value, { timeoutMs = DEFAULT_PROBE_TIMEOUT_MS } = {}) {
  const executable = executableRealpath(value);
  if (!executable) return null;
  // Invoke the supplied path, not its realpath: CPython finds a venv's adjacent
  // pyvenv.cfg through that path even when bin/python is a symlink to the base interpreter.
  const probe = spawnSync(path.resolve(value), ["-c", PROBE], {
    encoding: "utf8",
    timeout: timeoutMs,
    maxBuffer: 1024 * 1024,
    windowsHide: true,
  });
  if (probe.error || probe.signal || probe.status !== 0) return null;
  const identity = parseIdentity(probe.stdout);
  if (!identity) return null;
  const reportedExecutable = executableRealpath(identity.executable);
  if (!reportedExecutable || reportedExecutable !== executable) return null;
  return { executable, identity };
}

function isSupportedIdentity(identity) {
  return identity.version[0] === 3 &&
    identity.version[1] === 12 &&
    identity.implementation === "CPython" &&
    ["arm64", "aarch64"].includes(identity.architecture.toLowerCase()) &&
    identity.pointer_bits === 64;
}

export function selectStarVectorMacPython(candidatePaths, options) {
  for (const candidate of candidatePaths) {
    const result = probeStarVectorMacPython(candidate, options);
    if (result && isSupportedIdentity(result.identity)) return result;
  }
  throw new Error("StarVector terminal metrics require an existing absolute CPython 3.12 arm64 interpreter");
}

export function validateStarVectorMacVenv(bootstrapPath, venvPythonPath, options) {
  const bootstrap = selectStarVectorMacPython([bootstrapPath], options);
  const venv = probeStarVectorMacPython(venvPythonPath, options);
  if (!venv || !isSupportedIdentity(venv.identity)) {
    throw new Error("StarVector terminal metrics venv is not a supported CPython 3.12 arm64 interpreter");
  }
  const baseExecutable = executableRealpath(venv.identity.base_executable);
  if (!baseExecutable || baseExecutable !== bootstrap.executable || venv.identity.version.some((part, index) => part !== bootstrap.identity.version[index])) {
    throw new Error("StarVector terminal metrics venv is not bound to the selected exact CPython 3.12 arm64 interpreter");
  }
  const expectedPrefix = path.dirname(path.dirname(path.resolve(venvPythonPath)));
  if (!safeAbsolutePath(venv.identity.prefix) || !safeAbsolutePath(venv.identity.base_prefix)) {
    throw new Error("StarVector terminal metrics venv does not report valid prefix identities");
  }
  let canonicalExpectedPrefix;
  let canonicalObservedPrefix;
  let canonicalBasePrefix;
  try {
    canonicalExpectedPrefix = realpathSync(expectedPrefix);
    canonicalObservedPrefix = realpathSync(venv.identity.prefix);
    canonicalBasePrefix = realpathSync(venv.identity.base_prefix);
  } catch {
    throw new Error("StarVector terminal metrics venv does not report valid prefix identities");
  }
  if (canonicalObservedPrefix !== canonicalExpectedPrefix || canonicalObservedPrefix === canonicalBasePrefix) {
    throw new Error("StarVector terminal metrics Python is not an isolated venv at the expected metrics root");
  }
  return { bootstrap, venv };
}

function checkedExactRoot(targetRoot, allowedRoot) {
  if (!safeAbsolutePath(targetRoot) || !safeAbsolutePath(allowedRoot)) {
    throw new Error("refusing to remove a metrics venv outside the exact workflow-owned terminal root");
  }
  const target = path.resolve(targetRoot);
  const allowed = path.resolve(allowedRoot);
  if (target !== allowed || target === path.parse(target).root) {
    throw new Error("refusing to remove a metrics venv outside the exact workflow-owned terminal root");
  }
  return target;
}

export function removeStarVectorMacMetricsTree(targetRoot, allowedRoot) {
  const target = checkedExactRoot(targetRoot, allowedRoot);
  let root;
  try {
    root = lstatSync(target);
  } catch (error) {
    if (error?.code === "ENOENT") return;
    throw error;
  }
  if (root.isSymbolicLink()) {
    throw new Error(`refusing to remove a metrics venv containing a symlink: ${target}`);
  }

  const pending = [{ file: target, expanded: false }];
  while (pending.length > 0) {
    const frame = pending.pop();
    const item = lstatSync(frame.file);
    if (item.isSymbolicLink()) {
      unlinkSync(frame.file);
      continue;
    }
    if (!item.isDirectory()) {
      if (!item.isFile()) throw new Error(`refusing to remove a metrics venv containing a special file: ${frame.file}`);
      unlinkSync(frame.file);
      continue;
    }
    if (!frame.expanded) {
      const children = readdirSync(frame.file).map((name) => path.join(frame.file, name));
      pending.push({ file: frame.file, expanded: true });
      for (let index = children.length - 1; index >= 0; index -= 1) {
        pending.push({ file: children[index], expanded: false });
      }
      continue;
    }
    if (readdirSync(frame.file).length !== 0) {
      throw new Error(`refusing to remove a metrics venv that changed during validation: ${frame.file}`);
    }
    rmdirSync(frame.file);
  }
}

function usage() {
  throw new Error("usage: select-starvector-macos-python.mjs select <absolute candidates...> | verify-venv <bootstrap> <venv-python> | remove-metrics-tree <target> <exact-allowed-root>");
}

export function main(argv = process.argv.slice(2)) {
  const [command, ...args] = argv;
  if (command === "select" && args.length > 0) {
    process.stdout.write(`${selectStarVectorMacPython(args).executable}\n`);
    return;
  }
  if (command === "verify-venv" && args.length === 2) {
    validateStarVectorMacVenv(args[0], args[1]);
    return;
  }
  if (command === "remove-metrics-tree" && args.length === 2) {
    removeStarVectorMacMetricsTree(args[0], args[1]);
    return;
  }
  usage();
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  try {
    main();
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  }
}
