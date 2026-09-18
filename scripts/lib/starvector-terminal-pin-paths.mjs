import { lstat, mkdir, realpath } from "node:fs/promises";
import path from "node:path";
import { isExecutedModule } from "../starvector-terminal-cli.mjs";

const REVISION = /^[0-9a-f]{40}$/;

function fail(message) {
  throw new Error(`starvector terminal pin paths: ${message}`);
}

export function validateTerminalPermanentPin(permanentPin) {
  if (!REVISION.test(permanentPin ?? "")) {
    fail("permanent pin must be an exact lowercase 40-hex commit");
  }
  return permanentPin;
}

export function terminalPinPaths(hostRoot, permanentPin, pathApi = path) {
  validateTerminalPermanentPin(permanentPin);
  if (typeof hostRoot !== "string" || !hostRoot || /[\r\n]/.test(hostRoot) || !pathApi.isAbsolute(hostRoot)) {
    fail("host root must be an absolute single-line path");
  }
  const root = pathApi.resolve(hostRoot);
  const pinRoot = pathApi.join(root, "pins", permanentPin);
  const relative = pathApi.relative(root, pinRoot);
  if (!relative || relative === ".." || relative.startsWith(`..${pathApi.sep}`) || pathApi.isAbsolute(relative)) {
    fail("derived pin root escapes the host root");
  }
  const preflightRoot = pathApi.join(pinRoot, "inference-preflight");
  return Object.freeze({
    hostRoot: root,
    pinRoot,
    inferenceRoot: pathApi.join(pinRoot, "inference"),
    preflightRoot,
    preflightIndex: pathApi.join(preflightRoot, "starvector-terminal-preflight.json"),
    corpusAssetsRoot: pathApi.join(pinRoot, "corpora", "starvector-terminal-v1"),
  });
}

export function terminalPinEnvironment(hostRoot, permanentPin, pathApi = path) {
  const roots = terminalPinPaths(hostRoot, permanentPin, pathApi);
  return {
    STARVECTOR_TERMINAL_PIN_ROOT: roots.pinRoot,
    STARVECTOR_TERMINAL_INFERENCE_ROOT: roots.inferenceRoot,
    STARVECTOR_TERMINAL_INFERENCE_PREFLIGHT_ROOT: roots.preflightRoot,
    STARVECTOR_TERMINAL_INFERENCE_PREFLIGHT: roots.preflightIndex,
    STARVECTOR_TERMINAL_CORPUS_ASSETS_ROOT: roots.corpusAssetsRoot,
  };
}

function samePhysicalPath(left, right) {
  const normalize = (value) => process.platform === "win32" ? value.toLowerCase() : value;
  return normalize(path.normalize(left)) === normalize(path.normalize(right));
}

/**
 * Prove every existing component from the durable host root through `targetRoot`
 * is an ordinary directory. Missing suffixes are allowed because provision will
 * create them, then call this check again before publication. Symlinks and
 * Windows junctions are reported by Node as symbolic links; comparing each
 * component's physical path also refuses any other redirecting reparse point.
 */
export async function assertTerminalPhysicalContainment(hostRoot, targetRoot) {
  if (typeof hostRoot !== "string" || !hostRoot || typeof targetRoot !== "string" || !targetRoot) {
    fail("physical containment requires host and target roots");
  }
  const root = path.resolve(hostRoot);
  const target = path.resolve(targetRoot);
  const relative = path.relative(root, target);
  if (relative === ".." || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative)) {
    fail("physical containment target escapes the host root");
  }
  const suffix = relative ? relative.split(path.sep) : [];
  const components = [root];
  for (let index = 0; index < suffix.length; index += 1) {
    components.push(path.join(root, ...suffix.slice(0, index + 1)));
  }
  let physicalRoot;
  for (let index = 0; index < components.length; index += 1) {
    const component = components[index];
    const info = await lstat(component).catch((error) => {
      if (error?.code === "ENOENT") return null;
      throw error;
    });
    if (!info) {
      if (index === 0) fail(`physical containment requires an existing host root: ${root}`);
      break;
    }
    if (info.isSymbolicLink() || !info.isDirectory()) {
      fail(`physical containment rejects symlink, junction, reparse, or non-directory component: ${component}`);
    }
    const observed = await realpath(component);
    if (index === 0) {
      physicalRoot = observed;
    } else {
      const expected = path.join(physicalRoot, ...suffix.slice(0, index));
      if (!samePhysicalPath(observed, expected)) {
        fail(`physical containment rejects redirected component: ${component}`);
      }
    }
  }
  return target;
}

export async function ensureTerminalPhysicalDirectory(hostRoot, targetRoot) {
  const target = await assertTerminalPhysicalContainment(hostRoot, targetRoot);
  const root = path.resolve(hostRoot);
  const relative = path.relative(root, target);
  const suffix = relative ? relative.split(path.sep) : [];
  for (let index = 0; index < suffix.length; index += 1) {
    const component = path.join(root, ...suffix.slice(0, index + 1));
    try {
      await mkdir(component);
    } catch (error) {
      if (error?.code !== "EEXIST") throw error;
    }
    await assertTerminalPhysicalContainment(root, component);
  }
  return target;
}

export async function assertTerminalPinPhysicalContainment(hostRoot, permanentPin) {
  const roots = terminalPinPaths(hostRoot, permanentPin);
  await assertTerminalPhysicalContainment(roots.hostRoot, roots.pinRoot);
  return roots;
}

if (isExecutedModule(import.meta.url)) {
  try {
    const [hostRoot, permanentPin] = process.argv.slice(2);
    if (!permanentPin) fail("usage: <absolute-host-root> <permanent-pin>");
    await assertTerminalPinPhysicalContainment(hostRoot, permanentPin);
    for (const [name, value] of Object.entries(terminalPinEnvironment(hostRoot, permanentPin))) {
      console.log(`${name}=${value}`);
    }
  } catch (error) {
    console.error(error.message);
    process.exitCode = 1;
  }
}
