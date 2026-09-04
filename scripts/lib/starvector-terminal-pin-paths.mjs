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

if (isExecutedModule(import.meta.url)) {
  try {
    const [hostRoot, permanentPin] = process.argv.slice(2);
    if (!permanentPin) fail("usage: <absolute-host-root> <permanent-pin>");
    for (const [name, value] of Object.entries(terminalPinEnvironment(hostRoot, permanentPin))) {
      console.log(`${name}=${value}`);
    }
  } catch (error) {
    console.error(error.message);
    process.exitCode = 1;
  }
}
