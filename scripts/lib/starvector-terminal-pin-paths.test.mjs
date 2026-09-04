import assert from "node:assert/strict";
import { lstat, mkdir, mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import {
  terminalPinEnvironment,
  terminalPinPaths,
  validateTerminalPermanentPin,
} from "./starvector-terminal-pin-paths.mjs";

const firstPin = "1".repeat(40);
const secondPin = "2".repeat(40);
const legacyB2dPin = "b2d9e0917499517cf8c1518e0d360cac8693b0c0";

test("pin-keyed terminal roots use the identical immutable layout on macOS and Windows", () => {
  const mac = terminalPinPaths("/Users/Shared/SceneWorks/starvector-terminal", firstPin, path.posix);
  assert.deepEqual(mac, {
    hostRoot: "/Users/Shared/SceneWorks/starvector-terminal",
    pinRoot: `/Users/Shared/SceneWorks/starvector-terminal/pins/${firstPin}`,
    inferenceRoot: `/Users/Shared/SceneWorks/starvector-terminal/pins/${firstPin}/inference`,
    preflightRoot: `/Users/Shared/SceneWorks/starvector-terminal/pins/${firstPin}/inference-preflight`,
    preflightIndex: `/Users/Shared/SceneWorks/starvector-terminal/pins/${firstPin}/inference-preflight/starvector-terminal-preflight.json`,
    corpusAssetsRoot: `/Users/Shared/SceneWorks/starvector-terminal/pins/${firstPin}/corpora/starvector-terminal-v1`,
  });

  const windows = terminalPinPaths("D:\\sceneworks-terminal", firstPin, path.win32);
  assert.deepEqual(windows, {
    hostRoot: "D:\\sceneworks-terminal",
    pinRoot: `D:\\sceneworks-terminal\\pins\\${firstPin}`,
    inferenceRoot: `D:\\sceneworks-terminal\\pins\\${firstPin}\\inference`,
    preflightRoot: `D:\\sceneworks-terminal\\pins\\${firstPin}\\inference-preflight`,
    preflightIndex: `D:\\sceneworks-terminal\\pins\\${firstPin}\\inference-preflight\\starvector-terminal-preflight.json`,
    corpusAssetsRoot: `D:\\sceneworks-terminal\\pins\\${firstPin}\\corpora\\starvector-terminal-v1`,
  });
});

test("different pins isolate mutable terminal inputs while shared roots stay outside the namespace", () => {
  const root = "/terminal";
  const first = terminalPinEnvironment(root, firstPin, path.posix);
  const again = terminalPinEnvironment(root, firstPin, path.posix);
  const second = terminalPinEnvironment(root, secondPin, path.posix);
  assert.deepEqual(again, first);
  for (const key of Object.keys(first)) {
    assert.notEqual(first[key], second[key]);
    assert.match(first[key], new RegExp(`/pins/${firstPin}(?:/|$)`));
    assert.match(second[key], new RegExp(`/pins/${secondPin}(?:/|$)`));
  }
  for (const shared of ["weights", "metrics", "python", ".locks", "terminal-leases"]) {
    assert.ok(Object.values(first).every((value) => !value.includes(`/${shared}/`) && !value.endsWith(`/${shared}`)));
  }
});

test("path derivation never touches or falls back to a legacy fixed-root tree", async () => {
  const sandbox = await mkdtemp(path.join(tmpdir(), "starvector-pin-roots-"));
  const root = path.join(sandbox, "starvector-terminal");
  const legacyFiles = [
    path.join(root, "inference", "legacy-checkout.json"),
    path.join(root, "inference-preflight", "starvector-terminal-preflight.json"),
    path.join(root, "corpora", "starvector-terminal-v1", "starvector-terminal-row-index-v1.json"),
    path.join(sandbox, "terminal-leases", `starvector-terminal-${legacyB2dPin}.campaign.json`),
  ];
  const before = new Map();
  for (const [index, legacy] of legacyFiles.entries()) {
    const bytes = Buffer.from(`legacy-${legacyB2dPin}-${index}-must-remain-byte-identical\n`);
    await mkdir(path.dirname(legacy), { recursive: true });
    await writeFile(legacy, bytes);
    before.set(legacy, { bytes, inode: (await lstat(legacy)).ino });
  }
  const derived = terminalPinPaths(root, firstPin);
  for (const legacy of legacyFiles) {
    assert.deepEqual(await readFile(legacy), before.get(legacy).bytes);
    assert.equal((await lstat(legacy)).ino, before.get(legacy).inode);
  }
  assert.equal(await lstat(derived.pinRoot).catch((error) => error.code === "ENOENT" ? null : Promise.reject(error)), null);
  assert.notEqual(derived.inferenceRoot, path.join(root, "inference"));
  assert.notEqual(derived.preflightRoot, path.join(root, "inference-preflight"));
  assert.notEqual(derived.corpusAssetsRoot, path.join(root, "corpora", "starvector-terminal-v1"));
});

test("malformed or traversal-shaped pins and non-absolute roots fail closed", () => {
  for (const pin of ["", "a".repeat(39), "A".repeat(40), "g".repeat(40), "../" + "a".repeat(37), `${"a".repeat(40)}/..`, `a${"0".repeat(39)}\n`]) {
    assert.throws(() => validateTerminalPermanentPin(pin), /exact lowercase 40-hex commit/);
  }
  for (const root of ["relative", "../terminal", "/terminal\nGITHUB_ENV=pwned"]) {
    assert.throws(() => terminalPinPaths(root, firstPin, path.posix), /absolute single-line path/);
  }
});
