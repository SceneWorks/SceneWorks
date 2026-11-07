import assert from "node:assert/strict";
import test from "node:test";
import { mkdtemp, readFile, rm, writeFile, readdir } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { Readable } from "node:stream";
import { fileURLToPath } from "node:url";
import { canonicalJson, legacySortedJson, legacySortedValue, legacyUtf16Json } from "./canonical-json.mjs";
import { sha256, fileSha256 } from "./file-sha256.mjs";
import { execFileBounded, DEFAULT_EXEC_TIMEOUT_MS } from "./exec.mjs";
import { canonicalJson as memoryJson } from "../memory-calibration-harness.mjs";
import { canonicalJson as kvJson, sha256 as kvSha256, numericSemanticSha256 } from "../kv-baseline-harness.mjs";
import { stableJson as closureJson, closureSha256 } from "../starvector-production-closure.mjs";
import { stable as terminalJson } from "../starvector-terminal-recovery.mjs";

const value = { "\u{10000}": "astral", "\ue000": "bmp", z: [{ b: 2, a: 1 }], "2": true, "10": null };
const utf8 = '{"10":null,"2":true,"z":[{"a":1,"b":2}],"\ue000":"bmp","\u{10000}":"astral"}';
const utf16 = '{"10":null,"2":true,"z":[{"a":1,"b":2}],"\u{10000}":"astral","\ue000":"bmp"}';
const jsObject = '{"2":true,"10":null,"z":[{"a":1,"b":2}],"\u{10000}":"astral","\ue000":"bmp"}';

test("canonical UTF8 and explicit legacy formats preserve numeric, non-ASCII, nested and array order", () => {
  assert.equal(canonicalJson(value), utf8);
  assert.equal(legacyUtf16Json(value), utf16);
  assert.equal(legacySortedJson(value), jsObject);
  assert.equal(JSON.stringify(legacySortedValue(value)), jsObject);
  assert.equal(legacySortedJson(undefined), undefined);
  const pretty = JSON.stringify(JSON.parse(jsObject), null, 2);
  assert.equal(memoryJson(value), `${pretty}\n`);
  assert.equal(kvJson(value), pretty);
  assert.equal(closureJson(value), utf8);
  assert.equal(terminalJson(value), utf16);
  assert.equal(kvSha256(value), sha256(pretty));
  // Fixed protocol digests: changing helpers must never cause evidence to be restamped.
  assert.equal(closureSha256([{ path: "a", sha256: "0".repeat(64), bytes: 1 }]), "fce5d0749a73be590e325c1ae6a7e2e648d20b754ca30c9b3e6cad1cb19e1e11");
  assert.equal(numericSemanticSha256({ x: 1, y: -0, z: [0.5] }), "67ea0e29d1368d36bf2b81b14755be0d481f7bf2cebff18d7d93fa1482cc014f");
});

test("exact-byte hash and streaming file owner agree across chunk boundaries and propagate read errors", async () => {
  assert.equal(sha256("abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  const bytes = Buffer.from([0, 255, 13, 10, 0]);
  assert.equal(await fileSha256("fixture", { openReadStream: (file, options) => {
    assert.equal(file, "fixture"); assert.equal(options.highWaterMark, 1024 * 1024);
    return Readable.from([bytes.subarray(0, 2), bytes.subarray(2)]);
  } }), sha256(bytes));
  const directory = await mkdtemp(path.join(os.tmpdir(), "script-sha-"));
  try {
    const file = path.join(directory, "bytes"); await writeFile(file, bytes);
    assert.equal(await fileSha256(file), sha256(bytes));
    await assert.rejects(fileSha256(path.join(directory, "missing")), { code: "ENOENT" });
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test("streaming file hashes preserve cancellation and custom per-chunk guards", async () => {
  const abort = new AbortController(); abort.abort();
  await assert.rejects(fileSha256("unused", { signal: abort.signal }), { name: "AbortError" });
  const active = new AbortController();
  let visited = 0;
  await assert.rejects(fileSha256("fixture", {
    signal: active.signal,
    openReadStream: (_file, options) => { assert.equal(options.signal, active.signal); return Readable.from([Buffer.from("first"), Buffer.from("second")]); },
    onChunk: () => { visited++; active.abort(); },
  }), { name: "AbortError" });
  assert.equal(visited, 1);
  const failure = new Error("existing deadline policy");
  await assert.rejects(fileSha256("fixture", {
    openReadStream: () => Readable.from([Buffer.from("bytes")]), onChunk: () => { throw failure; },
  }), error => error === failure);
});

test("bounded exec preserves argv/env/cwd, native promise.child and streamed output", async () => {
  const expected = ["a b", "$(literal)", "quoted\"argument"];
  const pending = execFileBounded(process.execPath, ["-e", "process.stdout.write(JSON.stringify({args:process.argv.slice(1),env:process.env.HELPER_TEST,cwd:process.cwd()}));process.stderr.write('diagnostic')", ...expected], {
    cwd: os.tmpdir(), env: { ...process.env, HELPER_TEST: "retained" }, timeout: 5000,
  });
  assert(pending.child?.pid > 0);
  const streamed = []; pending.child.stdout.on("data", (chunk) => streamed.push(chunk));
  const result = await pending;
  assert.deepEqual(JSON.parse(result.stdout), { args: expected, env: "retained", cwd: await import("node:fs/promises").then(fs => fs.realpath(os.tmpdir())) });
  assert.equal(result.stdout, Buffer.concat(streamed.map(chunk => Buffer.from(chunk))).toString());
  assert.equal(result.stderr, "diagnostic");
  const buffer = await execFileBounded(process.execPath, ["-e", "process.stdout.write(Buffer.from([0,255]))"], { encoding: "buffer", timeout: 5000 });
  assert.deepEqual(buffer.stdout, Buffer.from([0, 255]));
});

test("bounded exec preserves nonzero exit output, spawn errors, cancellation, output limit and timeout", async () => {
  await assert.rejects(execFileBounded(process.execPath, ["-e", "process.stdout.write('out');process.stderr.write('err');process.exit(7)"], { timeout: 5000 }), error => error.code === 7 && error.stdout === "out" && error.stderr === "err");
  await assert.rejects(execFileBounded(path.join(os.tmpdir(), "missing-script-helper-command"), [], { timeout: 5000 }), { code: "ENOENT" });
  const abort = new AbortController();
  const pending = execFileBounded(process.execPath, ["-e", "setInterval(()=>{},1000)"], { signal: abort.signal, timeout: 5000 });
  abort.abort(); await assert.rejects(pending, { name: "AbortError" });
  await assert.rejects(execFileBounded(process.execPath, ["-e", "process.stdout.write('x'.repeat(10000))"], { timeout: 5000, maxBuffer: 8 }), { code: "ERR_CHILD_PROCESS_STDIO_MAXBUFFER" });
  await assert.rejects(execFileBounded(process.execPath, ["-e", "setInterval(()=>{},1000)"], { timeout: 50, killSignal: "SIGKILL" }), error => error.killed && error.signal === "SIGKILL");
});

test("bounded defaults and options are forwarded exactly, with no implicit shell or environment merge", async () => {
  const env = { ONLY_THIS: "yes" }; const args = ["$literal"];
  await execFileBounded("tool", args, { env, cwd: "/cwd", windowsHide: true }, (file, argv, options) => {
    assert.equal(file, "tool"); assert.equal(argv, args); assert.equal(options.env, env);
    assert.deepEqual(options, { timeout: DEFAULT_EXEC_TIMEOUT_MS, maxBuffer: 1024 * 1024, env, cwd: "/cwd", windowsHide: true });
    return Promise.resolve({ stdout: "", stderr: "" });
  });
  for (const options of [{ timeout: 0 }, { timeout: Infinity }, { maxBuffer: -1 }, { maxBuffer: Infinity }]) {
    assert.throws(() => execFileBounded("tool", [], options), RangeError);
  }
});

test("explicit extra file descriptors survive the bounded exec adapter", async () => {
  const result = await execFileBounded(process.execPath, ["-e", "require('node:fs').writeSync(3,'descriptor');process.stdout.write('out')"], { timeout: 5000, stdio: ["ignore", "pipe", "pipe", "pipe"] });
  assert.equal(result.stdout, "out");
});

test("scripts do not reintroduce trivial exact-byte SHA256 wrappers", async () => {
  const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
  const duplicates = [];
  async function scan(directory) {
    for (const entry of await readdir(directory, { withFileTypes: true })) {
      const file = path.join(directory, entry.name);
      if (entry.isDirectory()) await scan(file);
      else if (entry.name.endsWith(".mjs") && !entry.name.endsWith(".test.mjs") && file !== path.join(root, "lib/file-sha256.mjs")) {
        const source = await readFile(file, "utf8");
        if (/(?:function\s+\w+\(\w+\)\s*\{\s*return|(?:const|let)\s+\w+\s*=\s*\(?\w+\)?\s*=>)\s*(?:crypto\.)?createHash\(["']sha256["']\)\.update\(\w+\)\.digest\(["']hex["']\)/.test(source)) duplicates.push(path.relative(root, file));
      }
    }
  }
  await scan(root); assert.deepEqual(duplicates, []);
});
