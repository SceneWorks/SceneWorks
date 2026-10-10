import assert from "node:assert/strict";
import { mkdtempSync, readFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import { TIMESTAMP_URL, findSigntool, readSigningEnv, signFile, signtoolSignArgs } from "./sign-windows.mjs";

const here = path.dirname(fileURLToPath(import.meta.url));
const ENV = { ARTIFACT_SIGNING_DLIB: "C:\\asc\\Azure.CodeSigning.Dlib.dll", ARTIFACT_SIGNING_METADATA: "C:\\asc\\metadata.json", ARTIFACT_SIGNING_SIGNTOOL: "C:\\sdk\\signtool.exe" };

test("signs with SHA256, the Artifact Signing timestamp authority, and the dlib", () => {
  assert.deepEqual(signtoolSignArgs({ dlib: "d.dll", metadata: "m.json", file: "a.exe" }), [
    "sign", "/v", "/fd", "SHA256", "/tr", TIMESTAMP_URL, "/td", "SHA256", "/dlib", "d.dll", "/dmdf", "m.json", "a.exe",
  ]);
});

test("refuses to run without the prepared signing environment instead of shipping unsigned", () => {
  assert.throws(() => readSigningEnv({}), /ARTIFACT_SIGNING_DLIB, ARTIFACT_SIGNING_METADATA are unset/);
  assert.throws(() => readSigningEnv({ ARTIFACT_SIGNING_DLIB: "d" }), /ARTIFACT_SIGNING_METADATA is unset/);
});

test("signs then verifies the file", async () => {
  const calls = [];
  await signFile("app.exe", { env: ENV, run: (cmd, args) => calls.push([cmd, args[0]]), log: () => {} });
  assert.deepEqual(calls, [["C:\\sdk\\signtool.exe", "sign"], ["C:\\sdk\\signtool.exe", "verify"]]);
});

test("retries a transient signing failure, then gives up after three attempts", async () => {
  let signs = 0;
  const flaky = (cmd, args) => {
    if (args[0] === "sign" && (signs += 1) < 2) throw new Error("timestamp server busy");
  };
  await signFile("app.exe", { env: ENV, run: flaky, sleep: async () => {}, log: () => {} });
  assert.equal(signs, 2);

  const broken = (cmd, args) => {
    if (args[0] === "sign") throw new Error("403 Forbidden");
  };
  await assert.rejects(
    signFile("app.exe", { env: ENV, run: broken, sleep: async () => {}, log: () => {} }),
    /failed after 3 attempts:\n403 Forbidden/,
  );
});

test("records failures in the transcript the workflow prints, since Tauri hides the output", async () => {
  const logFile = path.join(mkdtempSync(path.join(os.tmpdir(), "sign-windows-")), "sign.log");
  const env = { ...ENV, ARTIFACT_SIGNING_LOG: logFile };
  const quiet = console.log;
  console.log = () => {};
  try {
    const broken = () => {
      throw Object.assign(new Error("Command failed"), { stdout: "SignTool Error: 403 Forbidden from eus" });
    };
    await assert.rejects(signFile("app.exe", { env, run: broken, sleep: async () => {} }));
    await assert.rejects(signFile("app.exe", { env: { ARTIFACT_SIGNING_LOG: logFile } }));
  } finally {
    console.log = quiet;
  }
  const transcriptText = readFileSync(logFile, "utf8");
  assert.match(transcriptText, /ERROR: Signing app\.exe failed after 3 attempts:[\s\S]*403 Forbidden from eus/);
  assert.match(transcriptText, /ERROR: Windows signing was requested but ARTIFACT_SIGNING_DLIB, ARTIFACT_SIGNING_METADATA are unset/);
});

test("finds the newest x64 SDK signtool", () => {
  const bin = path.join("K", "bin");
  const present = new Set([bin, path.join(bin, "10.0.22621.0", "x64", "signtool.exe"), path.join(bin, "10.0.19041.0", "x64", "signtool.exe")]);
  const found = findSigntool(bin, {
    exists: (p) => present.has(p),
    list: () => ["10.0.19041.0", "10.0.22621.0", "x86", "10.0.26100.0"],
  });
  assert.equal(found, path.join(bin, "10.0.22621.0", "x64", "signtool.exe"));
  assert.equal(findSigntool(bin, { exists: () => false, list: () => [] }), null);
});

test("the release config routes every Windows signature through this script", () => {
  const config = JSON.parse(readFileSync(path.join(here, "..", "windows-signing.release.conf.json"), "utf8"));
  assert.deepEqual(config.bundle.windows.signCommand, { cmd: "node", args: ["scripts/sign-windows.mjs", "%1"] });
});
