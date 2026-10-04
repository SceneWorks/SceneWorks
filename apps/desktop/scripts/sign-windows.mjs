#!/usr/bin/env node
// Authenticode-sign ONE Windows file with Azure Artifact Signing (sc-1357).
//
// Tauri runs this once per file through `bundle.windows.signCommand`
// (windows-signing.release.conf.json): the main exe, the sceneworks-api sidecar, any
// bundled exe/dll resource that is not already signed, the NSIS plugins + uninstaller,
// and the finished -setup.exe / .msi. It drives Microsoft's own path — SignTool with
// the Artifact Signing dlib (`Microsoft.ArtifactSigning.Client`) — and authenticates
// through the dlib's DefaultAzureCredential from AZURE_TENANT_ID / AZURE_CLIENT_ID /
// AZURE_CLIENT_SECRET.
//
// The release workflow only passes that config when the signing secrets exist, so
// reaching this script without its environment is a misconfiguration: it fails the
// build rather than let an installer ship that silently lost its signature. Unsigned
// builds (PR lanes, local `tauri build`) simply never call it.
//
// Environment (prepared by the release workflow's "Prepare Artifact Signing" step):
//   ARTIFACT_SIGNING_DLIB      path to bin\x64\Azure.CodeSigning.Dlib.dll
//   ARTIFACT_SIGNING_METADATA  path to metadata.json (endpoint, account, profile)
//   ARTIFACT_SIGNING_SIGNTOOL  optional explicit signtool.exe; else newest x64 SDK copy
//   ARTIFACT_SIGNING_LOG       optional transcript file. Tauri only surfaces a sign
//                              command's output at --verbose, so a failed signature
//                              would otherwise read as a bare "failed to run node";
//                              the workflow prints this file when the build fails.

import { execFileSync } from "node:child_process";
import { appendFileSync, existsSync, readdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

// Microsoft's Artifact Signing RFC 3161 timestamp authority. Timestamping is what keeps
// a signature valid after the short-lived (3-day) signing certificate expires.
export const TIMESTAMP_URL = "http://timestamp.acs.microsoft.com";
const ATTEMPTS = 3;

export function readSigningEnv(env = process.env) {
  const missing = ["ARTIFACT_SIGNING_DLIB", "ARTIFACT_SIGNING_METADATA"].filter((name) => !env[name]);
  if (missing.length) {
    throw new Error(
      `Windows signing was requested but ${missing.join(", ")} ${missing.length === 1 ? "is" : "are"} unset. ` +
        "Run the release workflow's Prepare Artifact Signing step, or build without windows-signing.release.conf.json.",
    );
  }
  return { dlib: env.ARTIFACT_SIGNING_DLIB, metadata: env.ARTIFACT_SIGNING_METADATA, signtool: env.ARTIFACT_SIGNING_SIGNTOOL };
}

export function signtoolSignArgs({ dlib, metadata, file }) {
  return [
    "sign",
    "/v",
    "/fd", "SHA256",
    "/tr", TIMESTAMP_URL,
    "/td", "SHA256",
    "/dlib", dlib,
    "/dmdf", metadata,
    file,
  ];
}

// Newest x64 signtool.exe under a Windows Kits `bin` directory. The dlib is x64, and
// SignTool must match its architecture.
export function findSigntool(
  kitsBin = path.join(process.env["ProgramFiles(x86)"] ?? "C:\\Program Files (x86)", "Windows Kits", "10", "bin"),
  { exists = existsSync, list = readdirSync } = {},
) {
  if (!exists(kitsBin)) return null;
  const versions = list(kitsBin)
    .filter((name) => /^\d+(\.\d+)+$/.test(name))
    .sort((a, b) => compareVersions(b, a));
  for (const version of versions) {
    const candidate = path.join(kitsBin, version, "x64", "signtool.exe");
    if (exists(candidate)) return candidate;
  }
  return null;
}

function compareVersions(a, b) {
  const pa = a.split(".").map(Number);
  const pb = b.split(".").map(Number);
  for (let i = 0; i < Math.max(pa.length, pb.length); i += 1) {
    const diff = (pa[i] ?? 0) - (pb[i] ?? 0);
    if (diff) return diff;
  }
  return 0;
}

export async function signFile(file, { env = process.env, run = execFileSync, sleep = defaultSleep, log = transcript(env) } = {}) {
  try {
    if (!file) throw new Error("usage: sign-windows.mjs <file-to-sign>");
    const { dlib, metadata, signtool: explicit } = readSigningEnv(env);
    const signtool = explicit || findSigntool();
    if (!signtool) throw new Error("signtool.exe not found; install the Windows 10/11 SDK or set ARTIFACT_SIGNING_SIGNTOOL.");
    const exec = (args) => log(String(run(signtool, args, { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] }) ?? "").trim());

    // The signing service and the timestamp authority both see the occasional transient
    // failure; a release build signs dozens of files, so retry a few times before failing.
    for (let attempt = 1; ; attempt += 1) {
      try {
        exec(signtoolSignArgs({ dlib, metadata, file }));
        break;
      } catch (error) {
        const detail = [error.message, error.stdout, error.stderr].filter(Boolean).join("\n").trim();
        if (attempt >= ATTEMPTS) throw new Error(`Signing ${file} failed after ${ATTEMPTS} attempts:\n${detail}`);
        log(`Signing ${file} failed (attempt ${attempt}/${ATTEMPTS}); retrying.\n${detail}`);
        await sleep(5_000 * attempt);
      }
    }
    // Prove the result chains to a trusted root before Tauri packages it.
    exec(["verify", "/pa", "/q", file]);
    log(`Signed ${file}`);
  } catch (error) {
    log(`ERROR: ${error.message}`);
    throw error;
  }
}

// Log to stdout (Tauri's --verbose view) and, when ARTIFACT_SIGNING_LOG is set, to the
// transcript the release workflow prints after a failed build.
function transcript(env) {
  return (message) => {
    if (!message) return;
    console.log(message);
    if (env.ARTIFACT_SIGNING_LOG) appendFileSync(env.ARTIFACT_SIGNING_LOG, `${message}\n`);
  };
}

function defaultSleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  signFile(process.argv[2]).catch(() => process.exit(1));
}
