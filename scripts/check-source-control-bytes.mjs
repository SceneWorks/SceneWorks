#!/usr/bin/env node

/**
 * Reject control bytes that can make reviewable source look binary to Git.
 *
 * SC-16056 reached a green `npm run check` with a literal NUL in an .mjs file.
 * Git's repository-wide `* text=auto` rule uses that NUL to classify the file
 * as binary, so the pull-request patch disappeared. This gate exists to make
 * that incident a build failure even if the explicit source `text` attributes
 * are later bypassed or removed.
 *
 * Selection is deterministic rather than content-sniffed: scan regular files
 * tracked in Git's index when their exact final extension is in
 * SOURCE_EXTENSIONS or their exact basename is in SOURCE_BASENAMES. Binary formats
 * are absent by construction. Within selected files, TAB (0x09), LF (0x0a),
 * and CR (0x0d) are the only allowed C0 controls; every other byte below 0x20
 * is rejected and reported at its zero-based byte offset.
 */

import { execFileSync } from "node:child_process";
import { lstatSync, readFileSync } from "node:fs";
import { basename, extname, resolve } from "node:path";
import { pathToFileURL } from "node:url";

export const SOURCE_EXTENSIONS = new Set([
  ".bat",
  ".c",
  ".cc",
  ".cjs",
  ".cmd",
  ".conf",
  ".cpp",
  ".css",
  ".csv",
  ".Dockerfile",
  ".dockerfile",
  ".env",
  ".example",
  ".gitattributes",
  ".gitignore",
  ".gitkeep",
  ".go",
  ".GPLv3",
  ".gplv3",
  ".graphql",
  ".h",
  ".hpp",
  ".html",
  ".ini",
  ".java",
  ".js",
  ".json",
  ".jsonc",
  ".jsx",
  ".kt",
  ".kts",
  ".lock",
  ".md",
  ".mdx",
  ".mjs",
  ".mts",
  ".plist",
  ".ps1",
  ".py",
  ".rs",
  ".scss",
  ".sh",
  ".snap",
  ".sql",
  ".svelte",
  ".svg",
  ".toml",
  ".ts",
  ".tsv",
  ".tsx",
  ".txt",
  ".vue",
  ".xml",
  ".yaml",
  ".yml",
]);

export const SOURCE_BASENAMES = new Set([
  ".dockerignore",
  ".env",
  ".gitattributes",
  ".gitignore",
  ".gitkeep",
  "Containerfile",
  "Dockerfile",
  "Justfile",
  "LICENSE",
  "Makefile",
  "pre-commit",
  "pre-push",
]);

const ALLOWED_C0_BYTES = new Set([0x09, 0x0a, 0x0d]);

export function isTrackedSourcePath(file) {
  return (
    SOURCE_EXTENSIONS.has(extname(file)) || SOURCE_BASENAMES.has(basename(file))
  );
}

export function findUnexpectedControlBytes(contents) {
  const findings = [];
  for (let offset = 0; offset < contents.length; offset += 1) {
    const byte = contents[offset];
    if (byte < 0x20 && !ALLOWED_C0_BYTES.has(byte)) {
      findings.push({ byte, offset });
    }
  }
  return findings;
}

function trackedPaths(cwd) {
  const output = execFileSync("git", ["ls-files", "--cached", "-z"], {
    cwd,
    encoding: "buffer",
    stdio: ["ignore", "pipe", "pipe"],
  });
  return output
    .toString("utf8")
    .split("\0")
    .filter(Boolean);
}

export function scanTrackedSourceFiles(cwd = process.cwd()) {
  const files = trackedPaths(cwd).filter(isTrackedSourcePath);
  const violations = [];
  let scannedFileCount = 0;

  for (const file of files) {
    const absolutePath = resolve(cwd, file);
    let metadata;
    try {
      metadata = lstatSync(absolutePath);
    } catch (error) {
      if (error?.code === "ENOENT") continue;
      throw error;
    }

    // Do not follow tracked symlinks into content outside the repository.
    if (!metadata.isFile()) continue;

    scannedFileCount += 1;
    for (const finding of findUnexpectedControlBytes(readFileSync(absolutePath))) {
      violations.push({ file, ...finding });
    }
  }

  return { scannedFileCount, violations };
}

function formatByte(byte) {
  return `0x${byte.toString(16).padStart(2, "0")}`;
}

function main() {
  const { scannedFileCount, violations } = scanTrackedSourceFiles();
  if (violations.length === 0) {
    console.log(
      `Source control-byte check passed (${scannedFileCount} tracked source files; TAB/LF/CR allowed).`,
    );
    return;
  }

  console.error(
    `Unexpected C0 control bytes found in tracked source files (${violations.length} violation(s)).`,
  );
  for (const { byte, file, offset } of violations) {
    console.error(`- ${file}: byte offset ${offset}: ${formatByte(byte)}`);
  }
  console.error("Allowed C0 bytes are TAB (0x09), LF (0x0a), and CR (0x0d) only.");
  process.exitCode = 1;
}

if (process.argv[1] && pathToFileURL(resolve(process.argv[1])).href === import.meta.url) {
  main();
}
