import { createHash } from "node:crypto";

const SHA256 = /^[a-f0-9]{64}$/;
const ENTRY_KEYS = ["path", "byte_size", "sha256"];

function invalid(message) {
  throw new Error(`terminal tree identity: ${message}`);
}

export function terminalTreeEntry(relativePath, byteSize, sha256) {
  const entry = { path: relativePath, byte_size: byteSize, sha256 };
  validateTerminalTreeEntry(entry);
  return entry;
}

export function sortTerminalTreeEntries(entries) {
  if (!Array.isArray(entries)) invalid("entries must use the object-array representation");
  return [...entries].sort((left, right) => left.path < right.path ? -1 : left.path > right.path ? 1 : 0);
}

export function terminalTreeSha256(entries) {
  if (!Array.isArray(entries)) invalid("entries must use the object-array representation");
  let previous;
  for (const entry of entries) {
    validateTerminalTreeEntry(entry);
    if (previous !== undefined && previous >= entry.path) invalid("entries must be uniquely sorted by portable path");
    previous = entry.path;
  }
  return createHash("sha256").update(JSON.stringify(entries)).digest("hex");
}

function validateTerminalTreeEntry(entry) {
  if (!entry || typeof entry !== "object" || Array.isArray(entry) || JSON.stringify(Object.keys(entry)) !== JSON.stringify(ENTRY_KEYS)) {
    invalid("entry must be exactly {path, byte_size, sha256}");
  }
  if (typeof entry.path !== "string" || !entry.path || entry.path.includes("\\") || entry.path.includes("\0") || entry.path.startsWith("/") || entry.path.split("/").some((part) => !part || part === "." || part === "..")) {
    invalid("entry path must be a safe portable relative path");
  }
  if (!Number.isSafeInteger(entry.byte_size) || entry.byte_size < 0) invalid("entry byte_size must be a non-negative safe integer");
  if (!SHA256.test(entry.sha256 ?? "")) invalid("entry sha256 must be lowercase SHA-256");
}
