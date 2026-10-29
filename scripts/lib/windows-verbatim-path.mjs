// sc-26691 — Windows verbatim paths in product output.
//
// The product returns Windows paths in verbatim form (`\\?\E:\...`, `\\?\UNC\server\share\...`):
// Rust's std::fs::canonicalize. Node's path module treats that prefix as a different root, so a path
// the service genuinely wrote below an owned directory looked like an escape to the campaign
// harness. These normalize the SPELLING only; every confinement check still applies, unchanged, to
// the normalized result.
import { realpath } from 'node:fs/promises';

const VERBATIM_PREFIX = /^\\\\\?\\(UNC\\)?/i;

/** Lexical: drop a `\\?\` / `\\?\UNC\` verbatim prefix; any other value is returned untouched. */
export function stripVerbatim(value) {
  if (typeof value !== 'string') return value;
  const match = VERBATIM_PREFIX.exec(value);
  if (!match) return value;
  return match[1] ? `\\\\${value.slice(match[0].length)}` : value.slice(match[0].length);
}

/** The canonical (realpath) form of a verbatim path when it exists, otherwise its stripped form. */
export async function canonicalWindowsPath(value) {
  if (typeof value !== 'string' || !VERBATIM_PREFIX.test(value)) return value;
  try { return await realpath(value); }
  catch (error) { if (!['ENOENT', 'ENOTDIR'].includes(error.code)) throw error; return stripVerbatim(value); }
}
