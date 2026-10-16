// Test-only host contracts, not production support declarations. Simulated Darwin
// telemetry still exercises real POSIX process groups/AF_UNIX peers on Linux.
import assert from "node:assert/strict";
import { statSync } from "node:fs";

export function platformFixtureSkip(capability, platform = process.platform) {
  const reasons = {
    "posix-processes": "Windows lacks the fixture's POSIX sessions/process groups, AF_UNIX peer credentials and /bin/ps process identities",
    "posix-process-identity": "Windows lacks the fixture's /bin/ps POSIX process-start identities",
    "posix-permissions": "Windows does not enforce the fixture's POSIX read-only file/directory mode seals",
  };
  assert.ok(Object.hasOwn(reasons, capability), `unknown fixture capability: ${capability}`);
  return platform === "win32" ? reasons[capability] : false;
}

export function externalFixtureSkip(configuredRoot, defaultRoot) {
  const root = configuredRoot ?? defaultRoot;
  let directory = false;
  try { directory = statSync(root).isDirectory(); }
  catch (error) { if (error.code !== "ENOENT") throw error; }
  if (directory) return false;
  if (configuredRoot !== undefined) throw new Error(`configured external inference fixture root is absent or not a directory: ${root}`);
  return `external inference integration not judged: optional default checkout absent: ${root}`;
}

export function assertFixtureChildSucceeded(result) {
  assert.equal(result.error, undefined, `fixture child could not start: ${result.error}`);
  assert.equal(result.signal, null, `fixture child terminated by ${result.signal}`);
  assert.equal(result.status, 0, `fixture child exit ${result.status}: ${result.stderr}`);
}
