// sc-26691: host-process hygiene for the H3 terminal campaign's owned services.
//
// MLX run 41044021373 failed its first case before any scenario with `listen EADDRINUSE
// 127.0.0.1:43210`: every case bound the same operator-chosen port, and a process left behind on
// the same Mac (an earlier, cancelled run's owned service, or another job) still held it. The
// runtime now binds an OS-assigned port per service epoch; this module holds the parts the runtime
// and the workflow's prior-run cleanup share:
//
// 1. Port conflicts are diagnosed, never guessed at: a bind failure names the actual listener
//    (`lsof` on macOS/Linux, `netstat -ano` on Windows) in the startup failure.
// 2. Before a case starts, the harness reads the `ownership-<epoch>.json` records earlier runs
//    (and earlier cases of this run) left under the shared `sc26691-private` parent and checks
//    whether any recorded PID is still alive AS THAT PROCESS - matched by its command and its OS
//    start time, so a PID the OS has since reused for something else is not misattributed. A live
//    owned process refuses the case with its run, case, PID and command. It is never killed: it
//    may hold Metal/CUDA memory, and stopping it is the operator's decision.
import { execFile } from 'node:child_process';
import { readdir, readFile } from 'node:fs/promises';
import path from 'node:path';

/**
 * A case service's requested API port. `0` asks the OS for a free loopback port at bind time: the
 * product API binds it and logs the bound address (`api_listening`), so there is no check-then-bind
 * window and no well-known port another process can already hold. Bootstrap plans always use it;
 * a fixed port (1025..65535) is accepted only so archived plans and evidence still validate.
 */
export const OS_ASSIGNED_PORT = 0;
export function isServicePort(port) { return Number.isInteger(port) && (port === OS_ASSIGNED_PORT || (port > 1024 && port <= 65535)); }
/** The port from the API's own `api_listening` JSON log line (flat envelope or `fields`), if logged yet. */
export function boundApiPort(logText) {
  for (const line of String(logText).split(/\r?\n/)) {
    if (!line.includes("api_listening")) continue;
    let event; try { event = JSON.parse(line); } catch { continue; }
    const fields = event.fields ?? event;
    if (fields.event !== "api_listening") continue;
    const match = /:(\d+)$/.exec(String(fields.address ?? ""));
    const port = match ? Number(match[1]) : NaN;
    if (port > 0 && port <= 65535) return port;
  }
  return undefined;
}

export const PRIVATE_PARENT = 'sc26691-private';
/** Output roots the bootstrap derives beside each run's private staging root. */
export const RUN_OUTPUT_ROOT = /^(\d+)-(\d+)-bootstrap-[0-9a-f]{40}$/;
/** `ps etime` has one-second resolution; allow that rounding and clock skew on both sides. */
export const START_TIME_TOLERANCE_MS = 5000;
// Records written before per-process spawn windows existed carry only `started` (taken right after
// the API spawn). The guardian and API are spawned immediately before it; the worker only after the
// API reports ready, which the runtime bounds at 60 s, so its window extends past `started`.
const LEGACY_WINDOWS = { guardian: [-30000, 5000], api: [-30000, 5000], worker: [-5000, 180000] };
/** What each owned role's command line must contain to be the process the record names. */
export const ROLE_MARKERS = { api: /sceneworks-rust-api/, worker: /sceneworks-rust-api/, guardian: /process\.stdin\.resume/ };

/** execFile that reports a non-zero exit as data: `ps`/`lsof` exit 1 when nothing matches. */
export function execQuiet(file, args, options = {}) {
  return new Promise(resolve => execFile(file, args, { encoding: 'utf8', windowsHide: true, maxBuffer: 4 * 1024 * 1024, ...options }, (error, stdout, stderr) => {
    const spawnFailure = error && typeof error.code !== 'number';
    resolve({ code: error ? (spawnFailure ? -1 : error.code) : 0, stdout: stdout ?? '', stderr: stderr ?? '', ...(spawnFailure ? { error: error.message } : {}) });
  }));
}

/** `ps -o etime` -> seconds: `[[dd-]hh:]mm:ss`. */
export function parseEtimeSeconds(text) {
  const match = /^\s*(?:(\d+)-)?(?:(\d+):)?(\d+):(\d+)\s*$/.exec(String(text));
  if (!match) return null;
  const [, days = 0, hours = 0, minutes, seconds] = match;
  return ((Number(days) * 24 + Number(hours)) * 60 + Number(minutes)) * 60 + Number(seconds);
}

/**
 * The live process behind `pid`, or null when none is running: `{ pid, command, startedAt }` with
 * `startedAt` in epoch milliseconds. macOS/Linux read `ps`; Windows reads Win32_Process.
 */
export async function processInfo(pid, { platform = process.platform, exec = execQuiet, now = Date.now } = {}) {
  if (!Number.isSafeInteger(pid) || pid <= 1) return null;
  if (platform === 'win32') {
    const script = `$p=Get-CimInstance Win32_Process -Filter "ProcessId=${pid}";if($p){$p.CreationDate.ToUniversalTime().ToString('o');$p.CommandLine}`;
    const result = await exec('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command', script]);
    const [created, ...command] = result.stdout.split(/\r?\n/);
    const startedAt = Date.parse(created?.trim() ?? '');
    return result.code === 0 && Number.isFinite(startedAt) ? { pid, command: command.join(' ').trim(), startedAt } : null;
  }
  const result = await exec('ps', ['-ww', '-o', 'etime=', '-o', 'command=', '-p', String(pid)]);
  const line = result.stdout.split('\n').find(value => value.trim());
  if (result.code !== 0 || !line) return null;
  const [, etime, command = ''] = /^\s*(\S+)\s+(.*)$/.exec(line) ?? [];
  const elapsed = parseEtimeSeconds(etime ?? '');
  return elapsed === null ? null : { pid, command: command.trim(), startedAt: now() - elapsed * 1000 };
}

/** `lsof -nP -iTCP:<port> -sTCP:LISTEN` rows -> `[{ command, pid, user, name }]`. */
export function parseLsofListen(text) {
  return String(text).split('\n').slice(1).map(line => line.trim().split(/\s+/)).filter(cols => cols.length >= 9 && /^\d+$/.test(cols[1]))
    .map(cols => ({ command: cols[0], pid: Number(cols[1]), user: cols[2], name: cols.slice(8).join(' ') }));
}
/** `netstat -ano -p TCP` rows LISTENING on `port` -> `[{ pid, local }]`. */
export function parseNetstatListen(text, port) {
  return String(text).split(/\r?\n/).map(line => line.trim().split(/\s+/))
    .filter(cols => cols.length >= 5 && /^TCP/i.test(cols[0]) && cols[1].endsWith(`:${port}`) && /^LISTEN/i.test(cols[3]) && /^\d+$/.test(cols[4]))
    .map(cols => ({ pid: Number(cols[4]), local: cols[1] }));
}

/**
 * Who is listening on a loopback TCP port, with the raw tool output kept for the evidence.
 * Never throws: an unavailable tool is reported as such, not as "no holder".
 */
export async function portHolder(port, { platform = process.platform, exec = execQuiet, info = processInfo } = {}) {
  const [tool, args] = platform === 'win32' ? ['netstat', ['-ano', '-p', 'TCP']] : ['lsof', ['-nP', `-iTCP:${port}`, '-sTCP:LISTEN']];
  const result = await exec(tool, args);
  const rows = platform === 'win32' ? parseNetstatListen(result.stdout, port) : parseLsofListen(result.stdout);
  const holders = [];
  for (const pid of [...new Set(rows.map(row => row.pid))]) {
    let live = null;
    try { live = await info(pid, { platform, exec }); } catch { /* the listing row still names the PID */ }
    holders.push({ pid, command: live?.command ?? rows.find(row => row.pid === pid).command ?? null, ...(live ? { startedAt: new Date(live.startedAt).toISOString() } : {}) });
  }
  const output = platform === 'win32' ? result.stdout.split(/\r?\n/).filter(line => new RegExp(`:${port}\\s`).test(line)).join('\n') : result.stdout.trim();
  return { port, tool: [tool, ...args].join(' '), exitCode: result.code, ...(result.error ? { toolError: result.error } : {}), output, holders };
}
/** One line naming the holder(s), for an error message. */
export function formatPortHolder(diagnosis) {
  if (!diagnosis) return 'port holder not diagnosed';
  if (diagnosis.toolError) return `port ${diagnosis.port} holder unknown (${diagnosis.tool}: ${diagnosis.toolError})`;
  if (!diagnosis.holders.length) return `no listener on port ${diagnosis.port} reported by ${diagnosis.tool}`;
  return `port ${diagnosis.port} is held by ${diagnosis.holders.map(h => `pid ${h.pid} (${h.command ?? 'command unknown'})`).join(', ')}`;
}
/** Whether an error is a TCP bind conflict worth diagnosing. */
export function isBindFailure(error) {
  return error?.code === 'EADDRINUSE' || (error?.code === 'EACCES' && error?.syscall === 'listen') || /EADDRINUSE|address already in use|address in use|failed to bind/i.test(String(error?.message ?? ''));
}

/** The processes an ownership record names, each with the identity a live PID must match. */
export function recordedProcesses(record) {
  if (Array.isArray(record?.processes)) {
    return record.processes.filter(p => ROLE_MARKERS[p.role] && Number.isSafeInteger(p.pid)).map(p => ({
      role: p.role, pid: p.pid, executable: p.executable ?? null,
      window: [Date.parse(p.spawnedFrom) - START_TIME_TOLERANCE_MS, Date.parse(p.spawnedTo) + START_TIME_TOLERANCE_MS],
    }));
  }
  const started = Date.parse(record?.started);
  return [['guardian', record?.guardianPid], ['api', record?.apiPid], ['worker', record?.workerPid]].filter(([, pid]) => Number.isSafeInteger(pid))
    .map(([role, pid]) => ({ role, pid, executable: null, window: [started + LEGACY_WINDOWS[role][0], started + LEGACY_WINDOWS[role][1]] }));
}
/** Whether a live process is the one a record names, not a later process that reused its PID. */
export function sameProcess(expected, live) {
  if (!live || live.pid !== expected.pid) return false;
  const command = String(live.command ?? '');
  if (!ROLE_MARKERS[expected.role].test(command)) return false;
  // ps/CIM report argv[0] as spawned: the absolute binary (or node) path the record names.
  if (expected.executable && !command.includes(expected.executable)) return false;
  const [from, to] = expected.window;
  return Number.isFinite(from) && Number.isFinite(to) && live.startedAt >= from && live.startedAt <= to;
}

/**
 * The run roots to inspect for `output`: every bootstrap output root under the shared
 * `sc26691-private` parent, or just `output` itself when it is not under that parent (a unit_fake
 * temp dir must never scan its siblings in the OS temp directory).
 */
export async function campaignRunRoots(output) {
  const resolved = path.resolve(output), parent = path.dirname(resolved);
  if (path.basename(parent) !== PRIVATE_PARENT) return [resolved];
  const entries = await readdir(parent, { withFileTypes: true }).catch(error => { if (error.code === 'ENOENT') return []; throw error; });
  const roots = entries.filter(entry => entry.isDirectory() && RUN_OUTPUT_ROOT.test(entry.name)).map(entry => path.join(parent, entry.name));
  return roots.includes(resolved) ? roots : [...roots, resolved];
}

/**
 * Run parents an older revision wrote on the WEIGHTS volume (E:\huggingface\sc26691-private on
 * Windows; <HF_HOME>/sc26691-private on macOS) before campaign writes moved to the work root. A
 * service those runs spawned can still be alive, so the leftover check reads their ownership
 * records too. READ-ONLY: these parents are only listed and read, never created, written or cleared.
 */
export function legacyWeightsRunParents({ platform = process.platform, env = process.env } = {}) {
  if (platform === 'win32') return [path.win32.join('E:\\', 'huggingface', PRIVATE_PARENT)];
  if (platform === 'darwin' && env.HF_HOME && path.isAbsolute(env.HF_HOME)) return [path.join(env.HF_HOME, PRIVATE_PARENT)];
  return [];
}
/** The bootstrap run roots under the legacy parents, by listing only (a missing parent is none). */
export async function legacyWeightsRunRoots({ parents = legacyWeightsRunParents(), exclude = [] } = {}) {
  const skip = new Set(exclude.map(value => path.resolve(value))), roots = [];
  for (const parent of parents) {
    const entries = await readdir(parent, { withFileTypes: true }).catch(error => { if (['ENOENT', 'ENOTDIR'].includes(error.code)) return []; throw error; });
    for (const entry of entries) if (entry.isDirectory() && RUN_OUTPUT_ROOT.test(entry.name) && !skip.has(path.resolve(parent, entry.name))) roots.push(path.join(parent, entry.name));
  }
  return roots;
}

/** Every `<root>/<case>/ownership-<epoch>.json` under the given run roots. */
export async function ownershipRecords(roots, { exclude = [] } = {}) {
  const skip = new Set(exclude.map(value => path.resolve(value))), records = [];
  for (const root of roots) {
    const run = RUN_OUTPUT_ROOT.exec(path.basename(root));
    const cases = await readdir(root, { withFileTypes: true }).catch(error => { if (['ENOENT', 'ENOTDIR'].includes(error.code)) return []; throw error; });
    for (const entry of cases.filter(value => value.isDirectory())) {
      const dir = path.join(root, entry.name);
      if (skip.has(path.resolve(dir))) continue;
      const files = (await readdir(dir).catch(() => [])).filter(name => /^ownership-\d+\.json$/.test(name)).sort();
      for (const name of files) {
        const file = path.join(dir, name);
        let record;
        try { record = JSON.parse(await readFile(file, 'utf8')); } catch { continue; } // A torn record names nothing verifiable.
        records.push({ file, runRoot: root, runId: run?.[1] ?? null, runAttempt: run?.[2] ?? null, caseId: entry.name, record });
      }
    }
  }
  return records;
}

/**
 * Owned processes from earlier runs/cases that are still alive as themselves. Each finding names
 * run, case, epoch, role, PID, command and start time. Read-only: nothing is signalled.
 */
export async function liveOwnedProcesses({ roots, exclude = [], info = processInfo, platform = process.platform } = {}) {
  const found = [];
  for (const entry of await ownershipRecords(roots, { exclude })) {
    for (const expected of recordedProcesses(entry.record)) {
      const live = await info(expected.pid, { platform });
      if (!sameProcess(expected, live)) continue;
      found.push({ runId: entry.runId, runAttempt: entry.runAttempt, caseId: entry.caseId, epoch: entry.record.epoch ?? null, role: expected.role, pid: expected.pid, command: live.command, startedAt: new Date(live.startedAt).toISOString(), ownershipFile: entry.file, ...(entry.record.port !== undefined ? { port: entry.record.port } : {}) });
    }
  }
  return found;
}
export function formatLeftovers(found) {
  return found.map(p => `run ${p.runId ?? 'unknown'}${p.runAttempt ? `-${p.runAttempt}` : ''} case ${p.caseId} epoch ${p.epoch} ${p.role} pid ${p.pid} (${p.command}, started ${p.startedAt})`).join('; ');
}
