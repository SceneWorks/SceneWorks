#!/usr/bin/env node
// Canonical secondary-Mac prerequisite only. No global installation, compiler shim or ELF execution.
import { createHash } from "node:crypto";
import { createReadStream, openSync, closeSync, readSync, writeSync, writeFileSync, readFileSync, statSync, lstatSync, realpathSync, mkdirSync, mkdtempSync, appendFileSync, rmSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { executeCpuCommand } from "./moss-consumer-assertions.mjs";

export const TOOLCHAIN = Object.freeze({
  repository: "messense/homebrew-macos-cross-toolchains", version: "15.2.0", asset_id: 356184618,
  url: "https://github.com/messense/homebrew-macos-cross-toolchains/releases/download/v15.2.0/x86_64-unknown-linux-gnu-aarch64-darwin.tar.gz",
  bytes: 202165149, sha256: "04746819d162f829dacf2ea6d6947b9fb4141ffbd91588f0bfbaab4c2c7e1a30",
});
export const LIMITS = Object.freeze({ bootstrap_ms: 300000, download_ms: 120000, command_ms: 30000, archive_bytes: 256 * 1024 ** 2, members: 10000, expanded_bytes: 1024 ** 3 });
export const EXISTING_DIRS = Object.freeze([
  "/opt/homebrew/bin", "/opt/homebrew/opt/x86_64-unknown-linux-gnu/bin", "/opt/homebrew/opt/x86_64-unknown-linux-gnu/toolchain/bin",
  "/usr/local/bin", "/usr/local/opt/x86_64-unknown-linux-gnu/bin", "/usr/local/opt/x86_64-unknown-linux-gnu/toolchain/bin",
]);
export const TOOL_KEYS = Object.freeze(["cc", "cxx", "ar", "ranlib", "as", "ld", "cc1", "cc1plus", "collect2"]);
const TARGET = "x86_64-unknown-linux-gnu", HOST = "aarch64-apple-darwin", ROOT = TARGET;
const fail = message => { throw new Error(`canonical cross toolchain: ${message}`); };
const requireFact = (value, message) => { if (!value) fail(message); };
const hash = value => createHash("sha256").update(value).digest("hex");
const digestPattern = /^[0-9a-f]{64}$/;
const absolute = value => typeof value === "string" && path.isAbsolute(value) && !/[\r\n\0]/.test(value);
const inside = (root, file) => file.startsWith(root + path.sep);
const targetMatches = value => value === TARGET || value === "x86_64-linux-gnu";
export const identityDigest = identity => hash(JSON.stringify(identity));
export async function fileDigest(file) { const h = createHash("sha256"); for await (const bytes of createReadStream(file)) h.update(bytes); return h.digest("hex"); }

export async function downloadArchive(destination, { source = TOOLCHAIN, fetchImpl = fetch, timeout = LIMITS.download_ms } = {}) {
  requireFact(source.url.startsWith("https://github.com/messense/homebrew-macos-cross-toolchains/releases/download/v"), "untrusted archive origin");
  const response = await fetchImpl(source.url, { signal: AbortSignal.timeout(Math.max(1, Math.floor(timeout))), redirect: "follow" });
  requireFact(response.ok && response.body, "archive download failed");
  const length = response.headers?.get("content-length");
  requireFact(!length || (/^[0-9]+$/.test(length) && Number(length) <= LIMITS.archive_bytes), "archive content length exceeds bound");
  const fd = openSync(destination, "wx", 0o600), h = createHash("sha256"); let total = 0;
  try {
    for await (const bytes of response.body) { total += bytes.length; requireFact(total <= LIMITS.archive_bytes, "archive download exceeds bound"); h.update(bytes); writeSync(fd, bytes); }
  } finally { closeSync(fd); }
  requireFact(total === source.bytes && h.digest("hex") === source.sha256, "archive size or SHA-256 mismatch");
}

// Two passes: authenticate/validate ALL metadata before writing any member. Links are restricted
// to this archive's single root; nothing may be written through a link ancestor. Python is isolated.
const EXTRACT = String.raw`import sys,tarfile,pathlib,posixpath,hashlib,shutil,json,gzip
archive,dest,digest,count_cap,byte_cap,mode=sys.argv[1:]
out=pathlib.Path(dest)
assert not out.is_symlink() and out.is_dir() and not any(out.iterdir()), 'destination must be new and empty'
h=hashlib.sha256()
with open(archive,'rb') as f:
 for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
assert h.hexdigest()==digest, 'archive digest mismatch before extraction'
root='x86_64-unknown-linux-gnu'
class Limited:
 def __init__(self,stream):self.stream=stream;self.cap=int(byte_cap)+64*1024*1024
 def tell(self):return self.stream.tell()
 def read(self,size):
  assert 0<=size<=4*1024*1024 and self.tell()+size<=self.cap, 'archive metadata/raw expansion bound'
  return self.stream.read(size)
 def seek(self,offset,whence=0):
  assert whence in (0,1), 'unbounded archive seek'
  target=offset if whence==0 else self.tell()+offset
  assert 0<=target<=self.cap, 'archive raw expansion bound'
  return self.stream.seek(target)
with gzip.open(archive,'rb') as payload,tarfile.open(fileobj=Limited(payload),mode='r:') as t:
 members=[];names={};links={};total=0
 for m in t:
  assert len(members)<int(count_cap), 'archive member bound'
  name=m.name.rstrip('/')
  assert name and len(name)<4096 and '\\' not in name and '\x00' not in name and not name.startswith('/'), 'unsafe member name'
  assert all(p not in ('','.','..') for p in name.split('/')), 'unsafe member component'
  assert name==root or name.startswith(root+'/'), 'wrong archive root'
  assert name not in names, 'duplicate archive member'
  assert m.isdir() or m.isfile() or m.issym() or m.islnk(), 'special archive member'
  assert not m.issparse() and m.size>=0 and m.size<=256*1024*1024 and not (m.mode & 0o6000), 'unsafe member size or permission'
  total+=m.size;assert total<=int(byte_cap), 'archive expansion bound'
  if m.issym() or m.islnk():
   assert m.linkname and not m.linkname.startswith('/') and '\\' not in m.linkname and '\x00' not in m.linkname, 'unsafe link name'
   target=posixpath.normpath(posixpath.join(posixpath.dirname(name),m.linkname) if m.issym() else m.linkname)
   assert target==root or target.startswith(root+'/'), 'link escapes archive root'
   links[name]=target
  names[name]=m;members.append(m)
 for name in names:
  parts=name.split('/')
  assert not any('/'.join(parts[:i]) in links for i in range(1,len(parts))), 'member below link ancestor'
 def resolve(name,seen):
  assert name not in seen, 'archive link cycle'
  if name in links:return resolve(links[name],seen|{name})
  parts=name.split('/')
  for i in range(1,len(parts)):
   parent='/'.join(parts[:i])
   if parent in links:return resolve(posixpath.join(links[parent],*parts[i:]),seen|{parent})
  assert name in names, 'missing link target'
  return name
 for name in links:
  target=resolve(name,set())
  if names[name].islnk():assert names[target].isfile(), 'hard link must reference regular data'
 assert shutil.disk_usage(out).free>=total+256*1024*1024, 'insufficient owned extraction space'
 assert mode in ('inspect','extract'), 'invalid extraction mode'
 if mode=='extract':t.extractall(out,members=members,filter='data')
 print(json.dumps({'members':len(members),'expanded_bytes':total}))`;

export async function extractArchive(archive, destination, { digest = TOOLCHAIN.sha256, run, inspectOnly = false } = {}) {
  requireFact(statSync(archive).size <= LIMITS.archive_bytes && await fileDigest(archive) === digest, "archive digest mismatch before extraction");
  requireFact(!lstatSync(destination).isSymbolicLink(), "symlink extraction destination");
  const result = await run(["python3", "-I", "-c", EXTRACT, archive, destination, digest, String(LIMITS.members), String(LIMITS.expanded_bytes), inspectOnly ? "inspect" : "extract"], 60000);
  return JSON.parse(result);
}

// These pinned tools are thin ARM64 Mach-O executables (verified read-only with file/otool).
// A script named gcc, host clang or an Intel-only binary is not an accepted cross compiler.
export async function nativeTool(file) {
  const real = realpathSync(file), stat = statSync(real);
  requireFact(stat.isFile() && (stat.mode & 0o111) && stat.size >= 32 && stat.size <= 256 * 1024 ** 2, `tool is not a bounded executable: ${file}`);
  const fd = openSync(real, "r"), header = Buffer.alloc(32);
  try { requireFact(readSync(fd, header, 0, 32, 0) === 32, "short compiler header"); } finally { closeSync(fd); }
  requireFact(header.readUInt32LE(0) === 0xfeedfacf && header.readUInt32LE(4) === 0x100000c && header.readUInt32LE(12) === 2, `tool is not a real Darwin ARM64 executable: ${file}`);
  return { path: real, sha256: await fileDigest(real), host: HOST };
}
export function elfIdentity(bytes) {
  requireFact(bytes.length >= 64 && bytes.subarray(0, 4).equals(Buffer.from([0x7f, 69, 76, 70])) && bytes[4] === 2 && bytes[5] === 1, "smoke output is not ELF64 little-endian");
  const type = bytes.readUInt16LE(16), machine = bytes.readUInt16LE(18);
  requireFact((type === 2 || type === 3) && machine === 62, "smoke output is not linked x86_64 ELF");
  return { class: 64, machine: 62, type, sha256: hash(bytes) };
}
export function validateToolchainIdentity(identity) {
  requireFact(identity?.schema_version === 1 && identity.host === HOST && identity.target === TARGET && targetMatches(identity.reported_target), "compiler identity host/target mismatch");
  requireFact(["existing", "pinned-archive"].includes(identity.source), "unknown compiler source");
  requireFact(absolute(identity.sysroot) && /^[^\r\n]{1,1024}$/.test(identity.cc_version ?? "") && /^[^\r\n]{1,1024}$/.test(identity.cxx_version ?? ""), "compiler version/sysroot missing");
  requireFact(identity.tools && JSON.stringify(Object.keys(identity.tools).sort()) === JSON.stringify([...TOOL_KEYS].sort()), "incomplete compiler tool identities");
  for (const tool of Object.values(identity.tools)) requireFact(absolute(tool.path) && digestPattern.test(tool.sha256 ?? "") && tool.host === HOST, "invalid compiler tool identity");
  requireFact(/(?:gcc|g\+\+|GCC).*\d+\.\d+/i.test(identity.cc_version) && /(?:gcc|g\+\+|GCC).*\d+\.\d+/i.test(identity.cxx_version), "compiler version is not GNU GCC");
  if (identity.source === "pinned-archive") {
    requireFact(JSON.stringify(identity.provider) === JSON.stringify(TOOLCHAIN), "compiler archive provider mismatch");
    requireFact(identity.cc_version.includes(TOOLCHAIN.version) && identity.cxx_version.includes(TOOLCHAIN.version), "pinned compiler version mismatch");
  } else requireFact(identity.provider === null, "existing compiler cannot claim downloaded provenance");
  for (const key of ["c", "cpp"]) { const smoke = identity.smoke?.[key]; requireFact(smoke?.class === 64 && smoke.machine === 62 && [2, 3].includes(smoke.type) && digestPattern.test(smoke.sha256 ?? ""), "missing real C/C++ ELF smoke"); }
  return identity;
}
export async function verifyToolchainFiles(identity) {
  validateToolchainIdentity(identity);
  for (const tool of Object.values(identity.tools)) {
    const current = await nativeTool(tool.path);
    requireFact(current.path === tool.path && current.sha256 === tool.sha256, "compiler tool bytes changed after setup");
  }
}
export const compilerEnvironment = identity => ({
  CC_x86_64_unknown_linux_gnu: identity.tools.cc.path, CXX_x86_64_unknown_linux_gnu: identity.tools.cxx.path,
  AR_x86_64_unknown_linux_gnu: identity.tools.ar.path, RANLIB_x86_64_unknown_linux_gnu: identity.tools.ranlib.path,
  CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER: identity.tools.cc.path,
});
export function readToolchainIdentity(env) {
  const file = env.SCENEWORKS_CANONICAL_CROSS_IDENTITY, temp = realpathSync(env.RUNNER_TEMP);
  requireFact(absolute(file) && inside(temp, file) && realpathSync(file) === file, "compiler identity must be an owned run-local file");
  requireFact(path.basename(file) === "identity.json" && path.basename(path.dirname(file)).startsWith(`canonical-cross-${env.GITHUB_RUN_ID}-${env.GITHUB_RUN_ATTEMPT}-`), "compiler identity belongs to a different run");
  const identity = validateToolchainIdentity(JSON.parse(readFileSync(file, "utf8")));
  for (const [key, value] of Object.entries(compilerEnvironment(identity))) requireFact(env[key] === value, `canonical compiler environment differs: ${key}`);
  return identity;
}

export async function bootstrap(env, { platform = process.platform, arch = process.arch, directories = EXISTING_DIRS, execute, download = downloadArchive, extract = extractArchive } = {}) {
  requireFact(platform === "darwin" && arch === "arm64" && env.RUNNER_NAME === "nax-macos-2" && env.VERIFICATION_PROFILE === "canonical-preflight", "bootstrap is secondary-Mac canonical-only");
  requireFact(/^[1-9][0-9]*$/.test(env.GITHUB_RUN_ID ?? "") && /^[1-9][0-9]*$/.test(env.GITHUB_RUN_ATTEMPT ?? ""), "exact job identity required");
  const temp = realpathSync(env.RUNNER_TEMP), workspace = path.resolve(env.GITHUB_WORKSPACE);
  requireFact(!inside(workspace, temp) && temp !== workspace, "bootstrap storage must be outside the Git workspace");
  for (const key of ["GITHUB_ENV", "GITHUB_PATH"]) requireFact(absolute(env[key]) && inside(temp, realpathSync(env[key])) && !lstatSync(env[key]).isSymbolicLink() && statSync(env[key]).isFile(), "job environment export must stay in runner temporary storage");
  for (let cursor = temp;; cursor = path.dirname(cursor)) { requireFact(![".git"].some(name => { try { lstatSync(path.join(cursor, name)); return true; } catch (error) { if (error.code === "ENOENT") return false; throw error; } }), "bootstrap storage must be outside Git"); if (cursor === path.dirname(cursor)) break; }
  const work = mkdtempSync(path.join(temp, `canonical-cross-${env.GITHUB_RUN_ID}-${env.GITHUB_RUN_ATTEMPT}-`));
  const deadline = performance.now() + LIMITS.bootstrap_ms; let sequence = 0;
  const run = async (args, cap = LIMITS.command_ms, childEnv = {}) => {
    const left = deadline - performance.now(); requireFact(left > 0, "original bootstrap deadline exhausted");
    const limit = Math.min(cap, left);
    if (execute) return execute(args, work, limit, childEnv);
    const previous = Object.fromEntries(Object.keys(childEnv).map(key => [key, process.env[key]]));
    Object.assign(process.env, childEnv);
    try {
      const result = await executeCpuCommand(work, args, path.join(work, `command-${++sequence}.log`), limit, { cancelled: false });
      requireFact(result.exit === 0, `prerequisite command failed: ${path.basename(args[0])}`);
      return result.output.trim();
    } finally { for (const [key, value] of Object.entries(previous)) { if (value === undefined) delete process.env[key]; else process.env[key] = value; } }
  };
  const probe = async (directory, prefix, source) => {
    const cc = await nativeTool(path.join(directory, `${prefix}-gcc`)), cxx = await nativeTool(path.join(directory, `${prefix}-g++`));
    const childEnv = { PATH: `${directory}${path.delimiter}${env.PATH ?? ""}` };
    const reported = (await run([cc.path, "-dumpmachine"], undefined, childEnv)).trim();
    requireFact(targetMatches(reported) && targetMatches((await run([cxx.path, "-dumpmachine"], undefined, childEnv)).trim()), "compiler is not GNU x86_64 Linux");
    const ccVersion = (await run([cc.path, "--version"], undefined, childEnv)).split("\n")[0], cxxVersion = (await run([cxx.path, "--version"], undefined, childEnv)).split("\n")[0];
    requireFact(/(?:gcc|g\+\+|GCC).*\d+\.\d+/i.test(ccVersion) && /(?:gcc|g\+\+|GCC).*\d+\.\d+/i.test(cxxVersion), "not a GNU compiler version");
    const tools = { cc, cxx };
    for (const tool of ["ar", "ranlib"]) tools[tool] = await nativeTool(path.join(directory, `${prefix}-${tool}`));
    for (const tool of ["as", "ld", "cc1", "cc1plus", "collect2"]) {
      const file = (await run([cc.path, `-print-prog-name=${tool}`], undefined, childEnv)).trim();
      requireFact(absolute(file), `compiler did not resolve ${tool} to a real tool path`);
      tools[tool] = await nativeTool(file);
    }
    const sysroot = realpathSync((await run([cc.path, "-print-sysroot"], undefined, childEnv)).trim());
    requireFact(statSync(sysroot).isDirectory(), "compiler sysroot is absent");
    if (source === "pinned-archive") {
      const extracted = path.join(work, "unpacked", ROOT);
      requireFact(inside(extracted, sysroot) && Object.values(tools).every(tool => inside(extracted, tool.path)), "pinned toolchain is not self-contained/relocatable");
    }
    const c = path.join(work, "smoke.c"), cpp = path.join(work, "smoke.cpp"), cElf = path.join(work, "smoke-c.elf"), cppElf = path.join(work, "smoke-cpp.elf");
    writeFileSync(c, '#include <stdio.h>\nint main(void) { return puts("cross") < 0; }\n');
    writeFileSync(cpp, '#include <vector>\nint main() { return std::vector<int>{1}.at(0) - 1; }\n');
    const object = path.join(work, "smoke-c.o"), library = path.join(work, "smoke-c.a");
    await run([cc.path, "-c", c, "-o", object], 60000, childEnv);
    await run([tools.ar.path, "crs", library, object], undefined, childEnv);
    await run([tools.ranlib.path, library], undefined, childEnv);
    await run([cc.path, library, "-o", cElf], 60000, childEnv);
    await run([cxx.path, "-std=c++17", cpp, "-o", cppElf], 60000, childEnv);
    // Read headers only. Neither Linux program is ever executed.
    const smoke = { c: elfIdentity(readFileSync(cElf)), cpp: elfIdentity(readFileSync(cppElf)) };
    for (const tool of Object.values(tools)) requireFact(await fileDigest(tool.path) === tool.sha256, "compiler changed during smoke");
    return validateToolchainIdentity({ schema_version: 1, source, provider: source === "pinned-archive" ? TOOLCHAIN : null, host: HOST, target: TARGET, reported_target: reported, cc_version: ccVersion, cxx_version: cxxVersion, sysroot, tools, smoke });
  };
  let identity;
  for (const directory of directories) {
    for (const prefix of [TARGET, "x86_64-linux-gnu"]) {
      try { identity = await probe(directory, prefix, "existing"); break; }
      catch (error) { console.log(`canonical cross discovery: ${path.join(directory, prefix)}: ${error.message}`); }
    }
    if (identity) break;
  }
  if (!identity) {
    const archive = path.join(work, "toolchain.tar.gz"), unpacked = path.join(work, "unpacked"); mkdirSync(unpacked, { mode: 0o700 });
    await download(archive, { timeout: Math.min(LIMITS.download_ms, Math.max(1, deadline - performance.now())) });
    await extract(archive, unpacked, { run });
    rmSync(archive); // Only this run's verified download; no shared toolchain/cache deletion.
    identity = await probe(path.join(unpacked, ROOT, "bin"), TARGET, "pinned-archive");
  }
  requireFact(performance.now() <= deadline, "original bootstrap deadline exceeded");
  const file = path.join(work, "identity.json"); writeFileSync(file, `${JSON.stringify(identity, null, 2)}\n`, { flag: "wx" });
  const exports = { ...compilerEnvironment(identity), SCENEWORKS_CANONICAL_CROSS_IDENTITY: file };
  for (const value of Object.values(exports)) requireFact(absolute(value), "unsafe job-scoped compiler path");
  appendFileSync(env.GITHUB_ENV, Object.entries(exports).map(([key, value]) => `${key}=${value}\n`).join(""));
  appendFileSync(env.GITHUB_PATH, `${path.dirname(identity.tools.cc.path)}\n`);
  console.log(JSON.stringify({ canonical_cross_toolchain: identity, identity_sha256: identityDigest(identity) }));
  return { identity, file, work };
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  if (process.argv.length !== 2) { console.error("canonical cross toolchain accepts no operator commands or arguments"); process.exitCode = 1; }
  else bootstrap(process.env).catch(error => { console.error(error.message); process.exitCode = 1; });
}
