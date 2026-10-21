import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { spawnSync } from "node:child_process";
import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, realpathSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath, pathToFileURL } from "node:url";
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const modulePath = process.env.CANONICAL_CROSS_UNDER_TEST ?? path.join(ROOT, "scripts/canonical-cross-toolchain.mjs");
const { TOOLCHAIN, LIMITS, EXISTING_DIRS, TOOL_KEYS, bootstrap, compilerEnvironment, downloadArchive, elfIdentity, extractArchive, identityDigest, nativeTool, readToolchainIdentity, validateToolchainIdentity, verifyToolchainFiles } = await import(pathToFileURL(modulePath));
const sha = bytes => createHash("sha256").update(bytes).digest("hex");
const posix = { skip: process.platform === "win32" };
function macho(cpu = 0x100000c) { const b = Buffer.alloc(64); b.writeUInt32LE(0xfeedfacf, 0); b.writeUInt32LE(cpu, 4); b.writeUInt32LE(2, 12); return b; }
function elf(machine = 62) { const b = Buffer.alloc(64); b.set([0x7f, 69, 76, 70, 2, 1]); b.writeUInt16LE(2, 16); b.writeUInt16LE(machine, 18); return b; }
function fixture() {
  const root = realpathSync(mkdtempSync(path.join(os.tmpdir(), "cross-toolchain-fixture-"))), temp = path.join(root, "temp"); mkdirSync(temp);
  const env = { RUNNER_NAME: "nax-macos-2", VERIFICATION_PROFILE: "canonical-preflight", RUNNER_TEMP: temp, GITHUB_WORKSPACE: path.join(root, "source"), GITHUB_RUN_ID: "41", GITHUB_RUN_ATTEMPT: "2", GITHUB_ENV: path.join(temp, "env"), GITHUB_PATH: path.join(temp, "path"), PATH: "/usr/bin:/bin" };
  writeFileSync(env.GITHUB_ENV, ""); writeFileSync(env.GITHUB_PATH, "");
  const install = base => {
    const bin = path.join(base, "bin"); mkdirSync(bin, { recursive: true }); mkdirSync(path.join(base, "sysroot"));
    for (const prefix of ["x86_64-unknown-linux-gnu", "x86_64-linux-gnu"]) for (const name of ["gcc", "g++", "ar", "ranlib"]) writeFileSync(path.join(bin, `${prefix}-${name}`), macho(), { mode: 0o755 });
    for (const name of ["as", "ld", "cc1", "cc1plus", "collect2"]) writeFileSync(path.join(bin, name), macho(), { mode: 0o755 });
    return bin;
  };
  const calls = [];
  const execute = async (args, _cwd, limit, childEnv) => {
    calls.push(args); assert.ok(limit > 0 && limit <= 60000); assert.ok(!args[0].endsWith(".elf"), "Linux smoke must never be executed");
    const bin = path.dirname(args[0]); assert.ok(childEnv.PATH.startsWith(bin + path.delimiter));
    if (args.includes("-dumpmachine")) return "x86_64-unknown-linux-gnu";
    if (args.includes("--version")) return path.basename(args[0]).includes("g++") ? "g++ (GCC) 15.2.0\nlicense" : "gcc (GCC) 15.2.0\nlicense";
    const program = args.find(arg => arg.startsWith("-print-prog-name="));
    if (program) return path.join(bin, program.split("=")[1]);
    if (args.includes("-print-sysroot")) return path.join(path.dirname(bin), "sysroot");
    if (args[0].endsWith("-ar")) { assert.equal(args[1], "crs"); assert.ok(existsSync(args[3])); writeFileSync(args[2], "!<arch>\nfixture"); return ""; }
    if (args[0].endsWith("-ranlib")) { assert.ok(existsSync(args[1])); return ""; }
    if (args.includes("-o")) { for (const file of args.filter(arg => arg.endsWith(".a"))) assert.ok(readFileSync(file, "utf8").startsWith("!<arch>")); writeFileSync(args[args.indexOf("-o") + 1], elf()); return ""; }
    throw new Error("unexpected fixture command");
  };
  return { root, env, install, execute, calls, options: { platform: "darwin", arch: "arm64", directories: [], execute } };
}
function archive(entries) {
  const code = `import sys,json,tarfile,io,base64\nout=io.BytesIO()\nwith tarfile.open(fileobj=out,mode='w:gz') as t:\n for e in json.loads(sys.argv[1]):\n  m=tarfile.TarInfo(e['name']);m.mode=e.get('mode',0o755)\n  kind=e.get('type','file');m.type={'dir':tarfile.DIRTYPE,'symlink':tarfile.SYMTYPE,'hardlink':tarfile.LNKTYPE,'fifo':tarfile.FIFOTYPE}.get(kind,tarfile.REGTYPE)\n  m.linkname=e.get('link','');data=e.get('data','x').encode();m.size=len(data) if kind=='file' else 0\n  t.addfile(m,io.BytesIO(data) if kind=='file' else None)\nsys.stdout.buffer.write(out.getvalue())`;
  const r = spawnSync(process.platform === "win32" ? "python" : "python3", ["-I", "-c", code, JSON.stringify(entries)], { maxBuffer: 4 * 1024 ** 2 });
  assert.equal(r.status, 0, r.stderr.toString()); return r.stdout;
}
async function extractFixture(bytes, { inspectOnly = false } = {}) {
  const root = mkdtempSync(path.join(os.tmpdir(), "cross-archive-fixture-")), input = path.join(root, "input.tar.gz"), output = path.join(root, "unpacked");
  mkdirSync(output); writeFileSync(input, bytes);
  const run = async args => {
    const r = spawnSync(process.platform === "win32" ? "python" : args[0], args.slice(1), { encoding: "utf8", timeout: 10000 });
    if (r.status !== 0) throw new Error(r.stderr);
    return r.stdout.trim();
  };
  try { const result = await extractArchive(input, output, { digest: sha(bytes), run, inspectOnly }); return { root, output, result }; }
  catch (error) { assert.deepEqual(readdirSync(output), [], "invalid archive wrote bytes before rejection"); rmSync(root, { recursive: true, force: true }); throw error; }
}

test("pinned provider is exact Apple Silicon host and GNU Linux target with finite bounds", () => {
  assert.equal(TOOLCHAIN.version, "15.2.0"); assert.equal(TOOLCHAIN.asset_id, 356184618); assert.equal(TOOLCHAIN.bytes, 202165149);
  assert.equal(TOOLCHAIN.sha256, "04746819d162f829dacf2ea6d6947b9fb4141ffbd91588f0bfbaab4c2c7e1a30");
  assert.match(TOOLCHAIN.url, /\/v15\.2\.0\/x86_64-unknown-linux-gnu-aarch64-darwin\.tar\.gz$/); assert.doesNotMatch(TOOLCHAIN.url, /latest/);
  assert.equal(EXISTING_DIRS.length, 6); assert.ok(EXISTING_DIRS.every(dir => path.isAbsolute(dir)));
  assert.equal(LIMITS.bootstrap_ms, 300000); assert.ok(LIMITS.archive_bytes < 300 * 1024 ** 2); assert.ok(LIMITS.expanded_bytes <= 1024 ** 3);
});
test("pinned download rejects incorrect size and hash before extraction", async () => {
  const root = mkdtempSync(path.join(os.tmpdir(), "cross-download-fixture-"));
  const bytes = Buffer.from("bounded archive fixture"), source = { ...TOOLCHAIN, bytes: bytes.length, sha256: sha(bytes) };
  const fetchImpl = async () => ({ ok: true, headers: { get: () => String(bytes.length) }, body: (async function* () { yield bytes; })() });
  try {
    const good = path.join(root, "good"); await downloadArchive(good, { source, fetchImpl }); assert.deepEqual(readFileSync(good), bytes);
    await assert.rejects(() => downloadArchive(path.join(root, "wrong-hash"), { source: { ...source, sha256: "0".repeat(64) }, fetchImpl }), /SHA-256 mismatch/);
    await assert.rejects(() => downloadArchive(path.join(root, "wrong-size"), { source: { ...source, bytes: bytes.length + 1 }, fetchImpl }), /size or SHA-256/);
    await assert.rejects(() => downloadArchive(path.join(root, "oversized"), { source, fetchImpl: async () => ({ ok: true, body: [], headers: { get: () => String(LIMITS.archive_bytes + 1) } }) }), /content length/);
  } finally { rmSync(root, { recursive: true, force: true }); }
});
test("archive checksum mismatch never invokes the extractor", async () => {
  const root = mkdtempSync(path.join(os.tmpdir(), "cross-before-extract-")), file = path.join(root, "archive"), out = path.join(root, "out");
  mkdirSync(out); writeFileSync(file, archive([{ name: "x86_64-unknown-linux-gnu/data" }])); let invoked = 0;
  try {
    await assert.rejects(() => extractArchive(file, out, { digest: "0".repeat(64), run: async () => { invoked++; return "{}"; } }), /digest mismatch/);
    assert.equal(invoked, 0); assert.deepEqual(readdirSync(out), []);
  } finally { rmSync(root, { recursive: true, force: true }); }
});
test("archive refuses a symlink extraction destination", posix, async () => {
  const root = mkdtempSync(path.join(os.tmpdir(), "cross-destination-")), file = path.join(root, "archive"), outside = path.join(root, "outside"), out = path.join(root, "out");
  mkdirSync(outside); symlinkSync(outside, out); const bytes = archive([{ name: "x86_64-unknown-linux-gnu/data" }]); writeFileSync(file, bytes);
  try { await assert.rejects(() => extractArchive(file, out, { digest: sha(bytes), run: () => { throw new Error("extractor must not run"); } }), /symlink extraction destination/); assert.deepEqual(readdirSync(outside), []); }
  finally { rmSync(root, { recursive: true, force: true }); }
});
test("safe archive accepts internal links and data without executing any member", posix, async () => {
  const f = await extractFixture(archive([{ name: "x86_64-unknown-linux-gnu/data", data: "not executable code" }, { name: "x86_64-unknown-linux-gnu/alias", type: "symlink", link: "data" }, { name: "x86_64-unknown-linux-gnu/hard", type: "hardlink", link: "x86_64-unknown-linux-gnu/data" }]));
  try { assert.equal(readFileSync(path.join(f.output, "x86_64-unknown-linux-gnu/alias"), "utf8"), "not executable code"); assert.equal(f.result.members, 3); }
  finally { rmSync(f.root, { recursive: true, force: true }); }
});
for (const [name, entries] of Object.entries({
  traversal: [{ name: "x86_64-unknown-linux-gnu/../escape" }], absolute: [{ name: "/x86_64-unknown-linux-gnu/file" }], backslash: [{ name: "x86_64-unknown-linux-gnu/back\\slash" }],
  duplicate: [{ name: "x86_64-unknown-linux-gnu/file" }, { name: "x86_64-unknown-linux-gnu/file" }],
  symlinkEscape: [{ name: "x86_64-unknown-linux-gnu/link", type: "symlink", link: "../../escape" }], hardlinkEscape: [{ name: "x86_64-unknown-linux-gnu/link", type: "hardlink", link: "../escape" }],
  linkAncestor: [{ name: "x86_64-unknown-linux-gnu/link", type: "symlink", link: "directory" }, { name: "x86_64-unknown-linux-gnu/link/file" }, { name: "x86_64-unknown-linux-gnu/directory", type: "dir" }],
  linkCycle: [{ name: "x86_64-unknown-linux-gnu/a", type: "symlink", link: "b" }, { name: "x86_64-unknown-linux-gnu/b", type: "symlink", link: "a" }],
  special: [{ name: "x86_64-unknown-linux-gnu/pipe", type: "fifo" }], permission: [{ name: "x86_64-unknown-linux-gnu/setuid", mode: 0o4755 }],
})) test(`archive rejects ${name} before any extraction`, async () => { await assert.rejects(() => extractFixture(archive(entries), { inspectOnly: true })); });
test("archive metadata count and expansion bombs fail before writing", async () => {
  await assert.rejects(() => extractFixture(archive(Array.from({ length: LIMITS.members + 1 }, (_, i) => ({ name: `x86_64-unknown-linux-gnu/item${i}`, data: "" })))));
  const r = spawnSync(process.platform === "win32" ? "python" : "python3", ["-I", "-c", "import tarfile,gzip,sys;m=tarfile.TarInfo('x86_64-unknown-linux-gnu/bomb');m.size=2**31;sys.stdout.buffer.write(gzip.compress(m.tobuf()+bytes(1024)))"]);
  assert.equal(r.status, 0); await assert.rejects(() => extractFixture(r.stdout));
  const pax = spawnSync(process.platform === "win32" ? "python" : "python3", ["-I", "-c", "import tarfile,gzip,sys;m=tarfile.TarInfo('pax');m.type=tarfile.XHDTYPE;m.size=8*1024*1024;sys.stdout.buffer.write(gzip.compress(m.tobuf()+bytes(1024)))"]);
  // Python 3.12 rejects the oversized metadata read at our bound; newer tarfile versions
  // chunk that read and reject this truncated header themselves. Both must reject before writes.
  assert.equal(pax.status, 0); await assert.rejects(() => extractFixture(pax.stdout), /bound|header|data/);
});
test("compiler validation rejects a script or wrong-host binary", posix, async () => {
  const root = mkdtempSync(path.join(os.tmpdir(), "cross-fake-compiler-")), file = path.join(root, "gcc");
  try {
    writeFileSync(file, "#!/bin/sh\nexit 0\n", { mode: 0o755 }); await assert.rejects(() => nativeTool(file), /bounded executable|short compiler header|real Darwin/);
    writeFileSync(file, macho(0x1000007)); await assert.rejects(() => nativeTool(file), /Darwin ARM64/);
    writeFileSync(file, macho()); chmodSync(file, 0o644); await assert.rejects(() => nativeTool(file), /bounded executable/);
  } finally { rmSync(root, { recursive: true, force: true }); }
});
test("ELF smoke requires linked little-endian ELF64 x86_64 and never executes output", () => {
  assert.equal(elfIdentity(elf()).machine, 62);
  for (const bytes of [macho(), elf(183), Buffer.alloc(8)]) assert.throws(() => elfIdentity(bytes), /ELF|x86_64/);
});
test("existing real compiler is preferred and target variables remain job scoped", posix, async () => {
  const f = fixture();
  try {
    const bin = f.install(path.join(f.root, "existing"));
    for (const tool of ["gcc", "g++", "ar", "ranlib"]) rmSync(path.join(bin, `x86_64-unknown-linux-gnu-${tool}`));
    const result = await bootstrap(f.env, { ...f.options, directories: [bin], download: () => { throw new Error("existing compiler must not download"); } });
    assert.equal(result.identity.source, "existing"); assert.equal(result.identity.provider, null);
    assert.ok(result.identity.tools.cc.path.endsWith("x86_64-linux-gnu-gcc"));
    assert.equal(existsSync(path.join(bin, "x86_64-unknown-linux-gnu-gcc")), false, "no prefix shim may be created");
    assert.equal(f.calls.filter(args => args.includes("-o")).length, 3); assert.ok(f.calls.every(args => !args[0].endsWith(".elf")));
    assert.equal(Object.keys(result.identity.tools).length, TOOL_KEYS.length);
    const exports = Object.fromEntries(readFileSync(f.env.GITHUB_ENV, "utf8").trim().split("\n").map(line => { const i = line.indexOf("="); return [line.slice(0, i), line.slice(i + 1)]; }));
    const identity = readToolchainIdentity({ ...f.env, ...exports }); assert.equal(identityDigest(identity), identityDigest(result.identity));
    await verifyToolchainFiles(identity);
    await assert.rejects(() => verifyToolchainFiles({ ...identity, tools: { ...identity.tools, cc: { ...identity.tools.cc, sha256: "0".repeat(64) } } }), /bytes changed/);
    assert.throws(() => readToolchainIdentity({ ...f.env, ...exports, CC_x86_64_unknown_linux_gnu: "/wrong/compiler" }), /environment differs/);
  } finally { rmSync(f.root, { recursive: true, force: true }); }
});
test("missing compiler falls back to the pinned owned toolchain with full C and C++ smokes", posix, async () => {
  const f = fixture(); let downloads = 0, extracts = 0;
  try {
    const result = await bootstrap(f.env, { ...f.options,
      download: async file => { downloads++; writeFileSync(file, "mock verified download; archive validation is tested separately"); },
      extract: async (_file, directory) => { extracts++; f.install(path.join(directory, "x86_64-unknown-linux-gnu")); },
    });
    assert.equal(downloads, 1); assert.equal(extracts, 1); assert.equal(result.identity.source, "pinned-archive"); assert.deepEqual(result.identity.provider, TOOLCHAIN);
    assert.ok(result.identity.tools.cc.path.startsWith(result.work)); assert.equal(f.calls.filter(args => args.includes("-o")).length, 3);
    for (const change of [value => { value.target = "aarch64-unknown-linux-gnu"; }, value => { value.reported_target = "arm64-apple-darwin"; }, value => { value.provider.sha256 = "0".repeat(64); }, value => { delete value.tools.cc1; }, value => { value.smoke.cpp.machine = 183; }]) {
      const bad = structuredClone(result.identity); change(bad); assert.throws(() => validateToolchainIdentity(bad));
    }
  } finally { rmSync(f.root, { recursive: true, force: true }); }
});
test("bootstrap refuses other profiles, runners and in-repository storage before provisioning", posix, async () => {
  const f = fixture(); let provisioning = 0;
  try {
    const options = { ...f.options, download: () => { provisioning++; throw new Error("fixture forbids real downloads"); } };
    for (const fields of [{ VERIFICATION_PROFILE: "consumer-assertions" }, { VERIFICATION_PROFILE: "facts-and-projections" }, { RUNNER_NAME: "nax" }, { GITHUB_WORKSPACE: f.env.RUNNER_TEMP }, { GITHUB_ENV: f.env.GITHUB_WORKSPACE }, { GITHUB_RUN_ID: "0" }]) await assert.rejects(() => bootstrap({ ...f.env, ...fields }, options));
    assert.equal(provisioning, 0); assert.equal(f.calls.length, 0); assert.equal(readFileSync(f.env.GITHUB_ENV, "utf8"), "");
  } finally { rmSync(f.root, { recursive: true, force: true }); }
});

if (!process.env.CANONICAL_CROSS_MUTATION_CHILD) test("named bootstrap bypasses fail their original Node tests without compiler execution", async t => {
  const source = readFileSync(path.join(ROOT, "scripts/canonical-cross-toolchain.mjs"), "utf8");
  for (const [id, target, old, replacement] of [
    ["CROSS-before-extract-sha", "archive checksum mismatch never invokes the extractor", 'await fileDigest(archive) === digest', 'true'],
    ["CROSS-no-ELF-exec", "missing compiler falls back to the pinned owned toolchain with full C and C++ smokes", 'const smoke = { c: elfIdentity(readFileSync(cElf))', 'await run([cElf]); const smoke = { c: elfIdentity(readFileSync(cElf))'],
    ["CROSS-profile", "bootstrap refuses other profiles, runners and in-repository storage before provisioning", 'env.VERIFICATION_PROFILE === "canonical-preflight"', 'true'],
    ["CROSS-download-sha", "pinned download rejects incorrect size and hash before extraction", 'h.digest("hex") === source.sha256', 'true'],
    ["CROSS-host", "compiler validation rejects a script or wrong-host binary", 'header.readUInt32LE(4) === 0x100000c', 'true'],
    ["CROSS-ELF", "ELF smoke requires linked little-endian ELF64 x86_64 and never executes output", '(type === 2 || type === 3) && machine === 62', '(type === 2 || type === 3)'],
    ["CROSS-target", "missing compiler falls back to the pinned owned toolchain with full C and C++ smokes", 'identity.target === TARGET && targetMatches(identity.reported_target)', 'true'],
  ]) await t.test(id, () => {
    const scratch = mkdtempSync(path.join(os.tmpdir(), "cross-bootstrap-mutation-"));
    try {
      assert.equal(source.split(old).length - 1, 1);
      const modified = path.join(scratch, "helper.mjs"); writeFileSync(modified, source.replace(old, replacement));
      writeFileSync(path.join(scratch, "moss-consumer-assertions.mjs"), readFileSync(path.join(ROOT, "scripts/moss-consumer-assertions.mjs")));
      const env = { ...process.env, CANONICAL_CROSS_MUTATION_CHILD: "1", CANONICAL_CROSS_UNDER_TEST: modified }; delete env.NODE_TEST_CONTEXT;
      const args = ["--test", `--test-name-pattern=^${target.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}$`, fileURLToPath(import.meta.url)];
      const red = spawnSync(process.execPath, args, { cwd: ROOT, env, encoding: "utf8", timeout: 15000 });
      assert.notEqual(red.status, 0, red.stdout + red.stderr); assert.match(red.stdout, /(?:ℹ|#) fail 1/); assert.doesNotMatch(red.stderr, /SyntaxError|ERR_MODULE_NOT_FOUND/);
      writeFileSync(modified, source);
      const green = spawnSync(process.execPath, args, { cwd: ROOT, env, encoding: "utf8", timeout: 15000 });
      assert.equal(green.status, 0, green.stdout + green.stderr);
    } finally { rmSync(scratch, { recursive: true, force: true }); }
  });
});
