import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const script = path.join(root, "scripts/ensure-ci-ffmpeg.sh");
// Native Windows spawn cannot execute the extensionless shebang stubs on PATH. These fixtures
// exercise the Ubuntu Bash helper; the static workflow contract below still runs everywhere.
const skipWithoutShebang = process.platform === "win32" && "the stub CLIs are shebang scripts on PATH";

function runFixture(options = {}) {
  const fixture = mkdtempSync(path.join(os.tmpdir(), "ci-ffmpeg-test-"));
  const bin = path.join(fixture, "bin");
  mkdirSync(bin);
  const osRelease = path.join(fixture, "os-release");
  writeFileSync(osRelease, `ID=${options.os ?? "ubuntu"}\nVERSION_CODENAME=noble\n`);
  const source = String.raw`
const fs = require('node:fs');
const path = require('node:path');
const {spawnSync} = require('node:child_process');
const name = path.basename(process.argv[1]);
const args = process.argv.slice(2);
const dir = process.env.FIXTURE_ROOT;
fs.appendFileSync(path.join(dir,'calls'), JSON.stringify({name,args})+'\n');
if(name === 'sudo') {
  const result = spawnSync(args[0],args.slice(1),{env:process.env,stdio:'inherit'});
  process.exit(result.status ?? 1);
}
if(name === 'dpkg') { console.log(process.env.ARCH); process.exit(0); }
if(name === 'timeout') {
  if(args.includes('update') && process.env.FAIL_PHASE === 'update-timeout') process.exit(124);
  if(args.includes('--download-only') && process.env.FAIL_PHASE === 'download-timeout') process.exit(124);
  const index = args.indexOf('apt-get');
  if(index < 0) process.exit(99);
  const result = spawnSync('apt-get',args.slice(index+1),{env:process.env,stdio:'inherit'});
  process.exit(result.status ?? 1);
}
if(name === 'apt-get') {
  const config = args.find(arg=>arg.startsWith('Dir::Etc::sourcelist='));
  if(!config) process.exit(98);
  fs.writeFileSync(path.join(dir,'sources'),fs.readFileSync(config.split('=')[1]));
  const phase = args.includes('update') ? 'update' : args.includes('--download-only') ? 'download' : 'install';
  if(process.env.FAIL_PHASE === phase) { console.error('APT fixture refuses '+phase); process.exit(100); }
  if(phase === 'install') fs.writeFileSync(path.join(dir,'installed'),'yes');
  process.exit(0);
}
if(name === 'ffmpeg') {
  const installed = fs.existsSync(path.join(dir,'installed'));
  if(!installed && process.env.EXISTING === 'missing') process.exit(127);
  if(args[0] === '-version') { console.log('ffmpeg version fixture\nconfiguration: fixture'); process.exit(0); }
  if(process.env.POST_BROKEN === '1' || (!installed && process.env.EXISTING === 'broken-codec')) process.exit(1);
  if(args.includes('-c:v')) fs.writeFileSync(args.at(-1),'fixture AV data');
  else if(!fs.existsSync(args[args.indexOf('-i')+1])) process.exit(1);
  process.exit(0);
}
process.exit(97);
`;
  for (const name of ["sudo", "dpkg", "timeout", "apt-get", "ffmpeg"]) {
    writeFileSync(path.join(bin, name), `#!${process.execPath}\n${source}`, { mode: 0o755 });
  }
  try {
    const result = spawnSync("bash", [script], {
      env: {
        ...process.env, PATH: `${bin}${path.delimiter}${process.env.PATH}`,
        TMPDIR: fixture, FIXTURE_ROOT: fixture, CI_FFMPEG_OS_RELEASE_FILE: osRelease,
        EXISTING: options.existing ?? "missing", ARCH: options.arch ?? "amd64",
        FAIL_PHASE: options.fail ?? "", POST_BROKEN: options.postBroken ? "1" : "0",
      },
      encoding: "utf8", timeout: 15_000,
    });
    const calls = readFileSync(path.join(fixture, "calls"), "utf8").trim().split("\n").map(JSON.parse);
    const sources = calls.some(call => call.name === "apt-get")
      ? readFileSync(path.join(fixture, "sources"), "utf8") : null;
    return { ...result, calls, sources };
  } finally {
    rmSync(fixture, { recursive: true, force: true });
  }
}

test("a real AV-mux/decode probe reuses a suitable existing ffmpeg without sudo or APT", { skip: skipWithoutShebang }, () => {
  const result = runFixture({ existing: "working" });
  assert.equal(result.status, 0, result.stderr);
  assert.deepEqual(result.calls.map(call => call.name), ["ffmpeg", "ffmpeg", "ffmpeg"]);
  const encode = result.calls[1].args;
  assert.ok(encode.includes("libx264") && encode.includes("aac") && encode.includes("+faststart"));
  assert.ok(result.calls[2].args.includes("0:v:0") && result.calls[2].args.includes("0:a:0"));
});

test("missing ffmpeg uses only signed official Ubuntu sources and separates download from unpack", { skip: skipWithoutShebang }, () => {
  const result = runFixture();
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.sources, /URIs: https:\/\/archive\.ubuntu\.com\/ubuntu/);
  assert.match(result.sources, /URIs: https:\/\/security\.ubuntu\.com\/ubuntu/);
  assert.match(result.sources, /Suites: noble noble-updates/);
  assert.match(result.sources, /Suites: noble-security/);
  assert.equal((result.sources.match(/Signed-By: \/usr\/share\/keyrings\/ubuntu-archive-keyring\.gpg/g) ?? []).length, 2);
  const apt = result.calls.filter(call => call.name === "apt-get");
  assert.equal(apt.length, 3);
  assert.ok(apt[0].args.includes("update"));
  assert.ok(apt[1].args.includes("--download-only"));
  assert.ok(apt[2].args.includes("--no-download"));
  for (const call of apt) {
    assert.ok(call.args.includes("Acquire::AllowInsecureRepositories=false"));
    assert.ok(call.args.includes("APT::Get::AllowUnauthenticated=false"));
    assert.ok(call.args.includes("APT::Update::Error-Mode=any"));
    assert.ok(call.args.includes("Dir::Etc::sourceparts=-"));
    assert.ok(call.args.includes("Acquire::Languages=none"));
    assert.ok(call.args.includes("Acquire::IndexTargets::deb::CNF::DefaultEnabled=false"));
    assert.ok(call.args.includes("Acquire::IndexTargets::deb::DEP-11::DefaultEnabled=false"));
    assert.ok(call.args.some(arg => arg.startsWith("Dir::State::lists=") && !arg.includes("/var/lib/apt")));
  }
  const deadlines = result.calls.filter(call => call.name === "timeout");
  assert.deepEqual(deadlines.map(call => call.args.slice(0, 3)), [
    ["--signal=TERM", "--kill-after=10s", "180s"],
    ["--signal=TERM", "--kill-after=10s", "180s"],
  ]);
});

test("a binary that reports a version but cannot encode is replaced and re-probed", { skip: skipWithoutShebang }, () => {
  const result = runFixture({ existing: "broken-codec" });
  assert.equal(result.status, 0, result.stderr);
  assert.equal(result.calls.filter(call => call.name === "apt-get").length, 3);
});

test("an APT metadata/signature failure stops before package download or installation", { skip: skipWithoutShebang }, () => {
  const result = runFixture({ fail: "update" });
  assert.equal(result.status, 100);
  assert.equal(result.calls.filter(call => call.name === "apt-get").length, 1);
});

test("a continuously slow download is bounded and never reaches dpkg installation", { skip: skipWithoutShebang }, () => {
  const result = runFixture({ fail: "download-timeout" });
  assert.equal(result.status, 124);
  assert.ok(!result.calls.some(call => call.name === "apt-get" && call.args.includes("--no-download")));
});

test("a continuously slow metadata update is bounded and never downloads or installs packages", { skip: skipWithoutShebang }, () => {
  const result = runFixture({ fail: "update-timeout" });
  assert.equal(result.status, 124);
  assert.ok(!result.calls.some(call => call.name === "apt-get"));
  assert.equal(result.calls.filter(call => call.name === "timeout").length, 1);
});

test("install failure and an unsuitable installed binary both fail closed", { skip: skipWithoutShebang }, () => {
  assert.equal(runFixture({ fail: "install" }).status, 100);
  assert.equal(runFixture({ postBroken: true }).status, 1);
});

test("automatic installation refuses a different distribution or unsupported architecture", { skip: skipWithoutShebang }, () => {
  for (const options of [{ os: "debian" }, { arch: "arm64" }]) {
    const result = runFixture(options);
    assert.equal(result.status, 1);
    assert.ok(!result.calls.some(call => call.name === "apt-get"));
  }
});

test("parity-rust uses the helper while retaining its real-ffmpeg gate and job limit", () => {
  const workflow = readFileSync(path.join(root, ".github/workflows/check.yml"), "utf8");
  const lane = workflow.split("  parity-rust:")[1].split("\n  parity-scaffold:")[0];
  assert.match(lane, /SCENEWORKS_REQUIRE_FFMPEG: "1"/);
  assert.match(lane, /^    timeout-minutes: 60$/m);
  assert.match(lane, /run: bash scripts\/ensure-ci-ffmpeg\.sh/);
});
