import assert from "node:assert/strict";
import { execFile as execFileCallback } from "node:child_process";
import { createHash } from "node:crypto";
import { chmod, copyFile, link, lstat, mkdir, mkdtemp, readFile, rm, stat, symlink, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { promisify } from "node:util";
import { treeIdentity, validateCorpusAssets, validateTerminalServiceClosure } from "./starvector-terminal-readiness.mjs";
import { terminalTreeEntry, terminalTreeSha256 } from "./lib/terminal-tree-identity.mjs";
import { closureTreeHash, copyRegularTree, productServiceActiveStatePath, productServiceBackendEnv, productServiceBuildArgs, productServiceLogPaths, productServiceLogsIdentity, productServiceStateRoot, productServiceTaskkillArguments, relocateProductServiceLibrary, stopProductService } from "./starvector-terminal-product-service.mjs";

const workflow = await readFile(".github/workflows/starvector-terminal.yml", "utf8");
const readiness = await readFile(".github/workflows/starvector-terminal-readiness.yml", "utf8");
const hash = (value) => createHash("sha256").update(value).digest("hex");
const pin = "c6d6a4dbd61ab09c26ff5526632cae2cefea60ed";
const execFile = promisify(execFileCallback);

async function availableLoopbackPort() {
  const reservation = createServer();
  await new Promise((resolve, reject) => { reservation.once("error", reject); reservation.listen(0, "127.0.0.1", resolve); });
  const port = reservation.address().port;
  await new Promise((resolve, reject) => reservation.close((error) => error ? reject(error) : resolve()));
  return port;
}

async function executableAlias(destination) {
  await mkdir(path.dirname(destination), { recursive: true });
  let copied = false;
  try { await link(process.execPath, destination); } catch (error) {
    if (!["EXDEV", "EPERM", "EACCES"].includes(error.code)) throw error;
    await copyFile(process.execPath, destination);
    copied = true;
  }
  if (copied && process.platform !== "win32") await chmod(destination, 0o755);
}

async function productServiceFixture({ tamperRelocation = false } = {}) {
  const sandbox = await mkdtemp(path.join(tmpdir(), "starvector-service-detach-"));
  const root = path.join(sandbox, "repo"), output = path.join(sandbox, "tuple"), weightsRoot = path.join(sandbox, "weights"), shimRoot = path.join(sandbox, "bin");
  await mkdir(root); await mkdir(shimRoot); await mkdir(path.join(weightsRoot, "app"), { recursive: true }); await mkdir(path.join(weightsRoot, "hf"), { recursive: true });
  await writeFile(path.join(root, ".gitignore"), "target/\n");
  await writeFile(path.join(root, "Cargo.toml"), `[dependencies]\ncandle-core = { git = "https://github.com/SceneWorks/inference", rev = "${pin}" }\n`);
  await execFile("git", ["init", "-q", root]);
  await execFile("git", ["-C", root, "add", "."]);
  await execFile("git", ["-C", root, "-c", "user.name=SceneWorks Test", "-c", "user.email=test@sceneworks.invalid", "commit", "-qm", "fixture"]);

  const runtime = path.join(sandbox, "fake-service.cjs");
  await writeFile(runtime, `const http = require("node:http");
const path = require("node:path");
if (path.basename(process.argv[1] ?? "") === "build") process.exit(0);
if (process.argv.length === 1) {
  const worker = process.env.SCENEWORKS_WORKER_ONLY === "1";
  let relocatedLibrary;
  let server;
  const timer = setInterval(() => {
    process.stdout.write((worker ? "worker" : "api") + " stdout " + process.pid + "\\n");
    process.stderr.write((worker ? "worker" : "api") + " stderr " + process.pid + "\\n");
  }, 25);
  if (!worker) {
    server = http.createServer((request, response) => {
      if (request.method === "POST" && request.url === "/api/v1/model-library/relocate") {
        let bytes = "";
        request.setEncoding("utf8");
        request.on("data", (chunk) => { bytes += chunk; });
        request.on("end", () => {
          const body = JSON.parse(bytes);
          if (body.path !== process.env.HF_HOME) { response.writeHead(409, { "content-type": "application/json" }); response.end(JSON.stringify({ detail: "wrong library" })); return; }
          relocatedLibrary = path.join(body.path, "hub");
          response.writeHead(200, { "content-type": "application/json" });
          response.end(JSON.stringify({ adopted: true, hfHome: body.path, libraryRoot: process.env.STARVECTOR_TEST_TAMPER_RELOCATION === "1" ? path.join(path.dirname(body.path), "tampered", "hub") : relocatedLibrary }));
        });
        return;
      }
      if (request.method === "GET" && request.url === "/api/v1/model-library") {
        response.writeHead(200, { "content-type": "application/json" });
        response.end(JSON.stringify({ available: relocatedLibrary !== undefined, probeStatus: relocatedLibrary ? "available" : "identity_mismatch", configuredLibraryPath: path.join(process.env.HF_HOME, "hub"), expectedLibrary: relocatedLibrary ? { configuredPath: relocatedLibrary } : null }));
        return;
      }
      response.writeHead(200, { "content-type": "application/json" });
      response.end(JSON.stringify({ status: "ok", readiness: { status: "ready" } }));
    });
    server.listen(Number(process.env.SCENEWORKS_API_PORT), "127.0.0.1");
  }
  const stop = () => {
    clearInterval(timer);
    if (server) server.close(() => process.exit(0)); else process.exit(0);
  };
  process.on("SIGTERM", stop);
  process.on("SIGINT", stop);
}
`);
  await executableAlias(path.join(shimRoot, process.platform === "win32" ? "cargo.exe" : "cargo"));
  await executableAlias(path.join(root, "target", "debug", process.platform === "win32" ? "sceneworks-rust-api.exe" : "sceneworks-rust-api"));
  await writeFile(path.join(weightsRoot, "app", "receipt.json"), "receipt");
  await writeFile(path.join(weightsRoot, "hf", "weights.bin"), "weights");
  const manifest = {
    terminal_service_closure: {
      app_data_relative_path: "app",
      app_data_sha256: await closureTreeHash(path.join(weightsRoot, "app")),
      hf_home_relative_path: "hf",
      hf_home_sha256: await closureTreeHash(path.join(weightsRoot, "hf")),
    },
    models: {
      "starvector-1b": { relative_path: "models/1b", revision: "1".repeat(40), inventory_sha256: "a".repeat(64) },
      "starvector-8b": { relative_path: "models/8b", revision: "2".repeat(40), inventory_sha256: "b".repeat(64) },
    },
  };
  const manifestPath = path.join(weightsRoot, "starvector-terminal-weights-v1.json");
  await writeFile(manifestPath, JSON.stringify(manifest));
  const cliEnv = { ...process.env, NODE_OPTIONS: `${process.env.NODE_OPTIONS ?? ""} --require="${runtime}"`.trim(), PATH: `${shimRoot}${path.delimiter}${process.env.PATH ?? ""}`, STARVECTOR_TEST_TAMPER_RELOCATION: tamperRelocation ? "1" : "0" };
  const serviceScript = path.resolve("scripts/starvector-terminal-product-service.mjs");
  return { sandbox, root, output, weightsRoot, manifest, manifestPath, cliEnv, serviceScript };
}

async function runProductServiceCli(fixture, command, port, timeout = 8_000) {
  const args = command === "start"
    ? [fixture.serviceScript, "start", fixture.root, fixture.output, pin, `http://127.0.0.1:${port}`, fixture.weightsRoot]
    : [fixture.serviceScript, "stop", fixture.output];
  return execFile(process.execPath, args, { env: fixture.cliEnv, timeout });
}

function assertPidsRunning(record) {
  for (const pid of [record.api_pid, record.worker_pid]) assert.doesNotThrow(() => process.kill(pid, 0));
}

async function forceFixtureCleanup(fixture, record) {
  if (record) {
    await runProductServiceCli(fixture, "stop", 0).catch(() => {});
    for (const pid of [record.worker_pid, record.api_pid]) { try { process.kill(pid, "SIGKILL"); } catch { /* already stopped */ } }
  }
  await rm(fixture.sandbox, { recursive: true, force: true });
}

test("terminal workflow is dispatch-only, serial, and seals raw evidence", () => {
  assert.match(workflow, /^\s+workflow_dispatch:/m);
  assert.doesNotMatch(workflow, /^\s+(push|pull_request|schedule):/m);
  for (const edge of ["needs: mlx-1b", "needs: mlx-8b", "needs: cuda-1b"]) assert.match(workflow, new RegExp(edge));
  assert.match(workflow, /needs: \[mlx-1b, mlx-8b, cuda-1b, cuda-8b\]/);
  assert.match(workflow, /starvector-terminal-producer\.mjs run/g);
  assert.match(workflow, /starvector-terminal-producer\.mjs seal/);
  assert.match(workflow, /STARVECTOR_TERMINAL_LEASE_ROOT: \/Users\/Shared\/SceneWorks\/terminal-leases/);
  assert.match(workflow, /STARVECTOR_TERMINAL_LEASE_ROOT: C:\\\\ProgramData\\\\SceneWorks\\\\terminal-leases/);
  assert.match(workflow, /scripts\/starvector-terminal-route\.mjs/);
  assert.equal((workflow.match(/starvector-terminal-product-service\.mjs start/g) ?? []).length, 4);
  assert.equal((workflow.match(/starvector-terminal-product-service\.mjs stop/g) ?? []).length, 4);
  assert.equal((workflow.match(/starvector-terminal-case-bundle\.mjs/g) ?? []).length, 4);
  assert.equal((workflow.match(/starvector-terminal-assets\.mjs/g) ?? []).length, 4);
  assert.match(workflow, /starvector_terminal_lease/);
  assert.equal((workflow.match(/cargo build --release --locked -p sceneworks-worker --bin starvector_terminal_lease/g) ?? []).length, 4);
  assert.doesNotMatch(workflow, /RUNNER_TEMP[^\n]*\.lease/);
  assert.match(workflow, /Upload combined evidence even on failure/);
  assert.equal((workflow.match(/timeout-minutes: 720/g) ?? []).length, 4);
});

test("every tuple stops its service before uploading final logs and stop provenance", () => {
  const jobs = ["mlx-1b", "mlx-8b", "cuda-1b", "cuda-8b"];
  for (let index = 0; index < jobs.length; index += 1) {
    const start = workflow.indexOf(`  ${jobs[index]}:`), end = index + 1 < jobs.length ? workflow.indexOf(`  ${jobs[index + 1]}:`) : workflow.indexOf("  seal-receipt:");
    const job = workflow.slice(start, end), stop = job.indexOf("starvector-terminal-product-service.mjs stop"), upload = job.indexOf("uses: actions/upload-artifact@");
    assert.ok(stop >= 0 && upload >= 0 && stop < upload, `${jobs[index]} must stop and seal logs before artifact upload`);
    assert.match(job.slice(job.lastIndexOf("- name:", stop), upload), /if: \$\{\{ always\(\) \}\}/);
  }
});

test("terminal workflow has no install or model download step", () => {
  assert.doesNotMatch(workflow, /(?:pip|npm|cargo)\s+install|huggingface-cli|curl .*models|wget .*models/i);
  assert.match(workflow, /STARVECTOR_TERMINAL_WEIGHTS_ROOT/);
  assert.match(workflow, /STARVECTOR_TERMINAL_METRICS_ROOT/);
  assert.match(workflow, /STARVECTOR_TERMINAL_METRICS_PYTHON/);
  assert.match(workflow, /STARVECTOR_TERMINAL_CASE_BUNDLE/);
  assert.match(workflow, /STARVECTOR_TERMINAL_CORPUS_ASSETS_ROOT/);
  assert.match(workflow, /STARVECTOR_TERMINAL_NO_JOB_DOWNLOADS: "1"/);
  assert.match(workflow, /cross-repository lease/);
  assert.match(workflow, /starvector-terminal-metrics-environment-v1\.json/);
});

test("source-built product service enables the native backend for each campaign host", () => {
  assert.deepEqual(productServiceBuildArgs("darwin"), ["build", "--locked", "-p", "sceneworks-rust-api"]);
  assert.deepEqual(productServiceBackendEnv("darwin"), {});
  assert.deepEqual(productServiceBuildArgs("win32"), ["build", "--locked", "-p", "sceneworks-rust-api", "--features", "backend-candle"]);
  assert.deepEqual(productServiceBackendEnv("win32"), { SCENEWORKS_BACKEND_CANDLE_ENABLED: "true" });
  assert.equal(productServiceStateRoot(path.join("tmp", "tuple")), path.join("tmp", "tuple-product-service-state"));
  assert.notEqual(productServiceStateRoot(path.join("tmp", "tuple")), path.join("tmp", "tuple", "product-service-state"));
});

test("product service streams copied closure identity and preserves portable ordering", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "starvector-service-copy-")), source = path.join(root, "source"), destination = path.join(root, "destination");
  await mkdir(path.join(source, "nested"), { recursive: true });
  await writeFile(path.join(source, "z.bin"), "z"); await writeFile(path.join(source, "nested", "a.bin"), "a");
  const copied = [];
  const digestFile = async (file) => { copied.push(path.relative(root, file).split(path.sep).join("/")); return hash(await readFile(file)); };
  await copyRegularTree(source, destination, digestFile);
  assert.equal(copied.length, 4);
  const hashed = [];
  const identity = await closureTreeHash(destination, async (file) => { hashed.push(file); return hash(await readFile(file)); });
  const rows = [terminalTreeEntry("nested/a.bin", 1, hash("a")), terminalTreeEntry("z.bin", 1, hash("z"))];
  assert.equal(identity, terminalTreeSha256(rows));
  assert.equal(hashed.length, 2);
  await symlink(path.join(source, "z.bin"), path.join(source, "link"));
  await assert.rejects(() => copyRegularTree(source, path.join(root, "rejected")), /weights closure rejects symlink/);
  await symlink(path.join(destination, "z.bin"), path.join(destination, "link"));
  await assert.rejects(() => closureTreeHash(destination), /weights closure copy contains symlink/);
});

test("product service start CLI exits while durable-log services remain alive and can later be stopped", { timeout: 30_000 }, async () => {
  const fixture = await productServiceFixture(), port = await availableLoopbackPort();
  let record;
  try {
    await runProductServiceCli(fixture, "start", port);
    record = JSON.parse(await readFile(path.join(fixture.output, "product-service-provenance.json"), "utf8"));
    assertPidsRunning(record);
    assert.deepEqual(record.offline.library_relocation, {
      adopted: true,
      hf_home: path.relative(fixture.output, path.join(productServiceStateRoot(fixture.output), "hf")),
      library_root: path.relative(fixture.output, path.join(productServiceStateRoot(fixture.output), "hf", "hub")),
      probe_status: "available",
    });
    const logPaths = productServiceLogPaths(fixture.output);
    assert.deepEqual(record.logs, Object.fromEntries(Object.entries(logPaths).map(([name, file]) => [name, path.relative(fixture.output, file)])));
    const initialSizes = await Promise.all(Object.values(logPaths).map(async (file) => (await stat(file)).size));
    await new Promise((resolve) => setTimeout(resolve, 150));
    const laterSizes = await Promise.all(Object.values(logPaths).map(async (file) => (await stat(file)).size));
    for (let index = 0; index < initialSizes.length; index += 1) assert.ok(laterSizes[index] > initialSizes[index], "service log stopped after the start CLI exited");

    await runProductServiceCli(fixture, "stop", 0);
    const stopped = JSON.parse(await readFile(path.join(fixture.output, "product-service-stopped.json"), "utf8"));
    const logs = await productServiceLogsIdentity(fixture.output);
    assert.equal(stopped.status, "stopped");
    assert.deepEqual(stopped.logs, logs);
    assert.equal(logs.entries.length, 4);
    assert.equal(logs.sha256, terminalTreeSha256(logs.entries));
    for (const entry of logs.entries) {
      const file = path.join(fixture.output, entry.path);
      assert.deepEqual(Object.keys(entry), ["path", "byte_size", "sha256"]);
      assert.equal(entry.byte_size, (await stat(file)).size);
      assert.equal(entry.sha256, hash(await readFile(file)));
    }
    assert.deepEqual(JSON.parse(await readFile(path.join(fixture.output, "product-service-logs.json"), "utf8")), logs);
    assert.equal((await readFile(path.join(fixture.output, "product-service-logs.sha256"), "utf8")).trim(), logs.sha256);
    for (const pid of [record.api_pid, record.worker_pid]) assert.throws(() => process.kill(pid, 0), { code: "ESRCH" });
    assert.equal(await lstat(productServiceStateRoot(fixture.output)).catch((error) => error.code === "ENOENT" ? null : Promise.reject(error)), null);
    record = null;
  } finally {
    await forceFixtureCleanup(fixture, record);
  }
});

test("product service stop requires an exact active sentinel before signaling", { timeout: 25_000 }, async () => {
  const fixture = await productServiceFixture(), port = await availableLoopbackPort();
  let record;
  try {
    await runProductServiceCli(fixture, "start", port);
    record = JSON.parse(await readFile(path.join(fixture.output, "product-service-provenance.json"), "utf8"));
    const activePath = productServiceActiveStatePath(fixture.output), activeText = await readFile(activePath, "utf8"), active = JSON.parse(activeText);
    assert.equal(active.instance_token, record.instance_token);
    assert.match(record.instance_token, /^[a-f0-9]{64}$/);

    await rm(activePath);
    await assert.rejects(() => runProductServiceCli(fixture, "stop", 0), /product-service-active\.json|ENOENT/);
    assertPidsRunning(record);
    await writeFile(activePath, activeText, { flag: "wx", mode: 0o600 });

    await writeFile(activePath, JSON.stringify({ ...active, instance_token: "0".repeat(64) }));
    await assert.rejects(() => runProductServiceCli(fixture, "stop", 0), /mismatches provenance field instance_token/);
    assertPidsRunning(record);

    await writeFile(activePath, JSON.stringify({ ...active, api_pid: process.pid }));
    await assert.rejects(() => runProductServiceCli(fixture, "stop", 0), /mismatches provenance field api_pid/);
    assertPidsRunning(record);

    await writeFile(activePath, JSON.stringify({ ...active, status: "stopped" }));
    await assert.rejects(() => runProductServiceCli(fixture, "stop", 0), /active product service state shape is invalid/);
    assertPidsRunning(record);

    await writeFile(activePath, activeText);
    await runProductServiceCli(fixture, "stop", 0);
    await assert.rejects(
      () => stopProductService(fixture.output, { terminate: async () => assert.fail("double-stop must reject before termination") }),
      /already stopped/,
    );
    record = null;
  } finally {
    await forceFixtureCleanup(fixture, record);
  }
});

test("product service rejects a pre-existing healthy listener before provenance", { timeout: 15_000 }, async () => {
  const fixture = await productServiceFixture(), listener = createServer((_request, response) => response.end(JSON.stringify({ status: "ok", readiness: { status: "ready" } })));
  try {
    await new Promise((resolve, reject) => { listener.once("error", reject); listener.listen(0, "127.0.0.1", resolve); });
    const port = listener.address().port;
    await assert.rejects(() => runProductServiceCli(fixture, "start", port), /API port is already occupied/);
    assert.equal(listener.listening, true);
    assert.equal(await lstat(path.join(fixture.output, "product-service-provenance.json")).catch((error) => error.code === "ENOENT" ? null : Promise.reject(error)), null);
  } finally {
    await new Promise((resolve) => listener.close(resolve));
    await forceFixtureCleanup(fixture);
  }
});

test("product service relocation requires the exact adopted offline path", async () => {
  const expected = path.resolve(tmpdir(), "starvector-relocated-hf");
  const ok = await relocateProductServiceLibrary("http://127.0.0.1:1", expected, {
    fetchImpl: async (url, init) => {
      assert.ok(init.signal);
      if (url.pathname === "/api/v1/model-library/relocate") {
        assert.deepEqual(JSON.parse(init.body), { path: expected });
        assert.equal(init.method, "POST");
        return { ok: true, json: async () => ({ adopted: true, hfHome: expected, libraryRoot: path.join(expected, "hub") }) };
      }
      assert.equal(url.pathname, "/api/v1/model-library");
      return { ok: true, json: async () => ({ available: true, probeStatus: "available", configuredLibraryPath: path.join(expected, "hub"), expectedLibrary: { configuredPath: path.join(expected, "hub") } }) };
    },
  });
  assert.deepEqual(ok, { adopted: true, hf_home: expected, library_root: path.join(expected, "hub"), probe_status: "available" });
  for (const body of [
    { adopted: false, hfHome: expected, libraryRoot: path.join(expected, "hub") },
    { adopted: true, hfHome: path.dirname(expected), libraryRoot: path.join(expected, "hub") },
    { adopted: true, hfHome: expected, libraryRoot: path.join(path.dirname(expected), "other", "hub") },
  ]) {
    await assert.rejects(
      () => relocateProductServiceLibrary("http://127.0.0.1:1", expected, { fetchImpl: async () => ({ ok: true, json: async () => body }) }),
      /returned an inexact binding/,
    );
  }
  await assert.rejects(
    () => relocateProductServiceLibrary("http://127.0.0.1:1", expected, {
      fetchImpl: async (url) => ({ ok: true, json: async () => url.pathname.endsWith("/relocate")
        ? { adopted: true, hfHome: expected, libraryRoot: path.join(expected, "hub") }
        : { available: false, probeStatus: "identity_mismatch", configuredLibraryPath: path.join(expected, "hub"), expectedLibrary: { configuredPath: path.join(expected, "hub") } } }),
    }),
    /did not read back as the exact available binding/,
  );
});

test("a rejected product relocation terminates both services and records the failure", { timeout: 15_000 }, async () => {
  const fixture = await productServiceFixture({ tamperRelocation: true }), port = await availableLoopbackPort();
  try {
    await assert.rejects(() => runProductServiceCli(fixture, "start", port), /returned an inexact binding/);
    const failure = JSON.parse(await readFile(path.join(fixture.output, "product-service-start-failed.json"), "utf8"));
    assert.equal(failure.cleanup.status, "terminated");
    assert.equal(failure.cleanup.state_retained, false);
    assert.match(failure.error, /returned an inexact binding/);
    for (const pid of [failure.api_pid, failure.worker_pid]) assert.throws(() => process.kill(pid, 0), { code: "ESRCH" });
    assert.equal(await lstat(productServiceStateRoot(fixture.output)).catch((error) => error.code === "ENOENT" ? null : Promise.reject(error)), null);
    assert.equal(await lstat(path.join(fixture.output, "product-service-provenance.json")).catch((error) => error.code === "ENOENT" ? null : Promise.reject(error)), null);
  } finally {
    await forceFixtureCleanup(fixture);
  }
});

test("product service records setup and log-open failures before removing child-free state", { timeout: 20_000 }, async () => {
  for (const failure of ["setup", "hf-hash", "open"]) {
    const fixture = await productServiceFixture(), port = await availableLoopbackPort();
    try {
      if (failure === "setup") await writeFile(fixture.manifestPath, JSON.stringify({ ...fixture.manifest, terminal_service_closure: { ...fixture.manifest.terminal_service_closure, app_data_sha256: "0".repeat(64) } }));
      else if (failure === "hf-hash") await writeFile(path.join(fixture.weightsRoot, "hf", "weights.bin"), "tampered");
      else { await mkdir(fixture.output); await writeFile(productServiceLogPaths(fixture.output).api_stdout, "stale log"); }
      await assert.rejects(() => runProductServiceCli(fixture, "start", port));
      const report = JSON.parse(await readFile(path.join(fixture.output, "product-service-start-failed.json"), "utf8"));
      assert.equal(report.status, "failed");
      assert.equal(report.cleanup.status, "not_needed");
      assert.equal(report.cleanup.state_retained, false);
      assert.match(report.logs.sha256, /^[a-f0-9]{64}$/);
      assert.equal((await readFile(path.join(fixture.output, "product-service-logs.sha256"), "utf8")).trim(), report.logs.sha256);
      assert.equal(await lstat(productServiceStateRoot(fixture.output)).catch((error) => error.code === "ENOENT" ? null : Promise.reject(error)), null);
    } finally {
      await forceFixtureCleanup(fixture);
    }
  }
});

test("failed termination records diagnostics and retains authenticated state for safe retry", { timeout: 20_000 }, async () => {
  const fixture = await productServiceFixture(), port = await availableLoopbackPort();
  let record;
  try {
    await runProductServiceCli(fixture, "start", port);
    record = JSON.parse(await readFile(path.join(fixture.output, "product-service-provenance.json"), "utf8"));
    await assert.rejects(() => stopProductService(fixture.output, { terminate: async () => { throw new Error("injected termination failure"); } }), /injected termination failure/);
    assertPidsRunning(record);
    assert.equal((await lstat(productServiceStateRoot(fixture.output))).isDirectory(), true);
    const report = JSON.parse(await readFile(path.join(fixture.output, "product-service-stop-failed.json"), "utf8"));
    assert.equal(report.status, "failed");
    assert.deepEqual(report.cleanup, { status: "failed", error: "injected termination failure", state_retained: true });
    assert.match(report.logs.sha256, /^[a-f0-9]{64}$/);
    assert.equal((await readFile(path.join(fixture.output, "product-service-logs.sha256"), "utf8")).trim(), report.logs.sha256);
    await runProductServiceCli(fixture, "stop", 0);
    record = null;
  } finally {
    await forceFixtureCleanup(fixture, record);
  }
});

test("product service Windows termination uses bounded process-tree taskkill", () => {
  assert.deepEqual(productServiceTaskkillArguments(4242), ["/PID", "4242", "/T", "/F"]);
  assert.throws(() => productServiceTaskkillArguments(0), /invalid product service PID/);
});

test("readiness workflow is an identity-only dispatch on both campaign hosts", () => {
  assert.match(readiness, /^\s+workflow_dispatch:/m);
  assert.doesNotMatch(readiness, /^\s+(push|pull_request|schedule):/m);
  assert.match(readiness, /runs-on: \[self-hosted, macOS, ARM64, rw-starvector\]/);
  assert.equal((workflow.match(/runs-on: \[self-hosted, macOS, ARM64, rw-starvector\]/g) ?? []).length, 3);
  assert.match(readiness, /runs-on: \[self-hosted, Windows, X64, cuda, real-weights\]/);
  assert.equal((readiness.match(/starvector-terminal-readiness\.mjs/g) ?? []).length, 2);
  assert.equal((readiness.match(/if: \$\{\{ always\(\) \}\}/g) ?? []).length, 2);
  assert.match(readiness, /permanent_pin:[\s\S]*required: true/);
  assert.equal((readiness.match(/starvector-terminal-pin-paths\.mjs/g) ?? []).length, 2);
  assert.equal((workflow.match(/starvector-terminal-pin-paths\.mjs/g) ?? []).length, 5);
  assert.equal((readiness.match(/\$\{\{ inputs\.permanent_pin \}\}/g) ?? []).length, 4);
  assert.equal((workflow.match(/\$\{\{ inputs\.permanent_pin \}\}/g) ?? []).length >= 15, true);
  assert.doesNotMatch(`${workflow}\n${readiness}`, /starvector-terminal[\\/]inference(?:[\\/]|\s|$)/);
  assert.doesNotMatch(`${workflow}\n${readiness}`, /starvector-terminal[\\/]inference-preflight(?:[\\/]|\s|$)/);
  assert.doesNotMatch(`${workflow}\n${readiness}`, /starvector-terminal[\\/]corpora[\\/]starvector-terminal-v1/);
  assert.doesNotMatch(`${workflow}\n${readiness}`, /\/opt\/sceneworks-terminal/);
  assert.match(readiness, /STARVECTOR_TERMINAL_WEIGHTS_ROOT: D:\\\\sceneworks-terminal\\\\weights/);
  assert.match(readiness, /STARVECTOR_TERMINAL_METRICS_ROOT: D:\\\\sceneworks-terminal\\\\metrics/);
  assert.doesNotMatch(readiness, /campaign_run_id|concurrency:/);
});

test("readiness workflow cannot start services, claim leases, or execute models", () => {
  assert.doesNotMatch(readiness, /starvector-terminal-product-service|starvector-terminal-producer\.mjs|starvector_terminal_lease|STARVECTOR_TERMINAL_LEASE|vector_generate|cargo\s+(?:build|run)|(?:pip|npm|cargo)\s+install|huggingface-cli|\bcurl\b|\bwget\b/i);
  assert.match(readiness, /Upload macOS readiness report even on failure/);
  assert.match(readiness, /Upload Windows readiness report even on failure/);
});

test("readiness CLI writes a structured failure report before returning nonzero", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "starvector-readiness-report-")), output = path.join(root, "nested", "report.json");
  await assert.rejects(() => execFile(process.execPath, ["scripts/starvector-terminal-readiness.mjs", root, path.join(root, "missing-plan.json"), root, root, root, process.execPath, root, "0".repeat(40), output]));
  const report = JSON.parse(await readFile(output, "utf8"));
  assert.equal(report.schema_version, 1); assert.equal(report.kind, "starvector_terminal_readiness"); assert.equal(report.status, "failed"); assert.match(report.error, /ENOENT/);
});

test("readiness validates the complete service tree closure without materializing it", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "starvector-readiness-weights-"));
  await mkdir(path.join(root, "app")); await mkdir(path.join(root, "hf"));
  await writeFile(path.join(root, "app", "receipts.json"), "receipts"); await writeFile(path.join(root, "hf", "weights.bin"), "weights");
  const tree = async (name) => {
    const file = path.join(root, name, name === "app" ? "receipts.json" : "weights.bin"), bytes = await readFile(file);
    return terminalTreeSha256([terminalTreeEntry(path.basename(file), bytes.length, hash(bytes))]);
  };
  const weights = { models: { "starvector-1b": {}, "starvector-8b": {} }, terminal_service_closure: { app_data_relative_path: "app", app_data_sha256: await tree("app"), hf_home_relative_path: "hf", hf_home_sha256: await tree("hf") } };
  const result = await validateTerminalServiceClosure(root, weights);
  assert.equal(result.app_data.file_count, 1); assert.equal(result.hf_home.file_count, 1);
  const streamed = [];
  assert.equal((await treeIdentity(root, "hf", "terminal HF closure", async (file) => { streamed.push(file); return hash(await readFile(file)); })).sha256, weights.terminal_service_closure.hf_home_sha256);
  assert.equal(streamed.length, 1);
  await writeFile(path.join(root, "hf", "weights.bin"), "drift");
  await assert.rejects(() => validateTerminalServiceClosure(root, weights), /service closure tree hash mismatch/);
});

test("readiness binds all 120 source assets and every suite identity to the pinned corpus", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "starvector-readiness-corpus-")), inference = path.join(root, "inference"), assets = path.join(root, "assets");
  await mkdir(path.join(inference, "release"), { recursive: true }); await mkdir(path.join(inference, "scripts", "release"), { recursive: true }); await mkdir(assets);
  for (const [name, bytes] of [["source.svg", "svg"], ["input.png", "input"], ["reference.png", "reference"]]) await writeFile(path.join(assets, name), bytes);
  const sources = Array.from({ length: 4 }, (_, index) => ({ dataset: `dataset-${index}`, revision: String(index + 1).repeat(40), row_identity_sha256: "" }));
  const rows = Array.from({ length: 120 }, (_, case_index) => ({ case_index, dataset: sources[Math.floor(case_index / 30)].dataset, revision: sources[Math.floor(case_index / 30)].revision, row_index: case_index % 30, filename: `${case_index}.svg`, svg_path: "source.svg", svg_sha256: hash("svg"), input_png_path: "input.png", png_sha256: hash("input"), reference_png: "reference.png", reference_png_sha256: hash("reference") }));
  const record = (row) => JSON.stringify({ dataset: row.dataset, revision: row.revision, row_index: row.row_index, filename: row.filename, svg_sha256: row.svg_sha256 });
  sources.forEach((source, index) => { source.row_identity_sha256 = hash(`${rows.slice(index * 30, index * 30 + 30).map(record).join("\n")}\n`); });
  const rowIdentity = hash(`${rows.map(record).join("\n")}\n`);
  const parityIdentity = hash(`${sources.flatMap((_, index) => rows.slice(index * 30, index * 30 + 5)).map(record).join("\n")}\n`);
  const prompts = Array.from({ length: 60 }, (_, case_index) => { const prompt = `prompt-${case_index}`; return { case_index, case_id: `prompt-v1-${case_index}`, prompt, prompt_sha256: hash(prompt), raster_model: "raster", vector_model: "starvector_8b", expected_raster_revision: "raster-revision", expected_vector_revision: "vector-revision" }; });
  const corpus = { upstream_image_quality_cases: { row_identity_sha256: rowIdentity, sources }, deterministic_parity_cases: { row_identity_sha256: parityIdentity }, sceneworks_owned_suites: { prompt_composition: { content_identity_sha256: hash(prompts.map((entry) => entry.prompt_sha256).join("\n")) } } };
  await writeFile(path.join(inference, "release", "corpus.json"), JSON.stringify(corpus));
  await writeFile(path.join(inference, "scripts", "release", "starvector_terminal_evidence.mjs"), `import { createHash } from "node:crypto"; export function validatePlan(value) { return createHash("sha256").update(JSON.stringify(value)).digest("hex"); }\n`);
  const lifecycle = ["load", "unload", "reload", "memory_reported"], limits = ["complete_root", "eos", "token_limit", "byte_limit", "wall_time_limit", "cancelled"];
  const index = { inference_revision: pin, row_identity_sha256: rowIdentity, rows, lifecycle_cases: Object.fromEntries(["mlx:1b", "mlx:8b", "candle-cuda:1b", "candle-cuda:8b"].map((tuple) => [tuple, lifecycle.map((operation) => ({ case_id: `${tuple}-${operation}`, operation }))])), limit_cases: Object.fromEntries(["mlx:1b", "mlx:8b", "candle-cuda:1b", "candle-cuda:8b"].map((tuple) => [tuple, limits.map((finish_reason) => ({ case_id: `${tuple}-${finish_reason}`, finish_reason }))])), prompt_composition: prompts };
  await writeFile(path.join(assets, "starvector-terminal-row-index-v1.json"), JSON.stringify(index));
  const result = await validateCorpusAssets(inference, "release/corpus.json", assets, pin);
  assert.equal(result.asset_file_references, 360); assert.equal(result.prompt_sha256, corpus.sceneworks_owned_suites.prompt_composition.content_identity_sha256);
  await writeFile(path.join(assets, "input.png"), "drift");
  await assert.rejects(() => validateCorpusAssets(inference, "release/corpus.json", assets, pin), /input PNG hash mismatch/);
});
