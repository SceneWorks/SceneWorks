// sc-26687 S8 real-browser proof driver.
//
// Runs ONLY on an ephemeral GitHub-hosted runner. It starts the isolated fixture
// Vite server (verify-fixtures.mjs, unknown /api -> 404, no forwarding), drives the
// real apps/web frontend in Chromium, and writes screenshots, HTML, console/network
// logs and the serialized-request evidence.
//
// Local execution is refused by guard.mjs. Unit tests for the guard live in
// guard.test.mjs and never start a server or a browser.
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, realpathSync, writeFileSync, writeSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import {
  PLAYWRIGHT_VERSION,
  REQUIRED_CASES,
  S8_MERGE_COMMIT,
  VERIFICATION_OVERLAY_FILES,
  VERIFICATION_OVERLAY_PREFIXES,
  assertPlaywrightVersion,
  assertSummaryPassed,
  buildSummary,
  evaluateEnvironment,
  evaluateSourceIdentity,
  evaluateCaseFieldSelectors,
  productionWebDigest,
  loadPlaywrightCommonJs,
  resolveArtifactDir,
  resolveBrowsersPath,
  resolvePlaywrightEntry,
} from "./guard.mjs";
import { CASES } from "./cases.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const WEB_ROOT = resolve(HERE, "../..");
const REPO_ROOT = resolve(WEB_ROOT, "../..");
const CONFIG_PATH = resolve(WEB_ROOT, "vite.verify.config.mjs");
const FIXTURES_PATH = resolve(WEB_ROOT, "verify-fixtures.mjs");
const PORT = Number(process.env.SC26687_VERIFY_PORT || 5317);

// Every wait in the proof is BOUNDED. A step that hangs must fail the proof, not hold the hosted
// job open until the job's own timeout: the run would then be reported as cancelled and a proof
// that actually ran could never be recorded as green. The bound is generous — the operations
// bounded here are sub-second in practice — and the timeout rejects, so the caller decides
// whether the failure is fatal.
const BOUND_MS = { browser: 15000, context: 15000, server: 10000, request: 10000 };

// Write a reason straight to the fd. A stream write to a pipe is asynchronous, so an immediate
// exit can truncate it — and a DEFERRED exit is worse: the run keeps going after the refusal, so
// later stages can still execute and the refusal receipt can be overwritten with a later stage
// instead of the root cause. A synchronous write hands the bytes to the OS before returning, so
// the process can stop at once with its reason intact.
function writeAll(fd, text) {
  const buffer = Buffer.from(String(text));
  let offset = 0;
  while (offset < buffer.length) {
    try {
      offset += writeSync(fd, buffer, offset, buffer.length - offset);
    } catch (error) {
      if (error.code === "EAGAIN") continue;
      return; // the stream is gone; the receipt file still carries the reason
    }
  }
}
async function bounded(label, work, ms) {
  let timer;
  try {
    return await Promise.race([
      work,
      new Promise((_, rejectBound) => {
        timer = setTimeout(() => rejectBound(new Error(`${label} did not complete within ${ms}ms`)), ms);
      }),
    ]);
  } finally {
    clearTimeout(timer);
  }
}

function git(args) {
  return execFileSync("git", args, { cwd: REPO_ROOT, encoding: "utf8" }).trim();
}

function gitRaw(args) {
  return execFileSync("git", args, { cwd: REPO_ROOT, encoding: "utf8" });
}

function isAncestor(commit) {
  try {
    execFileSync("git", ["merge-base", "--is-ancestor", commit, "HEAD"], {
      cwd: REPO_ROOT,
      stdio: "ignore",
    });
    return true;
  } catch {
    return false;
  }
}

function sha256File(path) {
  return createHash("sha256").update(readFileSync(path)).digest("hex");
}

function createRecorder(page) {
  const entries = [];
  const byRequest = new WeakMap();
  page.on("request", (request) => {
    let body = null;
    try {
      body = request.postDataJSON();
    } catch {
      body = request.postData() ?? null;
    }
    const entry = {
      method: request.method(),
      url: request.url(),
      body,
      status: null,
      resourceType: request.resourceType(),
    };
    entries.push(entry);
    byRequest.set(request, entry);
  });
  page.on("response", (response) => {
    const entry = byRequest.get(response.request());
    if (entry) entry.status = response.status();
  });
  return { entries };
}

function attachLogs(page, logs) {
  page.on("console", (message) => {
    logs.console.push({ type: message.type(), text: message.text() });
  });
  page.on("pageerror", (error) => {
    logs.pageErrors.push({ message: error.message, stack: error.stack ?? null });
  });
  page.on("requestfailed", (request) => {
    logs.requestFailures.push({
      url: request.url(),
      method: request.method(),
      failure: request.failure()?.errorText ?? "unknown",
    });
  });
}

async function main() {
  const startedAt = new Date();
  const startedMs = Date.now();

  // Artifact directory is resolved first so refusals can leave a receipt too. Writing is
  // limited to real CI runs; a local refusal never drops files into the checkout.
  let resolvedArtifactDir = null;
  if (process.env.GITHUB_ACTIONS === "true") {
    try {
      resolvedArtifactDir = resolveArtifactDir(process.env, { repoRoot: REPO_ROOT });
    } catch {
      resolvedArtifactDir = resolve(REPO_ROOT, "browser-artifacts/sc26687");
    }
    try {
      mkdirSync(resolvedArtifactDir, { recursive: true });
    } catch {
      resolvedArtifactDir = null;
    }
  }

  const unexecutedCases = (error) =>
    REQUIRED_CASES.map((id) => ({ id, status: "fail", executed: false, skipped: false, error }));

  const writeReceipt = (payload) => {
    if (!resolvedArtifactDir) return;
    try {
      writeFileSync(
        resolve(resolvedArtifactDir, payload.stage ? "refusal.json" : "summary.json"),
        JSON.stringify(payload, null, 2),
      );
    } catch {
      // a refused run must still exit non-zero even when the receipt cannot be written
    }
  };

  let refusalSource = {};
  // Where the proof had got to when it failed, so the refusal receipt names the stage instead of
  // blaming startup for a failure that happened after the browser was already running.
  let proofStage = "runtime-startup";
  const refuse = (stage, errors) => {
    const summary = buildSummary({
      cases: unexecutedCases(stage),
      startedAt: startedAt.toISOString(),
      durationMs: Date.now() - startedMs,
      source: refusalSource,
      requests: {},
      artifacts: { directory: resolvedArtifactDir },
    });
    writeReceipt({ ...summary, stage, errors, ok: false });
    writeAll(2, `sc-26687 hosted browser proof refused at ${stage}:\n`);
    for (const error of errors) writeAll(2, ` - ${error}\n`);
  };

  // The refusal reasons go to stderr, which is ASYNCHRONOUS when it is a pipe: an immediate exit
  // can truncate them, and the reason is the whole point of a refusal. The receipt file keeps the
  // same reasons, but the log is what a reader sees first.
  const refuseAndExit = (stage, errors, code = 2) => {
    refuse(stage, errors);
    process.exit(code);
  };

  const guard = evaluateEnvironment(process.env, { repoRoot: REPO_ROOT, realpath: realpathSync });
  if (!guard.ok) {
    refuseAndExit("environment-guard", guard.errors);
  }

  const commit = git(["rev-parse", "HEAD"]);
  // The FULL tree includes this verification overlay; the production-frontend digest
  // excludes exactly the overlay paths and is the closure receipt.
  const webTree = git(["rev-parse", "HEAD:apps/web"]);
  const productionDigest = productionWebDigest(gitRaw(["ls-tree", "-r", "-z", "HEAD", "--", "apps/web"]));
  const expectedHead = process.env.SC26687_EXPECTED_HEAD ?? null;
  const s8IsAncestor = isAncestor(S8_MERGE_COMMIT);
  refusalSource = {
    commit,
    gitSha: process.env.GITHUB_SHA ?? null,
    expectedHead,
    s8MergeAncestor: S8_MERGE_COMMIT,
    s8MergeIsAncestor: s8IsAncestor,
    webTree,
    productionWebDigest: productionDigest,
    verificationOverlay: {
      files: VERIFICATION_OVERLAY_FILES,
      prefixes: VERIFICATION_OVERLAY_PREFIXES,
      excludedFromProductionDigest: true,
    },
  };
  const identity = evaluateSourceIdentity({
    head: commit,
    gitSha: process.env.GITHUB_SHA,
    expectedHead,
    s8Ancestor: s8IsAncestor,
  });
  if (!identity.ok) {
    refuseAndExit("source-identity", identity.errors);
  }
  // The managed-form case must read the import panel's fields through their label elements.
  // Checked BEFORE the browser starts: a case whose field selector cannot resolve times out on
  // the product's MARKUP instead of failing on its BEHAVIOUR, and would spend the whole run's
  // budget proving nothing about the import panel.
  const caseSelectors = evaluateCaseFieldSelectors(readFileSync(resolve(HERE, "cases.mjs"), "utf8"));
  if (!caseSelectors.ok) {
    refuseAndExit("case-field-selectors", caseSelectors.errors);
  }
  if (!expectedHead) {
    console.warn(
      "note: SC26687_EXPECTED_HEAD was not supplied; identity is recorded from GITHUB_SHA. " +
        "The production frontend closure digest is in summary.json.",
    );
  }

  if (!resolvedArtifactDir) {
    refuseAndExit("artifact-directory", ["could not create the bounded artifact directory"]);
  }
  const artifactDir = resolvedArtifactDir;

  let chromium, playwrightTools, playwrightBrowsersPath;
  try {
    // Explicit private per-run tools root only: never the apps/web npm graph or a
    // global resolution fallback. Install and browser CLI paths are handled by the workflow.
    playwrightTools = resolvePlaywrightEntry({ root: process.env.SC26687_PLAYWRIGHT_ROOT });
    playwrightBrowsersPath = resolveBrowsersPath(process.env, { root: process.env.SC26687_PLAYWRIGHT_ROOT });
    // Playwright's entry is CJS; require the explicit private entry so getter/re-export
    // shapes that ESM named-export detection misses still expose chromium.
    ({ chromium } = loadPlaywrightCommonJs({ entry: playwrightTools.entry }));
  } catch (error) {
    const message =
      "The playwright package is required from the isolated per-run tools root for this hosted proof.\n" +
      "Set SC26687_PLAYWRIGHT_ROOT to the private prefix the workflow installed " +
      `playwright@${PLAYWRIGHT_VERSION} into (and PLAYWRIGHT_BROWSERS_PATH under it).\n` +
      `Original error: ${error.message}`;
    refuseAndExit("dependency-contract", [message]);
  }
  const installedPlaywright = assertPlaywrightVersion(playwrightTools.version);
  try {
    assertPlaywrightVersion(installedPlaywright);
  } catch (error) {
    refuseAndExit("dependency-contract", [error.message]);
  }

  const results = [];
  const artifactFiles = [];
  let browser = null;
  let server = null;
  let wroteSummary = false;
  try {
    const { createServer } = await import("vite");
    server = await createServer({
      configFile: CONFIG_PATH,
      root: WEB_ROOT,
      logLevel: "warn",
      clearScreen: false,
      server: { host: "127.0.0.1", port: PORT, strictPort: true },
    });
    await server.listen();
    const baseUrl = server.resolvedUrls?.local?.[0] ?? `http://127.0.0.1:${PORT}/`;
    // Refuse to run if the fixture middleware ever forwards or answers an unknown API path
    // with something other than 404: that would mean a real service could be reached.
    const unknownProbe = await fetch(`http://127.0.0.1:${PORT}/api/__verify/no-such-endpoint`);
    if (unknownProbe.status !== 404) {
      throw new Error(`fixture harness answered an unknown API path with ${unknownProbe.status}, expected 404`);
    }
    browser = await chromium.launch({ headless: true });
    const context = await browser.newContext({ baseURL: baseUrl, viewport: { width: 1440, height: 1000 } });

    proofStage = "cases";
    for (const [index, testCase] of CASES.entries()) {
      const caseStarted = Date.now();
      const logs = { console: [], pageErrors: [], requestFailures: [] };
      let recorder = { entries: [] };
      let page = null;
      const prefix = `${String(index + 1).padStart(2, "0")}-${testCase.id}`;
      const result = { id: testCase.id, status: "fail", executed: false, skipped: false, durationMs: 0 };
      try {
        page = await context.newPage();
        recorder = createRecorder(page);
        attachLogs(page, logs);
        result.executed = true; // Count invocation, not page/setup failure as an executed case.
        const outcome = await testCase.run(page, recorder);
        result.status = "pass";
        result.evidence = outcome?.evidence ?? null;
      } catch (error) {
        result.error = error.message;
        result.stack = error.stack ?? null;
      } finally {
        if (page) {
          try {
            const screenshotPath = resolve(artifactDir, `${prefix}.png`);
            await page.screenshot({ path: screenshotPath, fullPage: true });
            artifactFiles.push(`${prefix}.png`);
          } catch {
            // screenshot is best-effort on a crashed page
          }
          try {
            writeFileSync(resolve(artifactDir, `${prefix}.html`), await page.content());
            artifactFiles.push(`${prefix}.html`);
          } catch {
            // ignore
          }
          const networkPath = resolve(artifactDir, `${prefix}.network.json`);
          writeFileSync(networkPath, JSON.stringify({ requests: recorder.entries, logs }, null, 2));
          artifactFiles.push(`${prefix}.network.json`);
          await page.close();
        }
        result.durationMs = Date.now() - caseStarted;
        results.push(result);
      }
    }

    // The context close and the evidence fetch are bounded too, and both run BEFORE the summary is
    // written — an unbounded hang in either would lose every artifact the proof produced. The
    // request log is the proof's record of what the UI actually asked the API for, so failing to
    // collect it fails the proof rather than writing a summary with an empty log.
    await bounded("browser context close", context.close(), BOUND_MS.context).catch((error) => {
      writeAll(2, `${error.message}\n`);
    });
    proofStage = "request-log";
    const serverLog = await bounded(
      "the mock API's request log",
      fetch(`http://127.0.0.1:${PORT}/api/__verify/requests`).then((response) => response.json()),
      BOUND_MS.request,
    );

    const summary = buildSummary({
      cases: results,
      startedAt: startedAt.toISOString(),
      durationMs: Date.now() - startedMs,
      source: {
        commit,
        gitSha: process.env.GITHUB_SHA ?? null,
        expectedHead,
        s8MergeAncestor: S8_MERGE_COMMIT,
        s8MergeIsAncestor: isAncestor(S8_MERGE_COMMIT),
        // Full checked-out tree INCLUDING this verification overlay (for transparency).
        webTree,
        // Tracked production frontend closure: same tree minus only the overlay paths below.
        productionWebDigest: productionDigest,
        verificationOverlay: {
          files: VERIFICATION_OVERLAY_FILES,
          prefixes: VERIFICATION_OVERLAY_PREFIXES,
          excludedFromProductionDigest: true,
        },
        configPath: CONFIG_PATH,
        configSha256: sha256File(CONFIG_PATH),
        fixturesPath: FIXTURES_PATH,
        fixturesSha256: sha256File(FIXTURES_PATH),
        driverSha256: sha256File(fileURLToPath(import.meta.url)),
        casesSha256: sha256File(resolve(HERE, "cases.mjs")),
        guardSha256: sha256File(resolve(HERE, "guard.mjs")),
        playwrightVersion: installedPlaywright,
        expectedPlaywrightVersion: PLAYWRIGHT_VERSION,
        playwrightToolsRoot: process.env.SC26687_PLAYWRIGHT_ROOT ?? null,
        playwrightBrowsersPath,
        browserVersion: browser.version(),
        nodeVersion: process.version,
        port: PORT,
        baseUrl,
        unknownApiStatus: 404,
      },
      requests: {
        count: serverLog.requests.length,
        videoJobPosts: serverLog.requests.filter((entry) => entry.method === "POST" && entry.path === "/api/v1/video/jobs"),
        imageJobPosts: serverLog.requests.filter((entry) => entry.method === "POST" && entry.path === "/api/v1/image/jobs"),
        downloadPosts: serverLog.requests.filter((entry) => entry.method === "POST" && entry.path.endsWith("/download")),
        importPosts: serverLog.requests.filter((entry) => entry.method === "POST" && entry.path === "/api/v1/models/import"),
        eventSourceOpens: serverLog.eventSourceOpens,
      },
      artifacts: {
        directory: artifactDir,
        files: [...artifactFiles, "summary.json", "mock-server-requests.json"],
      },
    });
    writeFileSync(resolve(artifactDir, "summary.json"), JSON.stringify(summary, null, 2));
    writeFileSync(resolve(artifactDir, "mock-server-requests.json"), JSON.stringify(serverLog, null, 2));
    wroteSummary = true;

    writeAll(1, `${JSON.stringify(summary.totals, null, 2)}\n`);
    writeAll(1, `artifacts: ${artifactDir}\n`);
    try {
      assertSummaryPassed(summary);
    } catch (error) {
      writeAll(2, `${error.message}\n`);
      process.exitCode = 1;
    }
  } catch (error) {
    // A failure at any stage must still leave a structured refusal receipt naming the stage it
    // happened in, rather than blaming startup for a failure that happened after the browser was
    // already running.
    if (!wroteSummary) refuse(proofStage, [error.message]);
    throw error;
  } finally {
    // Teardown is BOUNDED. By this point the proof's own result, every artifact and the
    // structured summary are already on disk, so a browser or server that refuses to close must
    // not be allowed to hold the job open until its timeout: the run would be reported as
    // cancelled and a proof that actually ran could never be recorded as green.
    await bounded("browser close", browser ? browser.close() : Promise.resolve(), BOUND_MS.browser).catch((error) => {
      writeAll(2, `${error.message}\n`);
    });
    // Destroy any socket the fixture's event streams still hold. The stream lives in the Vite
    // server process, so it cannot be closed from here directly; dropping the connections is
    // what lets the server's own close() complete.
    try {
      server?.httpServer?.closeAllConnections?.();
    } catch {
      // best effort
    }
    await bounded("server close", server ? server.close() : Promise.resolve(), BOUND_MS.server).catch((error) => {
      writeAll(2, `${error.message}\n`);
    });
  }
  // The proof is over and everything it claims is on disk. Exit explicitly, so that no lingering
  // handle can keep the job open.
  process.exit(process.exitCode ?? 0);
}

main().catch((error) => {
  writeAll(2, `${error.stack ?? error.message}\n`);
  process.exit(1);
});
