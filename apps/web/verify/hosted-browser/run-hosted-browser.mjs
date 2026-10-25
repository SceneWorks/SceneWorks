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
import { mkdirSync, readFileSync, realpathSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
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
  productionWebDigest,
  resolveArtifactDir,
} from "./guard.mjs";
import { CASES } from "./cases.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const WEB_ROOT = resolve(HERE, "../..");
const REPO_ROOT = resolve(WEB_ROOT, "../..");
const CONFIG_PATH = resolve(WEB_ROOT, "vite.verify.config.mjs");
const FIXTURES_PATH = resolve(WEB_ROOT, "verify-fixtures.mjs");
const PORT = Number(process.env.SC26687_VERIFY_PORT || 5317);

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
    console.error(`sc-26687 hosted browser proof refused at ${stage}:`);
    for (const error of errors) console.error(` - ${error}`);
  };

  const guard = evaluateEnvironment(process.env, { repoRoot: REPO_ROOT, realpath: realpathSync });
  if (!guard.ok) {
    refuse("environment-guard", guard.errors);
    process.exit(2);
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
    refuse("source-identity", identity.errors);
    process.exit(2);
  }
  if (!expectedHead) {
    console.warn(
      "note: SC26687_EXPECTED_HEAD was not supplied; identity is recorded from GITHUB_SHA. " +
        "The production frontend closure digest is in summary.json.",
    );
  }

  if (!resolvedArtifactDir) {
    refuse("artifact-directory", ["could not create the bounded artifact directory"]);
    process.exit(2);
  }
  const artifactDir = resolvedArtifactDir;

  let chromium;
  try {
    ({ chromium } = await import("playwright"));
  } catch (error) {
    const message =
      "The playwright package is required for this hosted proof.\n" +
      `Install exactly it in the job before running this driver:\n` +
      `  npm install --no-save --prefix apps/web playwright@${PLAYWRIGHT_VERSION}\n` +
      "  (cd apps/web && npx playwright install --with-deps chromium)\n" +
      `Original error: ${error.message}`;
    refuse("dependency-contract", [message]);
    process.exit(2);
  }
  const installedPlaywright = playwrightVersion();
  try {
    assertPlaywrightVersion(installedPlaywright);
  } catch (error) {
    refuse("dependency-contract", [error.message]);
    process.exit(2);
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
        const outcome = await testCase.run(page, recorder);
        result.status = "pass";
        result.executed = true;
        result.evidence = outcome?.evidence ?? null;
      } catch (error) {
        result.executed = true;
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

    await context.close();

    const serverLog = await fetch(`http://127.0.0.1:${PORT}/api/__verify/requests`).then((response) => response.json());

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

    console.log(JSON.stringify(summary.totals, null, 2));
    console.log(`artifacts: ${artifactDir}`);
    try {
      assertSummaryPassed(summary);
    } catch (error) {
      console.error(error.message);
      process.exitCode = 1;
    }
  } catch (error) {
    // A startup failure (Vite or Chromium) must still leave a structured refusal receipt.
    if (!wroteSummary) refuse("runtime-startup", [error.message]);
    throw error;
  } finally {
    try {
      if (browser) await browser.close();
    } finally {
      if (server) await server.close();
    }
  }
}

function playwrightVersion() {
  try {
    const require = createRequire(import.meta.url);
    const manifest = JSON.parse(readFileSync(require.resolve("playwright/package.json"), "utf8"));
    return manifest.version ?? "unknown";
  } catch {
    return "unknown";
  }
}

main().catch((error) => {
  console.error(error.stack ?? error.message);
  process.exit(1);
});
