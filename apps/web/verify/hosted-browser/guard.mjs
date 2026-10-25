// Pure guards and result contracts for the sc-26687 hosted-browser proof.
//
// The driver may start the isolated fixture Vite server ONLY on an ephemeral
// GitHub-hosted CI runner. This module holds that decision separately from the
// browser work so it can be unit-tested locally without starting any server.
import { createHash } from "node:crypto";
import { isAbsolute, resolve, sep } from "node:path";

// Stable receipt: the S8 merge that must be an ancestor of whatever the job checks
// out. This is history, NOT a mutable expected tree — later consumer fixes and this
// verification overlay are allowed descendants.
export const S8_MERGE_COMMIT = "78d5ba648e3232ab8309ad1da2f824392fd334a5";

// Exact Playwright version installed by the hosted job. Mutable ranges are refused.
export const PLAYWRIGHT_VERSION = "1.59.1";

// The verification overlay this branch adds under apps/web. It must be excluded from
// the production-frontend closure digest: pinning the full apps/web tree to the
// pre-overlay value would make every run refuse itself.
export const VERIFICATION_OVERLAY_FILES = [
  "apps/web/verify-fixtures.mjs",
  "apps/web/vite.verify.config.mjs",
];
export const VERIFICATION_OVERLAY_PREFIXES = ["apps/web/verify/"];

export function isVerificationOverlayPath(path) {
  return (
    VERIFICATION_OVERLAY_FILES.includes(path) ||
    VERIFICATION_OVERLAY_PREFIXES.some((prefix) => path.startsWith(prefix))
  );
}

/**
 * Digest of the TRACKED production frontend closure from `git ls-tree -r HEAD -- apps/web`,
 * excluding this verification overlay. Any production file change changes the digest; the
 * overlay itself never does.
 */
export function productionWebDigest(lsTreeOutput) {
  const lines = String(lsTreeOutput)
    .split(/\0|\n/)
    .filter(Boolean)
    .map((line) => {
      const tab = line.indexOf("\t");
      return { raw: line, path: tab === -1 ? "" : line.slice(tab + 1) };
    })
    .filter((entry) => entry.path && !isVerificationOverlayPath(entry.path))
    .map((entry) => entry.raw)
    .sort();
  return createHash("sha256").update(lines.join("\n")).digest("hex");
}

/**
 * Source identity contract for one run.
 *
 * - the checkout HEAD must equal the workflow's GITHUB_SHA;
 * - when the workflow supplies SC26687_EXPECTED_HEAD (the publication source), HEAD must
 *   equal it too, so a wrong ref fails rather than proving another tree;
 * - the S8 merge must be an ancestor of HEAD (source actually contains S8).
 *
 * No expected apps/web tree is pinned here: the production closure digest is recorded,
 * not frozen.
 */
export function evaluateSourceIdentity({ head, gitSha, expectedHead, s8Ancestor }) {
  const errors = [];
  const sha = /^[0-9a-f]{40}$/;
  if (!sha.test(head ?? "")) errors.push(`HEAD receipt is not a full commit sha: ${head}`);
  if (!sha.test(gitSha ?? "")) errors.push(`GITHUB_SHA is missing or not a full commit sha: ${gitSha}`);
  if (head !== gitSha) errors.push(`checkout HEAD ${head} does not match GITHUB_SHA ${gitSha}`);
  if (expectedHead) {
    if (!sha.test(expectedHead)) errors.push(`SC26687_EXPECTED_HEAD is not a full commit sha: ${expectedHead}`);
    if (head !== expectedHead) {
      errors.push(`checkout HEAD ${head} does not match SC26687_EXPECTED_HEAD ${expectedHead}`);
    }
  }
  if (s8Ancestor !== true) errors.push(`S8 merge ${S8_MERGE_COMMIT} is not an ancestor of HEAD`);
  return { ok: errors.length === 0, errors };
}

export function assertPlaywrightVersion(actual) {
  if (actual !== PLAYWRIGHT_VERSION) {
    throw new Error(
      `playwright ${actual ?? "unknown"} is installed, but this run requires exactly ${PLAYWRIGHT_VERSION}`,
    );
  }
  return actual;
}

export const REQUIRED_CASES = [
  "model-manager-repair",
  "model-manager-reimport",
  "video-last-only",
  "image-empty-refs",
  "video-replay-expected-ref",
];

// macOS developer machines and scratch temp trees. A hosted Linux runner uses
// /home/runner/... which is legitimate and is validated by GITHUB_WORKSPACE equality.
const LOCAL_ROOTS = ["/Users", "/private/var/folders", "/var/folders"];

function isInside(parent, child) {
  const root = resolve(parent);
  const target = resolve(child);
  return target === root || target.startsWith(root + sep);
}

function underAny(path, roots) {
  return roots.some((root) => isInside(root, path));
}

/**
 * Decide whether this process may start the fixture server and a browser.
 * Every rule is required; a failure is reported, never silently downgraded.
 *
 * @param {Record<string, string|undefined>} env
 * @param {{repoRoot: string, realpath?: (p: string) => string}} options
 * @returns {{ok: boolean, errors: string[], workspace: string, repoRoot: string}}
 */
export function evaluateEnvironment(env, { repoRoot, realpath = (p) => p }) {
  const errors = [];
  const require = (condition, message) => {
    if (!condition) errors.push(message);
  };

  require(env.CI === "true", "CI must be 'true' (ephemeral hosted runner only)");
  require(env.GITHUB_ACTIONS === "true", "GITHUB_ACTIONS must be 'true' (GitHub Actions only)");
  require(
    env.RUNNER_ENVIRONMENT === "github-hosted",
    "RUNNER_ENVIRONMENT must be 'github-hosted' (self-hosted/local runners are refused)",
  );
  require(
    env.SC26687_HOSTED_BROWSER_PROOF === "1",
    "SC26687_HOSTED_BROWSER_PROOF must be '1' (explicit verification marker)",
  );

  const workspace = env.GITHUB_WORKSPACE || "";
  if (!workspace) {
    errors.push("GITHUB_ACTIONS requires GITHUB_WORKSPACE (bounded private root)");
  } else {
    let workspaceReal = workspace;
    let repoReal = repoRoot;
    try {
      workspaceReal = realpath(workspace);
      repoReal = realpath(repoRoot);
    } catch (error) {
      errors.push(`cannot resolve verification roots: ${error.message}`);
    }
    require(isAbsolute(workspace), "GITHUB_WORKSPACE must be an absolute path");
    require(
      repoReal === workspaceReal,
      `repo root (${repoRoot}) must be exactly GITHUB_WORKSPACE (${workspace})`,
    );
    require(
      !underAny(repoReal, LOCAL_ROOTS),
      "repository root is a local developer/scratch path; refusing to run locally",
    );
  }

  return { ok: errors.length === 0, errors, workspace, repoRoot };
}

/**
 * Artifact directory must stay inside the CI workspace or RUNNER_TEMP.
 * @returns {string} absolute artifact directory
 */
export function resolveArtifactDir(env, { repoRoot, fallback = "browser-artifacts/sc26687" }) {
  const chosen = env.SC26687_ARTIFACT_DIR || fallback;
  const absolute = isAbsolute(chosen) ? chosen : resolve(repoRoot, chosen);
  const bounded = [repoRoot, env.RUNNER_TEMP, env.GITHUB_WORKSPACE].filter(Boolean);
  if (!bounded.some((root) => isInside(root, absolute))) {
    throw new Error(
      `artifact directory ${absolute} must be inside the verified workspace or RUNNER_TEMP`,
    );
  }
  return absolute;
}

/**
 * Build the machine-readable summary. Statuses are only "pass" or "fail"; a case
 * that did not run is "fail" so a zero-case or skipped run can never look green.
 */
export function buildSummary({ cases, startedAt, durationMs, source, requests, artifacts }) {
  const required = new Set(REQUIRED_CASES);
  const executed = new Set(cases.map((entry) => entry.id));
  const missing = REQUIRED_CASES.filter((id) => !executed.has(id));
  const failed = cases.filter((entry) => entry.status !== "pass");
  const extra = [...executed].filter((id) => !required.has(id));
  return {
    schemaVersion: 1,
    story: "sc-26687",
    startedAt,
    durationMs,
    source,
    cases,
    requests,
    artifacts,
    totals: {
      required: REQUIRED_CASES.length,
      executed: cases.filter((entry) => entry.executed === true).length,
      passed: cases.filter((entry) => entry.status === "pass").length,
      failed: failed.length,
      missing: missing.length,
      unexpected: extra.length,
      skipped: cases.filter((entry) => entry.skipped === true).length,
    },
    ok:
      missing.length === 0 &&
      failed.length === 0 &&
      extra.length === 0 &&
      cases.every((entry) => entry.executed === true && entry.skipped !== true),
    missingCases: missing,
    unexpectedCases: extra,
  };
}

export function assertSummaryPassed(summary) {
  if (summary.ok) return;
  const details = [];
  if (summary.missingCases.length) details.push(`missing: ${summary.missingCases.join(", ")}`);
  if (summary.unexpectedCases.length) details.push(`unexpected: ${summary.unexpectedCases.join(", ")}`);
  for (const entry of summary.cases) {
    if (entry.status !== "pass") details.push(`${entry.id}: ${entry.error ?? "did not pass"}`);
  }
  throw new Error(`sc-26687 hosted browser proof failed (${details.join("; ")})`);
}
