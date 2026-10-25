// Pure unit tests for the sc-26687 hosted-browser driver contract.
// These run locally with Node/jsdom-less environment and start NO server and NO browser.
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

import {
  PLAYWRIGHT_VERSION,
  REQUIRED_CASES,
  S8_MERGE_COMMIT,
  assertPlaywrightVersion,
  assertSummaryPassed,
  buildSummary,
  evaluateEnvironment,
  evaluateSourceIdentity,
  productionWebDigest,
  resolveArtifactDir,
} from "./guard.mjs";
import { CASES } from "./cases.mjs";
import { Readable } from 'node:stream';
import {verifyMiddleware,MANAGED_REPAIR_REQUEST,MINIMAX_REF} from '../../verify-fixtures.mjs';

async function fixtureCall(method,url,body){
  const request=Readable.from(body===undefined?[]:[JSON.stringify(body)]);request.method=method;request.url=url;
  const response={statusCode:200,setHeader(){},end(value){this.value=JSON.parse(value);}};
  await verifyMiddleware()(request,response,()=>{throw new Error('API fixture must never forward');});return {status:response.statusCode,body:response.value};
}
describe('actual managed repair fixture contract, no server or real legal authority',()=>{
  it('refuses wrong-selection repair and accepts only exact stable-ID repair DTO',async()=>{
    await fixtureCall('POST','/api/__verify/reset');
    const invalid=await fixtureCall('POST','/api/v1/models/my_h3/repair',{...MANAGED_REPAIR_REQUEST,modelId:'renamed_new_model'});expect(invalid.status).toBe(422);
    const repaired=await fixtureCall('POST','/api/v1/models/my_h3/repair',MANAGED_REPAIR_REQUEST);expect(repaired.status).toBe(201);expect(repaired.body.payload.modelId).toBe('my_h3');
    const catalog=await fixtureCall('GET','/api/v1/models');expect(catalog.body.filter(model=>model.id==='my_h3')).toHaveLength(1);expect(catalog.body.find(model=>model.id==='my_h3').name).toBe('Edited name');
  });
  it('serves explicitly synthetic consent terms and does not authorize real upstream licenses',()=>{
    expect(MINIMAX_REF.licenseUrl).toBe('https://example.test/synthetic-license');expect(MINIMAX_REF.licenseNotice).toContain('not acceptance of any real model license');expect(MINIMAX_REF.requiresLicenseAcknowledgment).toBe(true);
  });
});

const HOSTED_ENV = {
  CI: "true",
  GITHUB_ACTIONS: "true",
  RUNNER_ENVIRONMENT: "github-hosted",
  SC26687_HOSTED_BROWSER_PROOF: "1",
  GITHUB_WORKSPACE: "/home/runner/work/SceneWorks/SceneWorks",
  RUNNER_TEMP: "/home/runner/work/_temp",
};

function env(overrides = {}) {
  return { ...HOSTED_ENV, ...overrides };
}

function workspacePath() {
  return HOSTED_ENV.GITHUB_WORKSPACE;
}

describe("hosted environment guard", () => {
  it("accepts an ephemeral GitHub-hosted workspace with the explicit marker", () => {
    const result = evaluateEnvironment(env(), { repoRoot: workspacePath() });
    expect(result.ok).toBe(true);
    expect(result.errors).toEqual([]);
  });

  it("refuses local execution and any missing hosted signal", () => {
    for (const missing of [
      { CI: "false" },
      { CI: undefined },
      { GITHUB_ACTIONS: undefined },
      { RUNNER_ENVIRONMENT: undefined },
      { RUNNER_ENVIRONMENT: "self-hosted" },
      { SC26687_HOSTED_BROWSER_PROOF: undefined },
      { SC26687_HOSTED_BROWSER_PROOF: "0" },
      { GITHUB_WORKSPACE: undefined },
    ]) {
      const result = evaluateEnvironment(env(missing), { repoRoot: workspacePath() });
      expect(result.ok, JSON.stringify(missing)).toBe(false);
      expect(result.errors.length).toBeGreaterThan(0);
    }
  });

  it("rejects a local macOS workspace and any workspace outside the declared root", () => {
    const localRepo = "/Users/someone/Repos/SceneWorks";
    const local = evaluateEnvironment(env({ GITHUB_WORKSPACE: localRepo }), { repoRoot: localRepo });
    expect(local.ok).toBe(false);
    expect(local.errors.join(" ")).toContain("refusing to run locally");

    const elsewhere = evaluateEnvironment(env(), { repoRoot: "/opt/other-checkout" });
    expect(elsewhere.ok).toBe(false);
    expect(elsewhere.errors.join(" ")).toContain("exactly GITHUB_WORKSPACE");
  });
});

describe("artifact directory boundary", () => {
  it("keeps artifacts inside the verified workspace or RUNNER_TEMP", () => {
    const inside = resolveArtifactDir(env(), { repoRoot: workspacePath() });
    expect(inside).toBe(resolve(workspacePath(), "browser-artifacts/sc26687"));

    const absoluteInside = resolveArtifactDir(
      env({ SC26687_ARTIFACT_DIR: "/home/runner/work/_temp/artifacts" }),
      { repoRoot: workspacePath() },
    );
    expect(absoluteInside).toBe("/home/runner/work/_temp/artifacts");

    expect(() =>
      resolveArtifactDir(env({ SC26687_ARTIFACT_DIR: "/tmp/elsewhere" }), { repoRoot: workspacePath() }),
    ).toThrow(/must be inside/);
  });
});

describe("summary contract", () => {
  const cases = REQUIRED_CASES.map((id) => ({ id, status: "pass", executed: true, skipped: false }));

  it("is green only when every required case actually executed and passed", () => {
    const summary = buildSummary({
      cases,
      startedAt: "2026-10-25T00:00:00.000Z",
      durationMs: 1234,
      source: {},
      requests: {},
      artifacts: {},
    });
    expect(summary.ok).toBe(true);
    expect(summary.totals.executed).toBe(REQUIRED_CASES.length);
    expect(() => assertSummaryPassed(summary)).not.toThrow();
  });

  it("fails on a missing, skipped, failed or zero-case run", () => {
    const missing = buildSummary({ cases: cases.slice(1), startedAt: "", durationMs: 0, source: {}, requests: {}, artifacts: {} });
    expect(missing.ok).toBe(false);
    expect(missing.missingCases).toEqual([REQUIRED_CASES[0]]);

    const skipped = buildSummary({
      cases: cases.map((entry, index) => (index === 0 ? { ...entry, executed: false, skipped: true } : entry)),
      startedAt: "",
      durationMs: 0,
      source: {},
      requests: {},
      artifacts: {},
    });
    expect(skipped.ok).toBe(false);

    const failed = buildSummary({
      cases: cases.map((entry, index) => (index === 0 ? { ...entry, status: "fail", error: "boom" } : entry)),
      startedAt: "",
      durationMs: 0,
      source: {},
      requests: {},
      artifacts: {},
    });
    expect(failed.ok).toBe(false);
    expect(() => assertSummaryPassed(failed)).toThrow(/boom/);

    const zero = buildSummary({ cases: [], startedAt: "", durationMs: 0, source: {}, requests: {}, artifacts: {} });
    expect(zero.ok).toBe(false);
    expect(zero.totals.executed).toBe(0);
  });
});

function identity({ head, gitSha, expectedHead = null, isAncestor }) {
  return evaluateSourceIdentity({ head, gitSha, expectedHead, s8Ancestor: isAncestor });
}

describe("source identity receipt", () => {
  const SHA = "068eb22b1e7e04cbcc23441fd29963f10ef2a496";
  const OTHER = "1".repeat(40);

  it("accepts the workflow HEAD when GH_SHA/expected match and b407 is an ancestor", () => {
    const ok = identity({ head: SHA, gitSha: SHA, expectedHead: SHA, isAncestor: true });
    expect(ok.ok).toBe(true);
    expect(ok.errors).toEqual([]);
    // GITHUB_SHA is sufficient when the workflow does not supply an expected source.
    expect(identity({ head: SHA, gitSha: SHA, expectedHead: null, isAncestor: true }).ok).toBe(true);
  });

  it("refuses a wrong source, a missing ancestor, or a malformed receipt", () => {
    expect(identity({ head: SHA, gitSha: OTHER, isAncestor: true }).ok).toBe(false);
    expect(identity({ head: SHA, gitSha: SHA, expectedHead: OTHER, isAncestor: true }).ok).toBe(false);
    expect(identity({ head: SHA, gitSha: SHA, isAncestor: false }).ok).toBe(false);
    expect(identity({ head: "HEAD", gitSha: SHA, isAncestor: true }).ok).toBe(false);
  });

  it("keeps the S8 merge commit as the ancestor receipt", () => {
    expect(S8_MERGE_COMMIT).toBe("78d5ba648e3232ab8309ad1da2f824392fd334a5");
    expect(S8_MERGE_COMMIT).not.toBe(SHA);
  });
});

describe("production frontend closure digest", () => {
  const lsTree = (entries) =>
    entries.map(([mode, hash, path]) => `${mode} blob ${hash}\t${path}`).join("\0") + "\0";

  const production = [
    ["100644", "a".repeat(40), "apps/web/src/App.jsx"],
    ["100644", "b".repeat(40), "apps/web/src/screens/ImageStudio.jsx"],
  ];
  const overlay = [
    ["100644", "c".repeat(40), "apps/web/verify-fixtures.mjs"],
    ["100644", "d".repeat(40), "apps/web/vite.verify.config.mjs"],
    ["100644", "e".repeat(40), "apps/web/verify/hosted-browser/guard.mjs"],
  ];

  it("excludes exactly the verification overlay from the production closure", () => {
    const prodOnly = productionWebDigest(lsTree(production));
    const withOverlay = productionWebDigest(lsTree([...production, ...overlay]));
    expect(withOverlay).toBe(prodOnly);
    expect(prodOnly).toMatch(/^[0-9a-f]{64}$/);
  });

  it("changes when a production frontend file changes, and is order-independent", () => {
    const base = productionWebDigest(lsTree(production));
    const changed = productionWebDigest(lsTree([production[0], ["100644", "f".repeat(40), production[1][2]]]));
    expect(changed).not.toBe(base);
    expect(productionWebDigest(lsTree([...production].reverse()))).toBe(base);
  });
});

describe("Playwright dependency contract", () => {
  it("requires the exact pinned version", () => {
    expect(PLAYWRIGHT_VERSION).toBe("1.59.1");
    expect(assertPlaywrightVersion("1.59.1")).toBe("1.59.1");
    expect(() => assertPlaywrightVersion("1.59.0")).toThrow(/requires exactly/);
    expect(() => assertPlaywrightVersion(undefined)).toThrow(/requires exactly/);
  });
});

describe("case manifest", () => {
  it("declares exactly the five required browser cases, each runnable and unskipped", () => {
    expect(CASES.map((entry) => entry.id)).toEqual(REQUIRED_CASES);
    expect(CASES).toHaveLength(5);
    for (const entry of CASES) {
      expect(entry.skipped).toBeUndefined();
      expect(typeof entry.run).toBe("function");
    }
  });
});
