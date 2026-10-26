// Pure unit tests for the sc-26687 hosted-browser driver contract.
// These run locally with Node/jsdom-less environment and start NO server and NO browser.
import { existsSync, mkdtempSync, mkdirSync, readFileSync, realpathSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { describe, expect, it } from "vitest";

import {
  MANAGED_FORM_FIELDS,
  PLAYWRIGHT_VERSION,
  REQUIRED_CASES,
  S8_MERGE_COMMIT,
  assertPlaywrightVersion,
  assertSummaryPassed,
  buildSummary,
  evaluateCaseFieldSelectors,
  evaluateEnvironment,
  evaluateSourceIdentity,
  productionWebDigest,
  loadPlaywrightCommonJs,
  resolveArtifactDir,
  resolveBrowsersPath,
  resolvePlaywrightEntry,
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

describe("managed-form field selectors", () => {
  // The case source is read from disk: `import.meta.url` is not a file URL under the jsdom
  // environment, so the path is resolved from the working directory instead.
  function hostedCaseSource() {
    const candidates = [
      resolve(process.cwd(), "verify/hosted-browser/cases.mjs"),
      resolve(process.cwd(), "apps/web/verify/hosted-browser/cases.mjs"),
    ];
    for (const candidate of candidates) {
      if (existsSync(candidate)) return readFileSync(candidate, "utf8");
    }
    throw new Error("cannot locate the hosted case source");
  }
  // The hosted run at 03cdc056c failed `model-manager-reimport` with a 30s timeout on
  // `getByLabel(/^Family$/)` — not because the panel was wrong (the captured DOM carries the
  // Family/Type/Compatible base model/MiniMax-H3 partition selects) but because Playwright matches
  // a <label>'s FULL text, and a label that wraps a <select> also carries the select's option text.
  // The anchored regex resolved nothing and the run proved nothing about the product.
  it("reads every managed-form field through its label element", () => {
    const source = hostedCaseSource();
    const result = evaluateCaseFieldSelectors(source);
    expect(result.errors).toEqual([]);
    expect(result.ok).toBe(true);
  });

  // The guard is scoped to the managed-form case, because that is where the defect is: an
  // anchored label lookup elsewhere — against an element's own aria-label — is legitimate.
  function managedCase(body) {
    return `async function caseModelManagerReimport(page, recorder) {\n${body}\n}`;
  }

  it("refuses an anchored label regex, whatever the field", () => {
    const result = evaluateCaseFieldSelectors(
      managedCase('  const value = await form.getByLabel(/^Family$/).inputValue();'),
    );
    expect(result.ok).toBe(false);
    expect(result.errors.join(" ")).toMatch(/anchored getByLabel/);
  });

  it("refuses a managed field read outside the label lookup", () => {
    const body = MANAGED_FORM_FIELDS.map((field) => `  labelledField(form, "${field}")`).join("\n")
      .replace('labelledField(form, "Name")', '  form.getByLabel(/^Name$/)');
    const result = evaluateCaseFieldSelectors(managedCase(body));
    expect(result.ok).toBe(false);
    expect(result.errors.join(" ")).toMatch(/"Name" must be read through labelledField/);
  });

  it("allows an anchored label lookup outside the managed-form case", () => {
    const source = [
      managedCase(MANAGED_FORM_FIELDS.map((field) => `  labelledField(form, "${field}")`).join("\n")),
      "async function caseSomethingElse(page) {",
      "  const value = await page.getByLabel(/^Something else$/).inputValue();",
      "}",
    ].join("\n");
    const result = evaluateCaseFieldSelectors(source);
    expect(result.errors).toEqual([]);
    expect(result.ok).toBe(true);
  });

  it("refuses a source with no managed reimport case", () => {
    const result = evaluateCaseFieldSelectors("export const CASES = [];");
    expect(result.ok).toBe(false);
    expect(result.errors.join(" ")).toMatch(/was not found/);
  });
});

function fakePlaywrightPackage(version = PLAYWRIGHT_VERSION) {
  const root = mkdtempSync(join(tmpdir(), "sc26687-playwright-"));
  const packageDir = join(root, "node_modules", "playwright");
  mkdirSync(packageDir, { recursive: true });
  writeFileSync(join(packageDir, "package.json"), JSON.stringify({ name: "playwright", version, main: "index.js" }));
  writeFileSync(join(packageDir, "index.js"), "module.exports = { chromium: {} };\n");
  return root;
}

describe("private Playwright tools resolution", () => {
  it("resolves the exact private root entry and pinned version without the apps/web graph", () => {
    const root = fakePlaywrightPackage();
    try {
      const tools = resolvePlaywrightEntry({ root });
      const realRoot = realpathSync(root);
      expect(tools.root).toBe(resolve(root));
      expect(tools.version).toBe(PLAYWRIGHT_VERSION);
      expect(tools.manifest).toBe(resolve(realRoot, "node_modules", "playwright", "package.json"));
      expect(tools.entry).toBe(resolve(realRoot, "node_modules", "playwright", "index.js"));
      expect(tools.entry.startsWith(resolve(realRoot, "node_modules") + "/")).toBe(true);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });
  it("refuses a missing package, a wrong version and a symlink escaping the private root", () => {
    const empty = mkdtempSync(join(tmpdir(), "sc26687-playwright-empty-"));
    try {
      expect(() => resolvePlaywrightEntry({ root: empty })).toThrow(/not installed in the private tools root/);
      expect(() => resolvePlaywrightEntry({ root: "" })).toThrow(/absolute private tools prefix/);
      expect(() => resolvePlaywrightEntry({ root: "apps/web" })).toThrow(/absolute private tools prefix/);
    } finally {
      rmSync(empty, { recursive: true, force: true });
    }
    const wrong = fakePlaywrightPackage("1.59.0");
    try {
      expect(() => resolvePlaywrightEntry({ root: wrong })).toThrow(/requires exactly 1\.59\.1/);
    } finally {
      rmSync(wrong, { recursive: true, force: true });
    }
    const outside = fakePlaywrightPackage();
    const linked = mkdtempSync(join(tmpdir(), "sc26687-playwright-linked-"));
    try {
      mkdirSync(join(linked, "node_modules"), { recursive: true });
      symlinkSync(join(outside, "node_modules", "playwright"), join(linked, "node_modules", "playwright"), "dir");
      expect(() => resolvePlaywrightEntry({ root: linked })).toThrow(/escapes the private tools root/);
    } finally {
      rmSync(linked, { recursive: true, force: true });
      rmSync(outside, { recursive: true, force: true });
    }
  });
  it("never falls back to a global/NODE_PATH package outside the private root", () => {
    const elsewhere = fakePlaywrightPackage();
    const unrelated = mkdtempSync(join(tmpdir(), "sc26687-playwright-unrelated-"));
    const previous = process.env.NODE_PATH;
    try {
      process.env.NODE_PATH = join(elsewhere, "node_modules");
      expect(() => resolvePlaywrightEntry({ root: unrelated })).toThrow(/not installed in the private tools root/);
    } finally {
      if (previous === undefined) delete process.env.NODE_PATH;
      else process.env.NODE_PATH = previous;
      rmSync(elsewhere, { recursive: true, force: true });
      rmSync(unrelated, { recursive: true, force: true });
    }
  });
  it("keeps browser binaries inside the private tools root", () => {
    const root = mkdtempSync(join(tmpdir(), "sc26687-playwright-browsers-"));
    try {
      expect(resolveBrowsersPath({}, { root })).toBe(resolve(root, "browsers"));
      expect(resolveBrowsersPath({ PLAYWRIGHT_BROWSERS_PATH: join(root, "custom") }, { root })).toBe(resolve(root, "custom"));
      expect(() => resolveBrowsersPath({ PLAYWRIGHT_BROWSERS_PATH: join(tmpdir(), "outside-browsers") }, { root })).toThrow(/must stay inside the private tools root/);
      expect(() => resolveBrowsersPath({ PLAYWRIGHT_BROWSERS_PATH: "relative/browsers" }, { root })).toThrow(/must be absolute/);
      expect(() => resolveBrowsersPath({}, { root: "relative" })).toThrow(/private playwright tools root/);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });
});

function fakeCommonJsPlaywright(indexSource, { implSource = null } = {}) {
  const root = mkdtempSync(join(tmpdir(), "sc26687-playwright-cjs-"));
  const packageDir = join(root, "node_modules", "playwright");
  mkdirSync(packageDir, { recursive: true });
  writeFileSync(join(packageDir, "package.json"), JSON.stringify({ name: "playwright", version: PLAYWRIGHT_VERSION, main: "index.js" }));
  writeFileSync(join(packageDir, "index.js"), indexSource);
  if (implSource !== null) writeFileSync(join(packageDir, "impl.js"), implSource);
  return root;
}

describe("Playwright CJS interop", () => {
  it("exposes chromium for plain, getter, exports and re-export CJS shapes", () => {
    const shapes = [
      "module.exports = { chromium: { launch() {} } };\n",
      "const out = {}; Object.defineProperty(out, 'chromium', { enumerable: true, get() { return { launch() {} }; } }); module.exports = out;\n",
      "exports.chromium = { launch() {} };\n",
      "module.exports = require('./impl');\n",
      "module.exports = { default: { chromium: { launch() {} } } };\n",
    ];
    for (const source of shapes) {
      const root = fakeCommonJsPlaywright(source, { implSource: "module.exports = { chromium: { launch() {} } };\n" });
      try {
        const tools = resolvePlaywrightEntry({ root });
        const loaded = loadPlaywrightCommonJs({ entry: tools.entry });
        expect(tools.version).toBe(PLAYWRIGHT_VERSION);
        expect(typeof loaded.chromium.launch).toBe("function");
        expect(loaded.chromium).toBe(loaded.chromium);
      } finally {
        rmSync(root, { recursive: true, force: true });
      }
    }
  });
  it("refuses a CJS module without a launchable chromium or a non-absolute entry", () => {
    const missing = fakeCommonJsPlaywright("module.exports = { chromium: {} };\n");
    try {
      const tools = resolvePlaywrightEntry({ root: missing });
      expect(() => loadPlaywrightCommonJs({ entry: tools.entry })).toThrow(/launchable chromium/);
    } finally {
      rmSync(missing, { recursive: true, force: true });
    }
    expect(() => loadPlaywrightCommonJs({ entry: "node_modules/playwright/index.js" })).toThrow(/absolute resolved Playwright entry/);
    expect(() => loadPlaywrightCommonJs({ entry: "" })).toThrow(/absolute resolved Playwright entry/);
  });
});
