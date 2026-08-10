import assert from "node:assert/strict";
import { execFileSync, spawn } from "node:child_process";
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  renameSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { hostname, tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import {
  FEATURE_RE,
  GitHubClient,
  GitRunner,
  REPOSITORIES,
  REQUIRED_CONTEXTS,
  ShortcutClient,
  adoptPullRequest,
  auditFeaturePolicies,
  auditState,
  bootstrap,
  classifyPullRequest,
  classifyAdoptedPullRequest,
  createPlan,
  enforceCiPolicy,
  evaluateEventTopology,
  extractInferencePin,
  featureBranch,
  featurePolicyPayloads,
  inspectShortcutEpic,
  loadState,
  recordEvidence,
  refreshStoryInventory,
  recover,
  resolveRemoteFeatureBranch,
  storyBranch,
  stateLockPath,
  verifyEffectiveFeaturePolicies,
  withStateLock,
  writeJsonAtomic,
} from "./lib/feature-epic.mjs";

const EPOCH = "2026-08-10T12:00:00Z";
const clock = () => new Date(EPOCH);
const GITHUB_ACTIONS_APP_ID = 15368;

function trustedCheck(name, id, overrides = {}) {
  return {
    id,
    name,
    app: { id: GITHUB_ACTIONS_APP_ID },
    status: "completed",
    conclusion: "success",
    started_at: EPOCH,
    completed_at: EPOCH,
    details_url: `https://github.com/checks/${id}`,
    ...overrides,
  };
}

function successfulChecks(repo, firstId = 1) {
  return REQUIRED_CONTEXTS[repo].map((name, index) => trustedCheck(name, firstId + index));
}

function run(command, args, options = {}) {
  return execFileSync(command, args, {
    cwd: options.cwd,
    encoding: "utf8",
    env: {
      ...process.env,
      GIT_AUTHOR_NAME: "Feature Epic Fixture",
      GIT_AUTHOR_EMAIL: "fixture@example.test",
      GIT_COMMITTER_NAME: "Feature Epic Fixture",
      GIT_COMMITTER_EMAIL: "fixture@example.test",
      GIT_AUTHOR_DATE: EPOCH,
      GIT_COMMITTER_DATE: EPOCH,
    },
    stdio: options.stdio ?? ["ignore", "pipe", "pipe"],
  }).trim();
}

async function waitForPath(path, timeoutMs = 5000) {
  const deadline = Date.now() + timeoutMs;
  while (!existsSync(path) && Date.now() < deadline) {
    await new Promise((resolve) => setTimeout(resolve, 20));
  }
  assert.equal(existsSync(path), true, `timed out waiting for ${path}`);
}

function waitForChild(child) {
  return new Promise((resolve, reject) => {
    let stderr = "";
    child.stderr?.on("data", (chunk) => { stderr += chunk; });
    child.once("error", reject);
    child.once("exit", (code, signal) => {
      if (code === 0) resolve();
      else reject(new Error(`child exited ${code ?? signal}: ${stderr}`));
    });
  });
}

function commitFile(repo, path, text, message) {
  const target = join(repo, path);
  mkdirSync(join(target, ".."), { recursive: true });
  writeFileSync(target, text);
  run("git", ["add", path], { cwd: repo });
  run("git", ["commit", "-m", message], { cwd: repo });
  return run("git", ["rev-parse", "HEAD"], { cwd: repo });
}

function initSource(path) {
  mkdirSync(path, { recursive: true });
  run("git", ["init", "-b", "main"], { cwd: path });
  run("git", ["config", "user.name", "Feature Epic Fixture"], { cwd: path });
  run("git", ["config", "user.email", "fixture@example.test"], { cwd: path });
}

function cloneBare(source, target) {
  run("git", ["clone", "--bare", source, target]);
  run("git", ["symbolic-ref", "HEAD", "refs/heads/main"], { cwd: target });
}

function manifest(pin) {
  return `[package]\nname = "fixture"\nversion = "0.0.0"\n\n[dependencies]\nruntime-macos = { git = "https://github.com/SceneWorks/inference", rev = "${pin}" }\n`;
}

class FakeShortcut {
  constructor(epic = "sc-18304", stories = ["sc-18418", "sc-18419"]) {
    this.epicId = Number(epic.slice(3));
    this.storyValues = stories.map((story) => ({
      id: Number(story.slice(3)),
      epic_id: this.epicId,
      name: `Fixture ${story}`,
      app_url: `https://app.shortcut.test/story/${story.slice(3)}`,
      updated_at: EPOCH,
      workflow_state_id: 30,
      completed: true,
      blocked: false,
      blocker: false,
      archived: false,
    }));
    this.epicValue = {
      id: this.epicId,
      name: `Fixture ${epic}`,
      app_url: `https://app.shortcut.test/epic/${this.epicId}`,
      state: "in progress",
      archived: false,
      completed: false,
      updated_at: EPOCH,
      stats: { num_stories_total: this.storyValues.length },
    };
    this.workflowValues = [{
      id: 10,
      states: [
        { id: 20, name: "In Progress", type: "started" },
        { id: 30, name: "Done", type: "done" },
      ],
    }];
  }

  epic(id) {
    assert.equal(id, this.epicId);
    return structuredClone(this.epicValue);
  }

  epicStories(id) {
    assert.equal(id, this.epicId);
    return structuredClone(this.storyValues);
  }

  workflows() {
    return structuredClone(this.workflowValues);
  }
}

class BareGitHub {
  constructor(origins) {
    this.origins = origins;
    this.defaults = { sceneworks: "main", inference: "main" };
    this.failCreateFor = null;
    this.failAfterCreateFor = null;
    this.failRulesetFor = null;
    this.failAfterRulesetFor = null;
    this.failUpdateFor = null;
    this.failAfterUpdateFor = null;
    this.ruleSets = { sceneworks: [], inference: [] };
    this.nextRulesetId = 3000;
    this.mutations = [];
    this.prs = new Map();
    this.checks = new Map();
    this.checkQueries = [];
    this.timelines = new Map();
    this.runs = new Map();
    this.deployments = new Map();
    this.statuses = new Map();
    this.comparisons = new Map();
  }

  key(slug) {
    const entry = Object.entries(REPOSITORIES).find(([, repo]) => repo.slug === slug);
    assert.ok(entry, `unexpected slug ${slug}`);
    return entry[0];
  }

  origin(slug) {
    return this.origins[this.key(slug)];
  }

  defaultBranch(slug) {
    return this.defaults[this.key(slug)];
  }

  mainSha(slug) {
    return run("git", ["rev-parse", "refs/heads/main"], { cwd: this.origin(slug) });
  }

  featureSha(slug, branch) {
    try {
      return run("git", ["rev-parse", `refs/heads/${branch}`], { cwd: this.origin(slug) });
    } catch {
      return null;
    }
  }

  createFeatureRef(slug, branch, sha) {
    const key = this.key(slug);
    this.mutations.push({ type: "create-ref", repo: key, branch, sha });
    if (this.failCreateFor === key) throw new Error(`injected create-ref failure for ${key}`);
    const exactRef = `refs/heads/${branch}`;
    const activeQueue = this.ruleSets[key].some(
      (ruleset) =>
        ruleset.enforcement === "active" &&
        ruleset.conditions?.ref_name?.include?.length === 1 &&
        ruleset.conditions.ref_name.include[0] === exactRef &&
        ruleset.rules?.some(({ type }) => type === "merge_queue"),
    );
    if (activeQueue) {
      const error = new Error("HTTP 422: Changes must be made through the merge queue");
      error.status = 422;
      throw error;
    }
    assert.equal(this.featureSha(slug, branch), null, "fixture create-ref is create-only");
    run("git", ["update-ref", `refs/heads/${branch}`, sha], { cwd: this.origin(slug) });
    if (this.failAfterCreateFor === key) throw new Error(`injected ambiguous create-ref failure for ${key}`);
    return { ref: `refs/heads/${branch}`, object: { sha } };
  }

  rulesets(slug) {
    return structuredClone(this.ruleSets[this.key(slug)]);
  }

  addRuleset(repo, payload, id = this.nextRulesetId++) {
    const value = { id, ...structuredClone(payload) };
    this.ruleSets[repo].push(value);
    return value;
  }

  createRuleset(slug, payload) {
    const key = this.key(slug);
    this.mutations.push({ type: "create-ruleset", repo: key, name: payload.name });
    if (this.failRulesetFor === key) throw new Error(`injected ruleset failure for ${key}`);
    const created = this.addRuleset(key, payload);
    if (this.failAfterRulesetFor === key) throw new Error(`injected ambiguous ruleset failure for ${key}`);
    return structuredClone(created);
  }

  updateRuleset(slug, id, payload) {
    const key = this.key(slug);
    this.mutations.push({
      type: "update-ruleset",
      repo: key,
      id,
      name: payload.name,
      enforcement: payload.enforcement,
    });
    if (this.failUpdateFor === key) throw new Error(`injected ruleset update failure for ${key}`);
    const index = this.ruleSets[key].findIndex((ruleset) => ruleset.id === id);
    assert.notEqual(index, -1, "fixture updates only a known ruleset id");
    this.ruleSets[key][index] = { id, ...structuredClone(payload) };
    if (this.failAfterUpdateFor === key) {
      throw new Error(`injected ambiguous ruleset update failure for ${key}`);
    }
    return structuredClone(this.ruleSets[key][index]);
  }

  branchRules(slug, branch) {
    const key = this.key(slug);
    const exactRef = `refs/heads/${branch}`;
    return this.ruleSets[key]
      .filter((ruleset) => {
        if (ruleset.enforcement !== "active") return false;
        const include = ruleset.conditions?.ref_name?.include ?? [];
        return include.includes(exactRef) || (branch.startsWith("feature/") && include.includes("refs/heads/feature/*"));
      })
      .flatMap((ruleset) => ruleset.rules.map((rule) => ({ ...structuredClone(rule), ruleset_id: ruleset.id })));
  }

  cargoManifests(slug, sha) {
    const output = run("git", ["ls-tree", "-r", "--name-only", sha], { cwd: this.origin(slug) });
    return output
      .split("\n")
      .filter((path) => /(^|\/)Cargo\.toml$/.test(path))
      .sort()
      .map((path) => ({ path, text: run("git", ["show", `${sha}:${path}`], { cwd: this.origin(slug) }) + "\n" }));
  }

  compare(slug, base, head) {
    const override = this.comparisons.get(`${this.key(slug)}:${base}:${head}`);
    if (override) return structuredClone(override);
    try {
      run("git", ["merge-base", "--is-ancestor", base, head], { cwd: this.origin(slug) });
      return { status: base === head ? "identical" : "ahead" };
    } catch {
      return { status: "diverged" };
    }
  }

  pullRequest(slug, number) {
    const value = this.prs.get(`${this.key(slug)}:${number}`);
    if (!value) return value;
    return {
      number,
      state: value.state ?? (value.merged_at ? "closed" : "open"),
      ...structuredClone(value),
    };
  }

  pullRequests(slug, { state = "all", base, head } = {}) {
    const key = this.key(slug);
    const headBranch = head?.slice(head.indexOf(":") + 1);
    return [...this.prs.entries()]
      .filter(([mapKey]) => mapKey.startsWith(`${key}:`))
      .map(([mapKey]) => this.pullRequest(slug, Number(mapKey.slice(mapKey.indexOf(":") + 1))))
      .filter((pullRequest) => state === "all" || pullRequest.state === state)
      .filter((pullRequest) => !base || pullRequest.base?.ref === base)
      .filter((pullRequest) => !head || pullRequest.head?.ref === headBranch);
  }

  checkRuns(slug, sha) {
    const key = this.key(slug);
    this.checkQueries.push({ repo: key, sha });
    return this.checks.get(`${key}:${sha}`) ?? [];
  }

  timeline(slug, number) {
    return this.timelines.get(`${this.key(slug)}:${number}`) ?? [];
  }

  workflowRun(slug, id) {
    return this.runs.get(`${this.key(slug)}:${id}`);
  }

  deployment(slug, id) {
    return this.deployments.get(`${this.key(slug)}:${id}`);
  }

  deploymentStatuses(slug, id) {
    return this.statuses.get(`${this.key(slug)}:${id}`) ?? [];
  }
}

class FakeGhProcess {
  constructor(responder) {
    this.responder = responder;
    this.calls = [];
  }

  exec(command, args) {
    assert.equal(command, "gh");
    this.calls.push([...args]);
    const response = this.responder(args);
    return typeof response === "string" ? response : JSON.stringify(response);
  }
}

function paginatedRulesetClient(pagesByRepo) {
  const listResponses = new Map();
  const details = new Map();
  for (const [repo, pages] of Object.entries(pagesByRepo)) {
    const slug = REPOSITORIES[repo].slug;
    listResponses.set(
      `repos/${slug}/rulesets?includes_parents=false&per_page=100`,
      pages.map((page) => page.map(({ id }) => ({ id }))),
    );
    for (const ruleset of pages.flat()) details.set(`repos/${slug}/rulesets/${ruleset.id}`, ruleset);
  }
  const process = new FakeGhProcess((args) => {
    const path = args[1];
    if (listResponses.has(path)) return listResponses.get(path);
    if (details.has(path)) return details.get(path);
    assert.fail(`unexpected gh api path ${path}`);
  });
  return { client: new GitHubClient(process), process };
}

class FixtureGit extends GitRunner {
  constructor(origins) {
    super();
    this.origins = origins;
    this.failCloneFor = null;
    this.failedOnce = false;
  }

  cloneFeature(repo, branch, target, staging) {
    const key = Object.entries(REPOSITORIES).find(([, fixed]) => fixed.slug === repo.slug)[0];
    if (this.failCloneFor === key && !this.failedOnce) {
      this.failedOnce = true;
      mkdirSync(staging, { recursive: true });
      writeFileSync(join(staging, "partial"), "injected failure\n");
      throw new Error(`injected clone failure for ${key}`);
    }
    run("git", ["clone", "--branch", branch, "--single-branch", this.origins[key], staging]);
    run("git", ["remote", "set-url", "origin", repo.cloneUrl], { cwd: staging });
    renameSync(staging, target);
  }
}

function fixture(t, options = {}) {
  const root = mkdtempSync(join(tmpdir(), "feature-epic-test-"));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const inferenceSource = join(root, "inference-source");
  initSource(inferenceSource);
  const inferenceMain = commitFile(inferenceSource, "Cargo.toml", "[workspace]\n", "inference main");
  const inferenceOrigin = join(root, "inference.git");
  cloneBare(inferenceSource, inferenceOrigin);

  const sceneWorksSource = join(root, "sceneworks-source");
  initSource(sceneWorksSource);
  const pin = options.pin ?? inferenceMain;
  const sceneWorksMain = commitFile(sceneWorksSource, "Cargo.toml", manifest(pin), "SceneWorks main");
  if (options.secondPin) {
    commitFile(
      sceneWorksSource,
      "crates/other/Cargo.toml",
      manifest(options.secondPin),
      "add second inference pin",
    );
  }
  if (options.missingPin) {
    writeFileSync(join(sceneWorksSource, "Cargo.toml"), "[workspace]\n");
    run("git", ["add", "Cargo.toml"], { cwd: sceneWorksSource });
    run("git", ["commit", "-m", "remove pin"], { cwd: sceneWorksSource });
  }
  const sceneWorksOrigin = join(root, "sceneworks.git");
  cloneBare(sceneWorksSource, sceneWorksOrigin);
  const origins = { sceneworks: sceneWorksOrigin, inference: inferenceOrigin };
  const github = new BareGitHub(origins);
  const resolvedSceneWorksMain = github.mainSha(REPOSITORIES.sceneworks.slug);
  github.checks.set("sceneworks:" + resolvedSceneWorksMain, successfulChecks("sceneworks", 100));
  github.checks.set("inference:" + inferenceMain, successfulChecks("inference", 200));
  for (const [index, key] of Object.keys(REPOSITORIES).entries()) {
    const policies = featurePolicyPayloads(key, "feature/sc-18304-pipeline-flexibility-mlx-perf", "sc-18304");
    github.addRuleset(key, policies.wildcardBase, 1000 + index * 100 + 1);
    github.addRuleset(key, policies.deletionGuard, 1000 + index * 100 + 2);
  }
  const git = new FixtureGit(origins);
  const shortcut = new FakeShortcut(
    options.epic ?? "sc-18304",
    options.stories ?? ["sc-18418", "sc-18419"],
  );
  const workspaceParent = join(root, "workspace");
  mkdirSync(workspaceParent);
  const statePath = join(root, "epic-state.json");
  const reportPath = join(root, "epic-report.json");
  return {
    root,
    origins,
    github,
    git,
    shortcut,
    workspaceParent,
    statePath,
    reportPath,
    inferenceMain,
    sceneWorksMain: resolvedSceneWorksMain,
  };
}

function planFixture(f, options = {}) {
  const epic = options.epic ?? "sc-18304";
  const stories = options.stories ?? ["sc-18418", "sc-18419"];
  f.shortcut = new FakeShortcut(epic, stories);
  return createPlan(
    {
      epic,
      slug: options.slug ?? "pipeline-flexibility-mlx-perf",
      stories,
      workspaceParent: f.workspaceParent,
      statePath: f.statePath,
      deploymentRequired: options.deploymentRequired ?? false,
      adoptExisting: options.adoptExisting ?? false,
    },
    { github: f.github, shortcut: f.shortcut, clock },
  );
}

function installActivePolicyPair(f, epic = "sc-18304", slug = "pipeline-flexibility-mlx-perf") {
  const branch = featureBranch(epic, slug);
  for (const key of Object.keys(REPOSITORIES)) {
    const main = f.github.mainSha(REPOSITORIES[key].slug);
    run("git", ["update-ref", `refs/heads/${branch}`, main], { cwd: f.origins[key] });
    f.github.addRuleset(key, featurePolicyPayloads(key, branch, epic).exactQueue);
  }
  return branch;
}

function adoptionPlanFixture(f, options = {}) {
  installActivePolicyPair(
    f,
    options.epic ?? "sc-18304",
    options.slug ?? "pipeline-flexibility-mlx-perf",
  );
  return planFixture(f, { ...options, adoptExisting: true });
}

function mergedAdoptionPr(
  f,
  state,
  {
    repo = "sceneworks",
    number,
    head,
    base,
    headSha = state.repositories[repo].plannedMainSha,
    mergeCommit = state.repositories[repo].plannedMainSha,
    queuedAt = "2026-08-10T12:10:00Z",
    mergedAt = "2026-08-10T12:20:00Z",
    queueEvent = "added_to_merge_queue",
  },
) {
  f.github.prs.set(`${repo}:${number}`, {
    html_url: `https://github.com/${REPOSITORIES[repo].slug}/pull/${number}`,
    state: "closed",
    head: { ref: head, sha: headSha, repo: { full_name: REPOSITORIES[repo].slug } },
    base: { ref: base, repo: { full_name: REPOSITORIES[repo].slug } },
    merged_at: mergedAt,
    merge_commit_sha: mergeCommit,
  });
  if (!f.github.checks.has(`${repo}:${headSha}`)) {
    f.github.checks.set(`${repo}:${headSha}`, successfulChecks(repo, 8000 + number * 10));
  }
  f.github.timelines.set(`${repo}:${number}`, [{ event: queueEvent, created_at: queuedAt }]);
}

function allowFixtureAncestry(f, repo, ancestor, descendant) {
  f.github.comparisons.set(`${repo}:${ancestor}:${descendant}`, {
    status: ancestor === descendant ? "identical" : "ahead",
  });
}

function installStagedPolicyPair(f, state) {
  for (const key of Object.keys(REPOSITORIES)) {
    const policy = state.policies[key].exactQueue;
    const created = f.github.addRuleset(key, policy.stagedPayload);
    policy.id = created.id;
    policy.status = "disabled";
    state.transaction.policyMutations.push({
      action: "create-ruleset",
      repo: key,
      name: policy.name,
      digest: policy.stagedDigest,
      id: created.id,
      at: EPOCH,
      status: "created",
    });
  }
}

function activatePolicyPair(f, state) {
  for (const key of Object.keys(REPOSITORIES)) {
    const policy = state.policies[key].exactQueue;
    f.github.updateRuleset(REPOSITORIES[key].slug, policy.id, policy.payload);
    policy.status = "active";
    state.transaction.policyMutations.push({
      action: "activate-ruleset",
      repo: key,
      name: policy.name,
      digest: policy.digest,
      id: policy.id,
      at: EPOCH,
      status: "created",
    });
  }
}

function activePolicyRecoveryState(f, { sceneWorksRulesetId } = {}) {
  const state = planFixture(f);
  installStagedPolicyPair(f, state);
  if (sceneWorksRulesetId !== undefined) {
    const policy = state.policies.sceneworks.exactQueue;
    const live = f.github.ruleSets.sceneworks.find(({ id }) => id === policy.id);
    const creation = state.transaction.policyMutations.find(
      ({ repo, action }) => repo === "sceneworks" && action === "create-ruleset",
    );
    live.id = sceneWorksRulesetId;
    policy.id = sceneWorksRulesetId;
    creation.id = sceneWorksRulesetId;
  }
  activatePolicyPair(f, state);
  state.phase = "recovery-required";
  state.transaction.recoveryReason = "simulated active exact queue created before its missing ref";
  writeJsonAtomic(f.statePath, state);
  return state;
}

function advanceMain(origin, file = "advance.txt", text = "advance\n") {
  const work = mkdtempSync(join(tmpdir(), "feature-epic-advance-"));
  try {
    run("git", ["clone", origin, work]);
    run("git", ["config", "user.name", "Feature Epic Fixture"], { cwd: work });
    run("git", ["config", "user.email", "fixture@example.test"], { cwd: work });
    commitFile(work, file, text, "advance main");
    run("git", ["push", "origin", "main"], { cwd: work });
    return run("git", ["rev-parse", "HEAD"], { cwd: work });
  } finally {
    rmSync(work, { recursive: true, force: true });
  }
}

function sideCommit(origin, file = "divergent.txt") {
  const work = mkdtempSync(join(tmpdir(), "feature-epic-side-"));
  try {
    run("git", ["clone", origin, work]);
    run("git", ["config", "user.name", "Feature Epic Fixture"], { cwd: work });
    run("git", ["config", "user.email", "fixture@example.test"], { cwd: work });
    run("git", ["switch", "-c", "fixture-divergent"], { cwd: work });
    const sha = commitFile(work, file, "divergent\n", "divergent side commit");
    run("git", ["push", "origin", "HEAD:refs/heads/fixture-divergent"], { cwd: work });
    return sha;
  } finally {
    rmSync(work, { recursive: true, force: true });
  }
}

test("branch names carry both Shortcut ownership ids and reject default-like feature refs", () => {
  assert.equal(
    storyBranch("sc-18419", "sc-18304", "pipeline-flexibility-mlx-perf"),
    "story/sc-18419-epic-18304-pipeline-flexibility-mlx-perf",
  );
  assert.match(featureBranch("sc-18304", "pipeline-flexibility-mlx-perf"), FEATURE_RE);
  assert.deepEqual(
    classifyAdoptedPullRequest("precanonical-story", "story/sc-18418-feature-epic-ci", "main"),
    { kind: "story", story: "sc-18418", epic: null, slug: null },
  );
  assert.throws(
    () => classifyAdoptedPullRequest(
      "precanonical-story",
      "story/sc-18418-epic-18304-ruleset-proof",
      "main",
    ),
    /precanonical story adoption requires/,
  );
  assert.throws(() => featureBranch("18304", "valid-slug"), /canonical sc-N/);
  assert.throws(() => featureBranch("sc-18304", "Not-Kebab"), /lower-case kebab-case/);
  assert.throws(
    () => createPlan({ epic: "sc-18304", slug: "main", stories: ["sc-1"], workspaceParent: "/tmp", statePath: "/tmp/x" }),
    /default-like/,
  );
});

test("policy payloads layer wildcard protection under one exact one-entry queue", () => {
  for (const repo of Object.keys(REPOSITORIES)) {
    const policies = featurePolicyPayloads(
      repo,
      "feature/sc-18304-pipeline-flexibility-mlx-perf",
      "sc-18304",
    );
    assert.deepEqual(policies.wildcardBase.conditions.ref_name.include, ["refs/heads/feature/*"]);
    assert.deepEqual(
      policies.wildcardBase.rules.map(({ type }) => type),
      ["non_fast_forward", "pull_request", "required_status_checks"],
    );
    assert.equal(policies.wildcardBase.rules.some(({ type }) => type === "merge_queue"), false);
    assert.deepEqual(policies.deletionGuard.rules, [{ type: "deletion" }]);
    assert.deepEqual(policies.exactQueue.conditions.ref_name.include, [
      "refs/heads/feature/sc-18304-pipeline-flexibility-mlx-perf",
    ]);
    assert.equal(policies.exactQueue.rules.length, 1);
    assert.equal(policies.exactQueue.rules[0].type, "merge_queue");
    assert.equal(policies.exactQueue.rules[0].parameters.max_entries_to_build, 5);
    assert.equal(policies.exactQueue.rules[0].parameters.max_entries_to_merge, 1);
    assert.equal(policies.exactQueue.rules[0].parameters.min_entries_to_merge_wait_minutes, 5);
    assert.equal(policies.exactQueue.enforcement, "active");
    assert.equal(policies.stagedExactQueue.enforcement, "disabled");
    assert.deepEqual(policies.stagedExactQueue.rules, policies.exactQueue.rules);
    for (const check of policies.wildcardBase.rules.find(({ type }) => type === "required_status_checks")
      .parameters.required_status_checks) {
      assert.equal(check.integration_id, 15368);
    }
  }
});

test("Shortcut client keeps credentials off argv and the epic snapshot proves exhaustive state", () => {
  const calls = [];
  const fake = new FakeShortcut();
  const processRunner = {
    exec(command, args, options) {
      calls.push({ command, args, input: options.input });
      const url = args.at(-1);
      if (url.endsWith("/epics/18304")) return JSON.stringify(fake.epicValue);
      if (url.endsWith("/epics/18304/stories")) return JSON.stringify(fake.storyValues);
      if (url.endsWith("/workflows")) return JSON.stringify(fake.workflowValues);
      assert.fail(`unexpected Shortcut URL ${url}`);
    },
  };
  const client = new ShortcutClient(processRunner, "super-secret-fixture-token");
  const snapshot = inspectShortcutEpic(client, "sc-18304", ["sc-18418", "sc-18419"]);
  assert.equal(snapshot.inventory.matches, true);
  assert.equal(snapshot.gates.allStoriesDone, true);
  assert.equal(snapshot.gates.noBlockedStories, true);
  assert.equal(snapshot.ok, true);
  assert.equal(calls.length, 3);
  for (const call of calls) {
    assert.equal(call.command, "curl");
    assert.equal(call.args.join(" ").includes("super-secret-fixture-token"), false);
    assert.match(call.input, /Shortcut-Token: super-secret-fixture-token/);
  }

  fake.epicValue.stats.num_stories_total += 1;
  assert.throws(
    () => inspectShortcutEpic(fake, "sc-18304", ["sc-18418", "sc-18419"]),
    /reports 3 stories but returned 2/,
  );
});

test("plan binds the exact live Shortcut inventory before any local or remote mutation", (t) => {
  const f = fixture(t);
  f.shortcut.storyValues.push({
    ...structuredClone(f.shortcut.storyValues[0]),
    id: 19000,
    name: "Unplanned live story",
  });
  f.shortcut.epicValue.stats.num_stories_total = f.shortcut.storyValues.length;
  assert.throws(
    () => createPlan(
      {
        epic: "sc-18304",
        slug: "pipeline-flexibility-mlx-perf",
        stories: ["sc-18418", "sc-18419"],
        workspaceParent: f.workspaceParent,
        statePath: f.statePath,
      },
      { github: f.github, shortcut: f.shortcut, clock },
    ),
    /unexpected sc-19000/,
  );
  assert.equal(existsSync(f.statePath), false);
  assert.equal(existsSync(stateLockPath(f.statePath)), false);
  assert.deepEqual(f.github.mutations, []);
});

test("inventory refresh journals additive live scope and freezes on final-PR intent", (t) => {
  const f = fixture(t);
  const state = planFixture(f);
  f.shortcut.storyValues.push({
    ...structuredClone(f.shortcut.storyValues[0]),
    id: 19000,
    name: "Added during execution",
    workflow_state_id: 999,
  });
  f.shortcut.epicValue.stats.num_stories_total = f.shortcut.storyValues.length;
  assert.throws(
    () => refreshStoryInventory(
      f.statePath,
      { apply: true },
      { github: f.github, shortcut: f.shortcut, clock },
    ),
    /unknown workflow state/,
  );
  assert.deepEqual(loadState(f.statePath).expectedStories, state.expectedStories);
  f.shortcut.storyValues.at(-1).workflow_state_id = 30;
  f.shortcut.workflowValues[0].states[1].name = "Verified Done";
  const { state: refreshed, refresh } = refreshStoryInventory(
    f.statePath,
    { apply: true },
    { github: f.github, shortcut: f.shortcut, clock },
  );
  assert.deepEqual(refresh.added, ["sc-19000"]);
  assert.deepEqual(refresh.removed, []);
  assert.deepEqual(refresh.workflowStateChanges.map(({ id }) => id), [30]);
  assert.deepEqual(refreshed.expectedStories, ["sc-18418", "sc-18419", "sc-19000"]);
  assert.deepEqual(loadState(f.statePath).shortcut.refreshes, [refresh]);

  f.shortcut.storyValues = f.shortcut.storyValues.filter(({ id }) => id !== 18418);
  f.shortcut.epicValue.stats.num_stories_total = f.shortcut.storyValues.length;
  assert.throws(
    () => refreshStoryInventory(
      f.statePath,
      { apply: true },
      { github: f.github, shortcut: f.shortcut, clock },
    ),
    /additive-only.*sc-18418/,
  );
  assert.deepEqual(loadState(f.statePath).expectedStories, refreshed.expectedStories);

  f.shortcut.storyValues.unshift({
    ...structuredClone(f.shortcut.storyValues[0]),
    id: 18418,
    name: "Fixture sc-18418",
  });
  f.shortcut.storyValues.push({
    ...structuredClone(f.shortcut.storyValues[0]),
    id: 19001,
    name: "Too late for final integration",
  });
  f.shortcut.epicValue.stats.num_stories_total = f.shortcut.storyValues.length;
  f.github.prs.set("sceneworks:99", {
    html_url: "https://github.com/SceneWorks/SceneWorks/pull/99",
    state: "open",
    head: {
      ref: state.featureBranch,
      sha: state.repositories.sceneworks.plannedMainSha,
      repo: { full_name: REPOSITORIES.sceneworks.slug },
    },
    base: { ref: "main", repo: { full_name: REPOSITORIES.sceneworks.slug } },
  });
  assert.throws(
    () => refreshStoryInventory(
      f.statePath,
      { apply: true },
      { github: f.github, shortcut: f.shortcut, clock },
    ),
    /inventory is frozen.*final-integration PR intent/,
  );
  const drifted = auditState(
    f.statePath,
    f.reportPath,
    { github: f.github, shortcut: f.shortcut, clock },
  );
  assert.equal(drifted.gates.shortcutInventoryMatches, false);
  assert.equal(drifted.gates.noOpenTrainPullRequests, false);
  assert.equal(drifted.cleanup.eligible, false);
});

test("state validation rejects a tampered Shortcut inventory-refresh chain", (t) => {
  const f = fixture(t);
  planFixture(f);
  f.shortcut.storyValues.push({
    ...structuredClone(f.shortcut.storyValues[0]),
    id: 19000,
    name: "Added during execution",
  });
  f.shortcut.epicValue.stats.num_stories_total = f.shortcut.storyValues.length;
  refreshStoryInventory(
    f.statePath,
    { apply: true },
    { github: f.github, shortcut: f.shortcut, clock },
  );
  const tampered = JSON.parse(readFileSync(f.statePath, "utf8"));
  tampered.shortcut.refreshes[0].previousDigest = "0".repeat(64);
  writeJsonAtomic(f.statePath, tampered);
  assert.throws(() => loadState(f.statePath), /inventory-refresh chain is inconsistent/);
});

test("plan validates fixed live mains and atomically records one reachable pin before mutation", (t) => {
  const f = fixture(t);
  const state = planFixture(f);
  assert.equal(state.phase, "planned");
  assert.equal(state.repositories.sceneworks.plannedMainSha, f.sceneWorksMain);
  assert.equal(state.repositories.inference.plannedMainSha, f.inferenceMain);
  assert.equal(state.inferencePin.sha, f.inferenceMain);
  assert.deepEqual(f.github.featureSha(REPOSITORIES.sceneworks.slug, state.featureBranch), null);
  assert.deepEqual(f.github.featureSha(REPOSITORIES.inference.slug, state.featureBranch), null);
  assert.deepEqual(loadState(f.statePath), state);
  assert.deepEqual(f.github.checkQueries.slice(0, 2), [
    { repo: "sceneworks", sha: f.sceneWorksMain },
    { repo: "inference", sha: f.inferenceMain },
  ]);
});

test("plan requires trusted terminal checks on both exact main commits", async (t) => {
  const later = "2026-08-10T13:00:00Z";
  const cases = [
    ["missing", (runs) => runs.slice(1)],
    ["pending", (runs) => [trustedCheck(runs[0].name, runs[0].id, {
      status: "in_progress",
      conclusion: null,
    }), ...runs.slice(1)]],
    ["failed", (runs) => [trustedCheck(runs[0].name, runs[0].id, { conclusion: "failure" }), ...runs.slice(1)]],
    ["cancelled", (runs) => [trustedCheck(runs[0].name, runs[0].id, { conclusion: "cancelled" }), ...runs.slice(1)]],
    ["wrong app", (runs) => [trustedCheck(runs[0].name, runs[0].id, { app: { id: 1 } }), ...runs.slice(1)]],
    ["newer failed duplicate", (runs) => [
      ...runs,
      trustedCheck(runs[0].name, runs[0].id + 1000, {
        conclusion: "failure",
        started_at: later,
        completed_at: later,
      }),
    ]],
    ["newer queued duplicate without timestamps", (runs) => [
      ...runs,
      trustedCheck(runs[0].name, runs[0].id + 1000, {
        status: "queued",
        conclusion: null,
        started_at: null,
        completed_at: null,
      }),
    ]],
  ];
  for (const repo of Object.keys(REPOSITORIES)) {
    for (const [label, mutate] of cases) {
      await t.test(`${repo}: ${label}`, (t) => {
        const f = fixture(t);
        const sha = repo === "sceneworks" ? f.sceneWorksMain : f.inferenceMain;
        const key = `${repo}:${sha}`;
        f.github.checks.set(key, mutate(f.github.checks.get(key)));
        assert.throws(() => planFixture(f), /lack terminal trusted required checks/);
        assert.equal(existsSync(f.statePath), false, "failed preflight wrote no state");
        assert.deepEqual(f.github.mutations, [], "failed preflight made no remote mutation");
      });
    }
  }
  await t.test("newer trusted success supersedes stale trusted failure", (t) => {
    const f = fixture(t);
    const key = `sceneworks:${f.sceneWorksMain}`;
    const runs = f.github.checks.get(key);
    runs[0] = trustedCheck(runs[0].name, runs[0].id, { conclusion: "failure" });
    runs.push(trustedCheck(runs[0].name, runs[0].id + 1000, {
      started_at: later,
      completed_at: later,
    }));
    assert.equal(planFixture(f).phase, "planned");
  });
  await t.test("newer wrong-app duplicate cannot replace trusted evidence", (t) => {
    const f = fixture(t);
    const key = `inference:${f.inferenceMain}`;
    const runs = f.github.checks.get(key);
    runs.push(trustedCheck(runs[0].name, runs[0].id + 1000, {
      app: { id: 1 },
      conclusion: "failure",
      started_at: later,
      completed_at: later,
    }));
    assert.equal(planFixture(f).phase, "planned");
  });
  await t.test("neutral and skipped trusted conclusions are terminal", (t) => {
    const f = fixture(t);
    const runs = f.github.checks.get(`sceneworks:${f.sceneWorksMain}`);
    runs[0].conclusion = "neutral";
    runs[1].conclusion = "skipped";
    assert.equal(planFixture(f).phase, "planned");
  });
});

test("check-run pagination drives exact latest trusted main-check decisions", async (t) => {
  const later = "2026-08-10T13:00:00Z";
  const makePageOne = (checks) => {
    const page = structuredClone(checks);
    while (page.length < 100) page.push(trustedCheck(`decoy-${page.length}`, 10000 + page.length));
    return page;
  };
  const installPagedChecks = (f, pages) => {
    const path = `repos/${REPOSITORIES.sceneworks.slug}/commits/${f.sceneWorksMain}/check-runs?per_page=100&filter=all`;
    const process = new FakeGhProcess((args) => {
      assert.equal(args[1], path);
      return pages.map((check_runs) => ({ total_count: pages.flat().length, check_runs }));
    });
    const client = new GitHubClient(process);
    const original = f.github.checkRuns.bind(f.github);
    f.github.checkRuns = (slug, sha) => slug === REPOSITORIES.sceneworks.slug
      ? client.checkRuns(slug, sha)
      : original(slug, sha);
    return process;
  };

  await t.test("newer trusted failure after item 100 blocks", (t) => {
    const f = fixture(t);
    const pageOne = makePageOne(successfulChecks("sceneworks", 100));
    const process = installPagedChecks(f, [pageOne, [trustedCheck(pageOne[0].name, 50000, {
      conclusion: "failure",
      started_at: later,
      completed_at: later,
    })]]);
    assert.throws(() => planFixture(f), /lack terminal trusted required checks/);
    assert.ok(process.calls[0].includes("--paginate"));
    assert.ok(process.calls[0].includes("--slurp"));
  });

  await t.test("newer trusted success after item 100 supersedes stale failure", (t) => {
    const f = fixture(t);
    const checks = successfulChecks("sceneworks", 100);
    checks[0].conclusion = "failure";
    const pageOne = makePageOne(checks);
    installPagedChecks(f, [pageOne, [trustedCheck(pageOne[0].name, 50000, {
      started_at: later,
      completed_at: later,
    })]]);
    assert.equal(planFixture(f).phase, "planned");
  });

  await t.test("wrong-app success after item 100 never satisfies a missing context", (t) => {
    const f = fixture(t);
    const checks = successfulChecks("sceneworks", 100).slice(1);
    const pageOne = makePageOne(checks);
    installPagedChecks(f, [pageOne, [trustedCheck(REQUIRED_CONTEXTS.sceneworks[0], 50000, {
      app: { id: 1 },
      started_at: later,
      completed_at: later,
    })]]);
    assert.throws(() => planFixture(f), /lack terminal trusted required checks/);
  });
});

test("plan refuses reused workspace, non-main default, missing/inconsistent/unreachable pins", async (t) => {
  await t.test("reused workspace", (t) => {
    const f = fixture(t);
    writeFileSync(join(f.workspaceParent, "owned-by-someone-else"), "x");
    assert.throws(() => planFixture(f), /must be empty/);
  });
  await t.test("non-main default", (t) => {
    const f = fixture(t);
    f.github.defaults.inference = "release/next";
    assert.throws(() => planFixture(f), /not main/);
  });
  await t.test("missing pin", (t) => {
    const f = fixture(t, { missingPin: true });
    assert.throws(() => planFixture(f), /no tracked Cargo manifest pins/);
  });
  await t.test("inconsistent pin", (t) => {
    const other = "1111111111111111111111111111111111111111";
    const f = fixture(t, { secondPin: other });
    assert.throws(() => planFixture(f), /inconsistent inference pins/);
  });
  await t.test("unreachable pin", (t) => {
    const f = fixture(t, { pin: "1111111111111111111111111111111111111111" });
    assert.throws(() => planFixture(f), /not reachable/);
  });
});

test("plan refuses wildcard queue, duplicate, mismatched, and partial exact policies", async (t) => {
  await t.test("wildcard merge queue is rejected like GitHub's HTTP 422 constraint", (t) => {
    const f = fixture(t);
    const base = f.github.ruleSets.sceneworks.find(({ name }) => name === "Feature integration base policy");
    base.rules.push(featurePolicyPayloads(
      "sceneworks",
      "feature/sc-18304-pipeline-flexibility-mlx-perf",
      "sc-18304",
    ).exactQueue.rules[0]);
    assert.throws(() => planFixture(f), /does not match the required semantic payload/);
  });
  await t.test("duplicate wildcard policy", (t) => {
    const f = fixture(t);
    f.github.ruleSets.sceneworks.push(structuredClone(f.github.ruleSets.sceneworks[0]));
    f.github.ruleSets.sceneworks.at(-1).id = 9999;
    assert.throws(() => planFixture(f), /exactly two wildcard feature policies/);
  });
  await t.test("mismatched required check app", (t) => {
    const f = fixture(t);
    const base = f.github.ruleSets.inference.find(({ name }) => name === "Feature integration base policy");
    base.rules.find(({ type }) => type === "required_status_checks")
      .parameters.required_status_checks[0].integration_id = 1;
    assert.throws(() => planFixture(f), /does not match the required semantic payload/);
  });
  await t.test("one pre-existing exact queue", (t) => {
    const f = fixture(t);
    const exact = featurePolicyPayloads(
      "sceneworks",
      "feature/sc-18304-pipeline-flexibility-mlx-perf",
      "sc-18304",
    ).exactQueue;
    f.github.addRuleset("sceneworks", exact);
    assert.throws(() => planFixture(f), /exactly one repository already has/);
  });
  await t.test("disabled exact queue outside a transaction", (t) => {
    const f = fixture(t);
    const staged = featurePolicyPayloads(
      "sceneworks",
      "feature/sc-18304-pipeline-flexibility-mlx-perf",
      "sc-18304",
    ).stagedExactQueue;
    f.github.addRuleset("sceneworks", staged);
    f.github.addRuleset("inference", featurePolicyPayloads(
      "inference",
      "feature/sc-18304-pipeline-flexibility-mlx-perf",
      "sc-18304",
    ).stagedExactQueue);
    assert.throws(() => planFixture(f), /disabled exact queue exists outside/);
  });
  await t.test("active queue before a ref makes GitHub create-ref impossible", (t) => {
    const f = fixture(t);
    const branch = "feature/sc-18304-pipeline-flexibility-mlx-perf";
    for (const key of Object.keys(REPOSITORIES)) {
      f.github.addRuleset(key, featurePolicyPayloads(key, branch, "sc-18304").exactQueue);
    }
    assert.throws(() => planFixture(f), /explicit adopt-existing mode is required/);
    assert.throws(
      () => f.github.createFeatureRef(REPOSITORIES.sceneworks.slug, branch, f.sceneWorksMain),
      /HTTP 422.*merge queue/,
    );
  });
  await t.test("duplicate exact queue", (t) => {
    const f = fixture(t);
    const exact = featurePolicyPayloads(
      "sceneworks",
      "feature/sc-18304-pipeline-flexibility-mlx-perf",
      "sc-18304",
    ).exactQueue;
    f.github.addRuleset("sceneworks", exact);
    f.github.addRuleset("sceneworks", exact);
    assert.throws(() => planFixture(f), /duplicate (?:or divergent )?(?:Feature queue|exact queue)/);
  });
  await t.test("same epic queue name on a divergent ref", (t) => {
    const f = fixture(t);
    const exact = featurePolicyPayloads(
      "sceneworks",
      "feature/sc-18304-pipeline-flexibility-mlx-perf",
      "sc-18304",
    ).exactQueue;
    exact.conditions.ref_name.include = ["refs/heads/feature/sc-18304-wrong-ref"];
    f.github.addRuleset("sceneworks", exact);
    assert.throws(() => planFixture(f), /duplicate or divergent/);
  });
});

test("adopt-existing plans immutable protected heads and bootstraps without remote mutation", (t) => {
  const f = fixture(t);
  const state = adoptionPlanFixture(f);
  assert.equal(state.mode, "adopt-existing");
  assert.equal(state.shortcut.inventoryFrozenAt, new Date(EPOCH).toISOString());
  for (const key of Object.keys(REPOSITORIES)) {
    assert.equal(state.repositories[key].adoptedFeatureSha, state.repositories[key].plannedMainSha);
    assert.equal(
      state.repositories[key].adoptedRequiredChecks.length,
      REQUIRED_CONTEXTS[key].length,
    );
    assert.equal(state.policies[key].exactQueue.status, "active");
  }
  assert.deepEqual(state.transaction.remoteMutations, []);
  assert.deepEqual(state.transaction.policyMutations, []);
  assert.deepEqual(f.github.mutations, []);

  const ready = bootstrap(
    f.statePath,
    { apply: true },
    { github: f.github, git: f.git, clock },
  );
  assert.equal(ready.phase, "ready");
  assert.deepEqual(ready.transaction.remoteMutations, []);
  assert.deepEqual(ready.transaction.policyMutations, []);
  assert.deepEqual(f.github.mutations, []);
});

test("adopt-existing fails closed on stale heads, policies, checks, or paired pin evidence", async (t) => {
  await t.test("planned main moved", (t) => {
    const f = fixture(t);
    adoptionPlanFixture(f);
    advanceMain(f.origins.sceneworks);
    assert.throws(
      () => bootstrap(f.statePath, { apply: true }, { github: f.github, git: f.git, clock }),
      /main moved from planned/,
    );
    assert.deepEqual(f.github.mutations, []);
  });
  await t.test("adopted feature moved", (t) => {
    const f = fixture(t);
    const state = adoptionPlanFixture(f);
    const moved = sideCommit(f.origins.inference, "moved-feature.txt");
    run("git", ["update-ref", `refs/heads/${state.featureBranch}`, moved], { cwd: f.origins.inference });
    assert.throws(
      () => bootstrap(f.statePath, { apply: true }, { github: f.github, git: f.git, clock }),
      /adopted feature moved/,
    );
    assert.deepEqual(f.github.mutations, []);
  });
  await t.test("exact policy changed", (t) => {
    const f = fixture(t);
    const state = adoptionPlanFixture(f);
    const exact = f.github.ruleSets.sceneworks.find(
      ({ id }) => id === state.policies.sceneworks.exactQueue.id,
    );
    exact.enforcement = "disabled";
    assert.throws(
      () => bootstrap(f.statePath, { apply: true }, { github: f.github, git: f.git, clock }),
      /adopted protection differs/,
    );
    assert.deepEqual(f.github.mutations, []);
  });
  await t.test("adopted required check changed", (t) => {
    const f = fixture(t);
    const state = adoptionPlanFixture(f);
    const sha = state.repositories.sceneworks.adoptedFeatureSha;
    f.github.checks.get(`sceneworks:${sha}`).push(trustedCheck(REQUIRED_CONTEXTS.sceneworks[0], 99999, {
      conclusion: "failure",
      started_at: "2026-08-10T13:00:00Z",
      completed_at: "2026-08-10T13:00:00Z",
    }));
    assert.throws(
      () => bootstrap(f.statePath, { apply: true }, { github: f.github, git: f.git, clock }),
      /required checks no longer match/,
    );
    assert.deepEqual(f.github.mutations, []);
  });
  await t.test("SceneWorks pin changed", (t) => {
    const f = fixture(t);
    adoptionPlanFixture(f);
    const original = f.github.cargoManifests.bind(f.github);
    f.github.cargoManifests = (slug, sha) => slug === REPOSITORIES.sceneworks.slug
      ? [{ path: "Cargo.toml", text: manifest("f".repeat(40)) }]
      : original(slug, sha);
    assert.throws(
      () => bootstrap(f.statePath, { apply: true }, { github: f.github, git: f.git, clock }),
      /pin pairing changed or is unreachable/,
    );
    assert.deepEqual(f.github.mutations, []);
  });
});

test("adopt-pr captures exact historical train and governance-proof receipts without proof coverage", (t) => {
  const f = fixture(t, { stories: ["sc-18418"] });
  const state = adoptionPlanFixture(f, { stories: ["sc-18418"] });
  const adoptionClock = () => new Date("2026-08-10T18:00:00Z");
  const swFeature = state.repositories.sceneworks.adoptedFeatureSha;
  const inferenceFeature = state.repositories.inference.adoptedFeatureSha;
  const liveShapes = [
    {
      repo: "sceneworks",
      number: 2237,
      disposition: "governance-proof",
      head: "story/sc-18418-epic-18304-ruleset-proof",
      base: "feature/ruleset-proof-sc-18418",
      headSha: "1d019af7c9d745cd8ffb1dc06708ab277519a5a1",
      mergeCommit: "6ec560f55dbb7d48a9cd64b3e416d3f662959013",
      queuedAt: "2026-08-10T13:22:00Z",
      mergedAt: "2026-08-10T13:43:05Z",
    },
    {
      repo: "inference",
      number: 534,
      disposition: "governance-proof",
      head: "story/sc-18418-epic-18304-ruleset-proof",
      base: "feature/ruleset-proof-sc-18418",
      headSha: "e829418f0c838b7a75aaf1a57f88011f57151c15",
      mergeCommit: "ceec98828c6ff3fe335398507161931c47a88937",
      queuedAt: "2026-08-10T13:08:00Z",
      mergedAt: "2026-08-10T13:11:17Z",
    },
    {
      repo: "sceneworks",
      number: 2233,
      disposition: "precanonical-story",
      head: "story/sc-18418-feature-epic-ci",
      base: "main",
      headSha: "0054b3ecab423a0c0a21ca4cc0aef0166dc94c9a",
      mergeCommit: "1dd7717f64a36ca5249d51b7b86c16981a23251e",
      queuedAt: "2026-08-10T12:26:00Z",
      mergedAt: "2026-08-10T12:44:59Z",
    },
    {
      repo: "sceneworks",
      number: 2236,
      disposition: "precanonical-story",
      head: "story/sc-18418-ruleset-scope-fix",
      base: "main",
      headSha: "b6d6830ed28d1094468441ec322b068d8ef097c8",
      mergeCommit: "2f739d175aaf477f907cd4260326501340a8c17c",
      queuedAt: "2026-08-10T13:31:33Z",
      mergedAt: "2026-08-10T13:52:02Z",
    },
    {
      repo: "sceneworks",
      number: 2239,
      disposition: "historical-train",
      head: "sync/sc-18304-main-2026-08-10",
      base: state.featureBranch,
      headSha: "a06b8e8a2d57a4792b2e5951bf48ec0e88f79d20",
      mergeCommit: "fa9ad8998d6d61207b167710e6cb6203101b9181",
      queuedAt: "2026-08-10T16:58:37Z",
      mergedAt: "2026-08-10T17:25:14Z",
    },
  ];
  for (const shape of liveShapes) {
    mergedAdoptionPr(f, state, shape);
    if (shape.disposition !== "governance-proof") {
      const baseline = shape.repo === "sceneworks" ? swFeature : inferenceFeature;
      allowFixtureAncestry(f, shape.repo, shape.mergeCommit, baseline);
      if (shape.disposition === "precanonical-story") {
        allowFixtureAncestry(
          f,
          shape.repo,
          shape.mergeCommit,
          state.repositories[shape.repo].plannedMainSha,
        );
      }
    }
  }

  const proof = adoptPullRequest(
    f.statePath,
    { repo: "sceneworks", number: 2237, disposition: "governance-proof" },
    { github: f.github, clock: adoptionClock },
  );
  assert.equal(proof.kind, "governance-proof");
  assert.equal(proof.canonicalFeatureSha, null);
  assert.deepEqual(proof.queue, {
    event: "added_to_merge_queue",
    at: "2026-08-10T13:22:00Z",
    headSha: liveShapes[0].headSha,
    mergeCommit: liveShapes[0].mergeCommit,
  });
  let report = auditState(
    f.statePath,
    f.reportPath,
    { github: f.github, shortcut: f.shortcut, clock: adoptionClock },
  );
  assert.equal(report.expectedStories["sc-18418"], false);
  assert.equal(
    report.livePullRequests.pullRequests.find(({ number }) => number === 2237).topologyError,
    null,
  );

  for (const shape of liveShapes.slice(1)) {
    adoptPullRequest(
      f.statePath,
      { repo: shape.repo, number: shape.number, disposition: shape.disposition },
      { github: f.github, clock: adoptionClock },
    );
  }
  const adopted = loadState(f.statePath).adoptedPullRequests;
  assert.deepEqual(adopted.map(({ repo, number }) => [repo, number]), [
    ["sceneworks", 2237],
    ["inference", 534],
    ["sceneworks", 2233],
    ["sceneworks", 2236],
    ["sceneworks", 2239],
  ]);
  assert.equal(adopted.find(({ number }) => number === 2239).kind, "sync");
  report = auditState(
    f.statePath,
    f.reportPath,
    { github: f.github, shortcut: f.shortcut, clock: adoptionClock },
  );
  assert.equal(report.expectedStories["sc-18418"], true);
  assert.equal(report.gates.allAdoptedReceiptsExact, true);
  assert.equal(report.gates.adoptedBaselineVerified, true);
  assert.deepEqual(f.github.mutations, []);
});

test("adopt-pr refuses waivers, open or malformed history, missing evidence, and non-ancestral merges", (t) => {
  const f = fixture(t, { stories: ["sc-18418"] });
  const state = adoptionPlanFixture(f, { stories: ["sc-18418"] });
  const adoptionClock = () => new Date("2026-08-10T18:00:00Z");
  const canonicalHead = "story/sc-18418-epic-18304-pipeline-flexibility-mlx-perf";
  mergedAdoptionPr(f, state, {
    number: 2300,
    head: canonicalHead,
    base: state.featureBranch,
  });
  assert.throws(
    () => adoptPullRequest(
      f.statePath,
      { repo: "sceneworks", number: 2300, disposition: "waiver" },
      { github: f.github, clock: adoptionClock },
    ),
    /unknown adoption disposition/,
  );

  const open = f.github.prs.get("sceneworks:2300");
  open.state = "open";
  open.merged_at = null;
  open.merge_commit_sha = null;
  assert.throws(
    () => adoptPullRequest(
      f.statePath,
      { repo: "sceneworks", number: 2300, disposition: "historical-train" },
      { github: f.github, clock: adoptionClock },
    ),
    /only an already-merged PR/,
  );

  mergedAdoptionPr(f, state, {
    number: 2301,
    head: "story/sc-18418-wrong-base",
    base: state.featureBranch,
  });
  assert.throws(
    () => adoptPullRequest(
      f.statePath,
      { repo: "sceneworks", number: 2301, disposition: "precanonical-story" },
      { github: f.github, clock: adoptionClock },
    ),
    /requires story\/sc-N-slug -> main/,
  );

  mergedAdoptionPr(f, state, {
    number: 2302,
    head: canonicalHead,
    base: state.featureBranch,
  });
  f.github.timelines.set("sceneworks:2302", []);
  assert.throws(
    () => adoptPullRequest(
      f.statePath,
      { repo: "sceneworks", number: 2302, disposition: "historical-train" },
      { github: f.github, clock: adoptionClock },
    ),
    /no timestamped merge-queue event/,
  );

  const nonAncestor = "9".repeat(40);
  mergedAdoptionPr(f, state, {
    number: 2303,
    head: canonicalHead,
    base: state.featureBranch,
    mergeCommit: nonAncestor,
  });
  assert.throws(
    () => adoptPullRequest(
      f.statePath,
      { repo: "sceneworks", number: 2303, disposition: "historical-train" },
      { github: f.github, clock: adoptionClock },
    ),
    /not an ancestor of the canonical feature head/,
  );

  mergedAdoptionPr(f, state, {
    number: 2304,
    head: canonicalHead,
    base: state.featureBranch,
  });
  f.github.checks.set(`sceneworks:${state.repositories.sceneworks.plannedMainSha}`, []);
  assert.throws(
    () => adoptPullRequest(
      f.statePath,
      { repo: "sceneworks", number: 2304, disposition: "historical-train" },
      { github: f.github, clock: adoptionClock },
    ),
    /required checks no longer match|lacks terminal trusted required checks/,
  );
  assert.equal(loadState(f.statePath).adoptedPullRequests.length, 0);
  assert.deepEqual(f.github.mutations, []);
});

test("state validation exhaustively binds PR, adoption, final, runtime, and deployment receipts", async (t) => {
  const f = fixture(t, { stories: ["sc-18418"] });
  const state = adoptionPlanFixture(f, { stories: ["sc-18418"], deploymentRequired: true });
  const laterClock = () => new Date("2026-08-10T18:00:00Z");
  const historicalHead = "7".repeat(40);
  mergedAdoptionPr(f, state, {
    number: 2233,
    head: "story/sc-18418-feature-epic-ci",
    base: "main",
    headSha: historicalHead,
    mergeCommit: state.repositories.sceneworks.plannedMainSha,
  });
  adoptPullRequest(
    f.statePath,
    { repo: "sceneworks", number: 2233, disposition: "precanonical-story" },
    { github: f.github, clock: laterClock },
  );

  const finalNumber = 2400;
  const finalHead = state.repositories.sceneworks.adoptedFeatureSha;
  f.github.prs.set(`sceneworks:${finalNumber}`, {
    html_url: `https://github.com/SceneWorks/SceneWorks/pull/${finalNumber}`,
    state: "open",
    head: {
      ref: state.featureBranch,
      sha: finalHead,
      repo: { full_name: REPOSITORIES.sceneworks.slug },
    },
    base: { ref: "main", repo: { full_name: REPOSITORIES.sceneworks.slug } },
  });
  f.github.runs.set("sceneworks:4400", {
    head_sha: finalHead,
    html_url: "https://github.com/SceneWorks/SceneWorks/actions/runs/4400",
    path: ".github/workflows/windows-candle.yml",
    event: "workflow_dispatch",
    status: "completed",
    conclusion: "success",
  });
  recordEvidence(
    f.statePath,
    {
      repo: "sceneworks",
      number: finalNumber,
      runReceipts: [{ repo: "sceneworks", id: 4400 }],
    },
    { github: f.github, clock: laterClock },
  );
  const finalMerge = state.repositories.sceneworks.plannedMainSha;
  Object.assign(f.github.prs.get(`sceneworks:${finalNumber}`), {
    state: "closed",
    merged_at: "2026-08-10T17:30:00Z",
    merge_commit_sha: finalMerge,
  });
  f.github.deployments.set("sceneworks:5500", {
    sha: finalMerge,
    url: "https://api.github.test/deployments/5500",
  });
  recordEvidence(
    f.statePath,
    {
      repo: "sceneworks",
      number: finalNumber,
      deploymentIds: [{ repo: "sceneworks", id: 5500 }],
    },
    { github: f.github, clock: laterClock },
  );
  const valid = loadState(f.statePath);
  assert.equal(valid.privilegedRuns[0].finalPrNumber, finalNumber);
  assert.equal(valid.deployments[0].finalPrNumber, finalNumber);

  const cases = [
    ["canonical exact keys", (value) => { value.pullRequests[0].unexpected = true; }, /canonical PR evidence/],
    ["canonical derived kind", (value) => { value.pullRequests[0].kind = "story"; }, /kind or story conflicts/],
    ["adopted derived story", (value) => { value.adoptedPullRequests[0].story = null; }, /kind, story, or feature binding/],
    ["adopted queue binding", (value) => { value.adoptedPullRequests[0].queue.headSha = "8".repeat(40); }, /adopted PR evidence/],
    ["global PR identity", (value) => {
      value.adoptedPullRequests[0].number = finalNumber;
      value.adoptedPullRequests[0].url = `https://github.com/SceneWorks/SceneWorks/pull/${finalNumber}`;
    }, /duplicate canonical\/adopted PR evidence/],
    ["final pointer linkage", (value) => {
      value.finalPullRequests.sceneworks.number += 1;
      value.finalPullRequests.sceneworks.url =
        `https://github.com/SceneWorks/SceneWorks/pull/${value.finalPullRequests.sceneworks.number}`;
    }, /does not link its canonical receipt/],
    ["runtime exact keys", (value) => { value.privilegedRuns[0].unexpected = true; }, /runtime evidence/],
    ["runtime final linkage", (value) => { value.privilegedRuns[0].finalPrNumber += 1; }, /runtime evidence/],
    ["runtime uniqueness", (value) => { value.privilegedRuns.push(structuredClone(value.privilegedRuns[0])); }, /runtime evidence/],
    ["deployment exact keys", (value) => { value.deployments[0].unexpected = true; }, /deployment evidence/],
    ["deployment final linkage", (value) => { value.deployments[0].finalPrNumber += 1; }, /deployment evidence/],
    ["deployment uniqueness", (value) => { value.deployments.push(structuredClone(value.deployments[0])); }, /deployment evidence/],
  ];
  for (const [name, mutate, pattern] of cases) {
    await t.test(name, () => {
      const corrupted = structuredClone(valid);
      mutate(corrupted);
      writeJsonAtomic(f.statePath, corrupted);
      assert.throws(() => loadState(f.statePath), pattern);
      writeJsonAtomic(f.statePath, valid);
    });
  }
});

test("ruleset discovery exhausts every GitHub page and fails closed on repeated pages", async (t) => {
  const pageOne = (f, repo, idBase) => {
    const rulesets = structuredClone(f.github.ruleSets[repo]);
    while (rulesets.length < 100) {
      const id = idBase + rulesets.length;
      rulesets.push({
        id,
        name: `Unrelated policy ${id}`,
        target: "branch",
        enforcement: "active",
        bypass_actors: [],
        conditions: { ref_name: { include: [`refs/heads/unrelated-${id}`], exclude: [] } },
        rules: [],
      });
    }
    return rulesets;
  };

  await t.test("page-two duplicate wildcard blocks planning", (t) => {
    const f = fixture(t);
    const firstPage = pageOne(f, "sceneworks", 10000);
    const duplicate = structuredClone(firstPage[0]);
    duplicate.id = 19999;
    const { client, process } = paginatedRulesetClient({ sceneworks: [firstPage, [duplicate]] });
    const original = f.github.rulesets.bind(f.github);
    f.github.rulesets = (slug) => slug === REPOSITORIES.sceneworks.slug
      ? client.rulesets(slug)
      : original(slug);
    assert.throws(() => planFixture(f), /exactly two wildcard feature policies/);
    const listCall = process.calls.find((args) => args[1].includes("includes_parents=false"));
    assert.ok(listCall.includes("--paginate"));
    assert.ok(listCall.includes("--slurp"));
    assert.ok(process.calls.some((args) => args[1].endsWith(`/rulesets/${duplicate.id}`)));
  });

  await t.test("page-two unowned disabled policies in both repositories block planning", (t) => {
    const f = fixture(t);
    const pagesByRepo = {};
    for (const [index, repo] of Object.keys(REPOSITORIES).entries()) {
      const firstPage = pageOne(f, repo, 20000 + index * 1000);
      const staged = {
        id: 29990 + index,
        ...featurePolicyPayloads(repo, "feature/sc-18304-pipeline-flexibility-mlx-perf", "sc-18304")
          .stagedExactQueue,
      };
      pagesByRepo[repo] = [firstPage, [staged]];
    }
    const { client } = paginatedRulesetClient(pagesByRepo);
    f.github.rulesets = (slug) => client.rulesets(slug);
    assert.throws(() => planFixture(f), /disabled exact queue exists outside a recorded/);
  });

  await t.test("repeated ruleset id across pages is rejected", (t) => {
    const f = fixture(t);
    const repeated = structuredClone(f.github.ruleSets.sceneworks[0]);
    const { client } = paginatedRulesetClient({ sceneworks: [[repeated], [repeated]] });
    assert.throws(
      () => client.rulesets(REPOSITORIES.sceneworks.slug),
      /invalid or repeated ruleset id/,
    );
  });
});

test("live pull-request discovery exhausts every GitHub page and rejects repeated evidence", () => {
  const encode = (value) => Buffer.from(JSON.stringify({
    state: "closed",
    html_url: `https://github.test/pull/${value.number}`,
    head: { ref: `story/sc-${value.number}-epic-18304-fixture` },
    base: { ref: "feature/sc-18304-pipeline-flexibility-mlx-perf" },
    ...value,
  })).toString("base64");
  let records = [encode({ number: 1 }), encode({ number: 101 })];
  const process = new FakeGhProcess((args) => {
    assert.equal(
      args[1],
      "repos/SceneWorks/SceneWorks/pulls?state=all&per_page=100&sort=created&direction=asc&base=feature%2Fsc-18304-pipeline-flexibility-mlx-perf",
    );
    assert.ok(args.includes("--paginate"));
    assert.ok(args.includes("--jq"));
    assert.equal(args.includes("--slurp"), false);
    return records.join("\n");
  });
  const github = new GitHubClient(process);
  assert.deepEqual(
    github.pullRequests(REPOSITORIES.sceneworks.slug, {
      state: "all",
      base: "feature/sc-18304-pipeline-flexibility-mlx-perf",
    }).map(({ number }) => number),
    [1, 101],
  );
  records = [encode({ number: 1 }), encode({ number: 1 })];
  assert.throws(
    () => github.pullRequests(REPOSITORIES.sceneworks.slug, {
      state: "all",
      base: "feature/sc-18304-pipeline-flexibility-mlx-perf",
    }),
    /repeated pull-request number/,
  );
});

test("audit discovers page-two epic claims in either ref and rejects alternate or malformed topology", (t) => {
  const f = fixture(t);
  const state = planFixture(f);
  const candidates = [
    {
      number: 9101,
      head: "feature/alternate-sc-18304",
      base: "main",
    },
    {
      number: 9102,
      head: "fix/ordinary-looking-head",
      base: "feature/alternate-sc-18304",
    },
    {
      number: 9103,
      head: "story/sc-18419-epic-18304_bad-shape",
      base: state.featureBranch,
    },
  ];
  for (const candidate of candidates) {
    f.github.prs.set(`sceneworks:${candidate.number}`, {
      html_url: `https://github.com/SceneWorks/SceneWorks/pull/${candidate.number}`,
      state: "open",
      head: {
        ref: candidate.head,
        sha: f.sceneWorksMain,
        repo: { full_name: REPOSITORIES.sceneworks.slug },
      },
      base: { ref: candidate.base, repo: { full_name: REPOSITORIES.sceneworks.slug } },
    });
  }
  const originalPullRequests = f.github.pullRequests.bind(f.github);
  f.github.pullRequests = (slug, query = {}) => {
    if (slug !== REPOSITORIES.sceneworks.slug || query.base || query.head) {
      return originalPullRequests(slug, query);
    }
    const unrelated = Array.from({ length: 100 }, (_, index) => ({
      number: 8000 + index,
      state: "closed",
      head: { ref: `fix/unrelated-${index}` },
      base: { ref: "main" },
    }));
    return [
      ...unrelated,
      ...candidates.map(({ number }) => f.github.pullRequest(slug, number)),
    ];
  };
  const report = auditState(
    f.statePath,
    f.reportPath,
    { github: f.github, shortcut: f.shortcut, clock },
  );
  assert.deepEqual(
    report.livePullRequests.open.map(({ number }) => number),
    candidates.map(({ number }) => number),
  );
  assert.deepEqual(
    report.livePullRequests.liveTopologyErrors.map(({ number }) => number),
    candidates.map(({ number }) => number),
  );
  assert.equal(report.gates.noLiveTrainTopologyErrors, false);
});

test("audit requires the exact live Shortcut workflow map frozen in state", (t) => {
  const f = fixture(t);
  planFixture(f);
  f.shortcut.workflowValues[0].states.find(({ id }) => id === 30).name = "Completed";
  const report = auditState(
    f.statePath,
    f.reportPath,
    { github: f.github, shortcut: f.shortcut, clock },
  );
  assert.equal(report.gates.shortcutInventoryMatches, true);
  assert.equal(report.gates.shortcutStoriesDone, true);
  assert.equal(report.gates.shortcutWorkflowMapMatchesFrozen, false);
  assert.equal(report.ok, false);
});

test("state lock excludes a concurrent process and safely reclaims a dead same-host owner", async (t) => {
  const f = fixture(t);
  planFixture(f);
  const readyPath = join(f.root, "lock-ready");
  const releasePath = join(f.root, "lock-release");
  const moduleUrl = new URL("./lib/feature-epic.mjs", import.meta.url).href;
  const childCode = `
    import { existsSync, writeFileSync } from "node:fs";
    import { withStateLock } from ${JSON.stringify(moduleUrl)};
    const [, statePath, readyPath, releasePath] = process.argv;
    withStateLock(statePath, { operation: "fixture-child" }, () => {
      writeFileSync(readyPath, "ready\\n");
      const buffer = new Int32Array(new SharedArrayBuffer(4));
      while (!existsSync(releasePath)) Atomics.wait(buffer, 0, 0, 20);
    });
  `;
  const child = spawn(
    process.execPath,
    ["--input-type=module", "--eval", childCode, f.statePath, readyPath, releasePath],
    { stdio: ["ignore", "ignore", "pipe"] },
  );
  t.after(() => {
    if (existsSync(f.root)) writeFileSync(releasePath, "release\n");
    if (child.exitCode === null) child.kill("SIGKILL");
  });
  await waitForPath(readyPath);
  assert.throws(
    () => bootstrap(f.statePath, { apply: true }, { github: f.github, git: f.git, clock }),
    /state is locked by pid/,
  );
  assert.deepEqual(f.github.mutations, []);
  writeFileSync(releasePath, "release\n");
  await waitForChild(child);
  assert.equal(existsSync(stateLockPath(f.statePath)), false);

  const abandoned = stateLockPath(f.statePath);
  mkdirSync(abandoned, { mode: 0o700 });
  writeJsonAtomic(join(abandoned, "owner.json"), {
    schemaVersion: 1,
    token: "dead-owner",
    pid: 999999,
    hostname: hostname(),
    operation: "crashed-command",
    startedAt: EPOCH,
  });
  let entered = false;
  withStateLock(f.statePath, { operation: "recovery", clock }, () => { entered = true; });
  assert.equal(entered, true);
  assert.equal(existsSync(abandoned), false);

  mkdirSync(abandoned, { mode: 0o700 });
  writeJsonAtomic(join(abandoned, "owner.json"), {
    schemaVersion: 1,
    token: "foreign-owner",
    pid: 42,
    hostname: "another-host.example.test",
    operation: "unknown-remote-command",
    startedAt: "2026-08-10T11:00:00Z",
  });
  assert.throws(
    () => withStateLock(f.statePath, { operation: "unsafe-recovery", clock }, () => {}),
    /state is locked by pid 42/,
  );
  assert.throws(
    () => withStateLock(
      f.statePath,
      { operation: "too-short-recovery", clock, staleAfterMs: 299_999 },
      () => {},
    ),
    /at least 300000 milliseconds/,
  );
  withStateLock(
    f.statePath,
    { operation: "authorized-stale-recovery", clock, staleAfterMs: 300_000 },
    () => {},
  );
  assert.equal(existsSync(abandoned), false);
});

test("bootstrap stages disabled queues, creates refs, activates exact ids, then exposes clean clones", (t) => {
  const f = fixture(t);
  const planned = planFixture(f);
  const state = bootstrap(f.statePath, { apply: true }, { github: f.github, git: f.git, clock });
  assert.equal(state.phase, "ready");
  for (const key of Object.keys(REPOSITORIES)) {
    assert.equal(f.github.featureSha(REPOSITORIES[key].slug, planned.featureBranch), planned.repositories[key].plannedMainSha);
    assert.equal(state.repositories[key].workspace.status, "ready");
    assert.equal(f.git.inspectWorkspace(state.repositories[key].workspace.path).dirty, false);
  }
  assert.deepEqual(state.transaction.remoteMutations.map(({ action }) => action), ["create-ref", "create-ref"]);
  assert.deepEqual(state.transaction.policyMutations.map(({ action }) => action), [
    "create-ruleset",
    "create-ruleset",
    "activate-ruleset",
    "activate-ruleset",
  ]);
  assert.equal(Object.values(state.policies).every(({ exactQueue }) => exactQueue.status === "active"), true);
  assert.deepEqual(
    f.github.mutations.map(({ type }) => type),
    ["create-ruleset", "create-ruleset", "create-ref", "update-ruleset", "create-ref", "update-ruleset"],
    "both disabled policies precede per-repository ref creation, activation, and verification",
  );
  for (const key of Object.keys(REPOSITORIES)) {
    const audit = auditFeaturePolicies(f.github, key, state.featureBranch, state.epic);
    assert.equal(audit.exactQueue.enforcement, "active");
    assert.deepEqual(
      verifyEffectiveFeaturePolicies(f.github, key, state.featureBranch, audit).rulesetIds,
      [audit.wildcardBase.id, audit.deletionGuard.id, audit.exactQueue.id],
    );
  }
  const again = bootstrap(f.statePath, { apply: true }, { github: f.github, git: f.git, clock });
  assert.equal(again.transaction.remoteMutations.length, 2, "matching pair is idempotent");
});

test("bootstrap revalidates both planned main check matrices before its first mutation", async (t) => {
  for (const repo of Object.keys(REPOSITORIES)) {
    await t.test(repo, (t) => {
      const f = fixture(t);
      const planned = planFixture(f);
      const sha = planned.repositories[repo].plannedMainSha;
      const key = `${repo}:${sha}`;
      const runs = f.github.checks.get(key);
      f.github.checks.set(key, [
        trustedCheck(runs[0].name, runs[0].id + 1000, {
          conclusion: "failure",
          started_at: "2026-08-10T13:00:00Z",
          completed_at: "2026-08-10T13:00:00Z",
        }),
        ...runs,
      ]);
      f.github.checkQueries = [];
      assert.throws(
        () => bootstrap(f.statePath, { apply: true }, { github: f.github, git: f.git, clock }),
        /lack terminal trusted required checks/,
      );
      assert.deepEqual(f.github.checkQueries, [
        { repo: "sceneworks", sha: planned.repositories.sceneworks.plannedMainSha },
        { repo: "inference", sha: planned.repositories.inference.plannedMainSha },
      ]);
      assert.deepEqual(f.github.mutations, [], "check revalidation precedes remote mutations");
      const unchanged = loadState(f.statePath);
      assert.equal(unchanged.phase, "planned");
      assert.deepEqual(unchanged.transaction.policyMutations, []);
      assert.deepEqual(unchanged.transaction.remoteMutations, []);
    });
  }
  await t.test("a check flip during the final paginated ruleset detail read blocks mutation", (t) => {
    const f = fixture(t);
    const planned = planFixture(f);
    const pagesByRepo = Object.fromEntries(
      Object.keys(REPOSITORIES).map((repo) => [repo, [structuredClone(f.github.ruleSets[repo])]]),
    );
    const { client, process } = paginatedRulesetClient(pagesByRepo);
    const respond = process.responder;
    let detailReads = 0;
    process.responder = (args) => {
      const response = respond(args);
      if (/\/rulesets\/[1-9][0-9]*$/.test(args[1])) {
        detailReads += 1;
        if (detailReads === 4) {
          const key = `sceneworks:${planned.repositories.sceneworks.plannedMainSha}`;
          const runs = f.github.checks.get(key);
          f.github.checks.set(key, [
            ...runs,
            trustedCheck(runs[0].name, runs[0].id + 1000, {
              conclusion: "failure",
              started_at: "2026-08-10T13:00:00Z",
              completed_at: "2026-08-10T13:00:00Z",
            }),
          ]);
        }
      }
      return response;
    };
    f.github.rulesets = (slug) => client.rulesets(slug);
    f.github.checkQueries = [];
    assert.throws(
      () => bootstrap(f.statePath, { apply: true }, { github: f.github, git: f.git, clock }),
      /lack terminal trusted required checks/,
    );
    assert.equal(detailReads, 4, "both repositories were exhaustively read before revalidation");
    assert.deepEqual(f.github.checkQueries, [
      { repo: "sceneworks", sha: planned.repositories.sceneworks.plannedMainSha },
      { repo: "inference", sha: planned.repositories.inference.plannedMainSha },
    ]);
    assert.deepEqual(f.github.mutations, []);
    const unchanged = loadState(f.statePath);
    assert.equal(unchanged.phase, "planned");
    assert.deepEqual(unchanged.transaction.policyMutations, []);
    assert.deepEqual(unchanged.transaction.remoteMutations, []);
  });
});

test("ruleset failure persists a policy-only partial and recover completes policy before refs", (t) => {
  const f = fixture(t);
  const planned = planFixture(f);
  f.github.failRulesetFor = "inference";
  assert.throws(
    () => bootstrap(f.statePath, { apply: true }, { github: f.github, git: f.git, clock }),
    /injected ruleset failure/,
  );
  const partial = loadState(f.statePath);
  assert.equal(partial.phase, "recovery-required");
  assert.ok(partial.policies.sceneworks.exactQueue.id);
  assert.equal(partial.policies.inference.exactQueue.id, null);
  assert.equal(f.github.featureSha(REPOSITORIES.sceneworks.slug, partial.featureBranch), null);
  assert.equal(f.github.featureSha(REPOSITORIES.inference.slug, partial.featureBranch), null);
  advanceMain(f.origins.sceneworks, "policy-failure-main.txt");
  advanceMain(f.origins.inference, "policy-failure-main.txt");
  f.github.failRulesetFor = null;
  const recovered = recover(f.statePath, { complete: true }, { github: f.github, git: f.git, clock });
  assert.equal(recovered.phase, "ready");
  assert.equal(Object.values(recovered.policies).every(({ exactQueue }) => exactQueue.status === "active"), true);
  assert.equal(Object.values(recovered.repositories).every(({ remoteRef }) => Boolean(remoteRef)), true);
  for (const key of Object.keys(REPOSITORIES)) {
    assert.equal(recovered.repositories[key].remoteRef, planned.repositories[key].plannedMainSha);
  }
  assert.deepEqual(
    recovered.transaction.policyMutations.map(({ action, status }) => `${action}:${status}`),
    [
      "create-ruleset:created",
      "create-ruleset:failed",
      "recover-create-ruleset:created",
      "recover-activate-ruleset:created",
      "recover-activate-ruleset:created",
    ],
  );
});

test("the first exact-queue creation failure leaves a journal-owned recoverable transaction", (t) => {
  const f = fixture(t);
  const planned = planFixture(f);
  f.github.failRulesetFor = "sceneworks";
  assert.throws(
    () => bootstrap(f.statePath, { apply: true }, { github: f.github, git: f.git, clock }),
    /injected ruleset failure/,
  );
  const partial = loadState(f.statePath);
  assert.equal(partial.phase, "recovery-required");
  assert.deepEqual(
    partial.transaction.policyMutations.map(({ action, repo, status, id }) => ({ action, repo, status, id })),
    [{ action: "create-ruleset", repo: "sceneworks", status: "failed", id: null }],
  );
  assert.equal(Object.values(partial.repositories).every(({ remoteRef }) => remoteRef === null), true);

  f.github.failRulesetFor = null;
  const recovered = recover(f.statePath, { complete: true }, { github: f.github, git: f.git, clock });
  assert.equal(recovered.phase, "ready");
  assert.equal(Object.values(recovered.policies).every(({ exactQueue }) => exactQueue.status === "active"), true);
  for (const key of Object.keys(REPOSITORIES)) {
    assert.equal(recovered.repositories[key].remoteRef, planned.repositories[key].plannedMainSha);
  }
  assert.deepEqual(
    recovered.transaction.policyMutations.map(({ action, repo, status }) => `${action}:${repo}:${status}`),
    [
      "create-ruleset:sceneworks:failed",
      "recover-create-ruleset:sceneworks:created",
      "recover-create-ruleset:inference:created",
      "recover-activate-ruleset:sceneworks:created",
      "recover-activate-ruleset:inference:created",
    ],
  );
});

test("bootstrap never persists a staging phase before its first policy intent", () => {
  const automation = readFileSync(new URL("./lib/feature-epic.mjs", import.meta.url), "utf8");
  assert.doesNotMatch(
    automation,
    /state\.phase = "staging-policies";\s*saveState\(statePath, state, clock\);/,
  );
});

test("recover reconciles an ambiguous exact-queue creation without creating a duplicate", (t) => {
  const f = fixture(t);
  planFixture(f);
  f.github.failAfterRulesetFor = "inference";
  assert.throws(
    () => bootstrap(f.statePath, { apply: true }, { github: f.github, git: f.git, clock }),
    /ambiguous ruleset failure/,
  );
  const before = f.github.ruleSets.inference.length;
  f.github.failAfterRulesetFor = null;
  const recovered = recover(f.statePath, { complete: true }, { github: f.github, git: f.git, clock });
  assert.equal(recovered.phase, "ready");
  assert.equal(f.github.ruleSets.inference.length, before);
  assert.equal(
    recovered.transaction.policyMutations.find(
      ({ repo, action }) => repo === "inference" && action === "create-ruleset",
    ).status,
    "reconciled",
  );
});

test("activation failure leaves refs protected by disabled exact policies and recover activates recorded ids", (t) => {
  const f = fixture(t);
  planFixture(f);
  f.github.failUpdateFor = "inference";
  assert.throws(
    () => bootstrap(f.statePath, { apply: true }, { github: f.github, git: f.git, clock }),
    /injected ruleset update failure/,
  );
  const interrupted = loadState(f.statePath);
  assert.equal(interrupted.phase, "recovery-required");
  assert.equal(interrupted.policies.sceneworks.exactQueue.status, "active");
  assert.equal(interrupted.policies.inference.exactQueue.status, "failed");
  assert.equal(Object.values(interrupted.repositories).every(({ remoteRef }) => Boolean(remoteRef)), true);
  assert.equal(Object.values(interrupted.repositories).every(({ workspace }) => workspace.status === "pending"), true);
  const ids = Object.fromEntries(
    Object.entries(interrupted.policies).map(([key, { exactQueue }]) => [key, exactQueue.id]),
  );
  f.github.failUpdateFor = null;
  const recovered = recover(f.statePath, { complete: true }, { github: f.github, git: f.git, clock });
  assert.equal(recovered.phase, "ready");
  assert.deepEqual(
    Object.fromEntries(Object.entries(recovered.policies).map(([key, { exactQueue }]) => [key, exactQueue.id])),
    ids,
  );
  assert.equal(Object.values(recovered.policies).every(({ exactQueue }) => exactQueue.status === "active"), true);
  assert.equal(
    recovered.transaction.policyMutations.filter(({ action }) => action === "recover-activate-ruleset").length,
    1,
  );
});

test("recover reconciles an ambiguous PUT activation and never updates an unknown ruleset", async (t) => {
  await t.test("ambiguous PUT", (t) => {
    const f = fixture(t);
    const planned = planFixture(f);
    f.github.failAfterUpdateFor = "inference";
    assert.throws(
      () => bootstrap(f.statePath, { apply: true }, { github: f.github, git: f.git, clock }),
      /ambiguous ruleset update failure/,
    );
    const updatesBefore = f.github.mutations.filter(({ type }) => type === "update-ruleset").length;
    advanceMain(f.origins.sceneworks, "activation-main.txt");
    advanceMain(f.origins.inference, "activation-main.txt");
    f.github.failAfterUpdateFor = null;
    const recovered = recover(f.statePath, { complete: true }, { github: f.github, git: f.git, clock });
    assert.equal(recovered.phase, "ready");
    for (const key of Object.keys(REPOSITORIES)) {
      assert.equal(recovered.repositories[key].remoteRef, planned.repositories[key].plannedMainSha);
    }
    assert.equal(
      f.github.mutations.filter(({ type }) => type === "update-ruleset").length,
      updatesBefore,
      "a live active payload reconciles the ambiguous PUT without another update",
    );
    assert.equal(
      recovered.transaction.policyMutations.find(
        ({ repo, action }) => repo === "inference" && action === "activate-ruleset",
      ).status,
      "reconciled",
    );
  });

  await t.test("recorded id drift", (t) => {
    const f = fixture(t);
    planFixture(f);
    f.github.failUpdateFor = "sceneworks";
    assert.throws(
      () => bootstrap(f.statePath, { apply: true }, { github: f.github, git: f.git, clock }),
      /injected ruleset update failure/,
    );
    const interrupted = loadState(f.statePath);
    const live = f.github.ruleSets.sceneworks.find(
      ({ name }) => name === interrupted.policies.sceneworks.exactQueue.name,
    );
    live.id += 10000;
    f.github.failUpdateFor = null;
    assert.throws(
      () => recover(f.statePath, { complete: true }, { github: f.github, git: f.git, clock }),
      /not owned by this transaction/,
    );
    assert.equal(
      f.github.mutations.filter(({ type }) => type === "update-ruleset").length,
      1,
      "recovery did not PUT an unknown ruleset id",
    );
  });
});

test("an incomplete effective-rule aggregate blocks clone exposure until recovery verifies it", (t) => {
  const f = fixture(t);
  planFixture(f);
  const branchRules = f.github.branchRules.bind(f.github);
  f.github.branchRules = (slug, branch) =>
    branchRules(slug, branch).filter(({ type }) => type !== "merge_queue");
  assert.throws(
    () => bootstrap(f.statePath, { apply: true }, { github: f.github, git: f.git, clock }),
    /effective feature rules are missing/,
  );
  const interrupted = loadState(f.statePath);
  assert.equal(interrupted.phase, "recovery-required");
  assert.equal(Object.values(interrupted.repositories).every(({ workspace }) => workspace.status === "pending"), true);
  f.github.branchRules = branchRules;
  const recovered = recover(f.statePath, { complete: true }, { github: f.github, git: f.git, clock });
  assert.equal(recovered.phase, "ready");
  assert.equal(Object.values(recovered.repositories).every(({ workspace }) => workspace.status === "ready"), true);
});

test("recover temporarily disables only recorded active exact ids before creating missing refs", (t) => {
  const f = fixture(t);
  const interrupted = activePolicyRecoveryState(f, { sceneWorksRulesetId: 3621880479 });
  const before = f.github.mutations.length;
  const recovered = recover(f.statePath, { complete: true }, { github: f.github, git: f.git, clock });
  assert.equal(recovered.phase, "ready");
  assert.equal(recovered.policies.sceneworks.exactQueue.id, 3621880479);
  assert.deepEqual(
    f.github.mutations.slice(before).map(({ type, repo, enforcement }) =>
      `${type}:${repo}:${enforcement ?? "create"}`),
    [
      "update-ruleset:sceneworks:disabled",
      "update-ruleset:inference:disabled",
      "create-ref:sceneworks:create",
      "update-ruleset:sceneworks:active",
      "create-ref:inference:create",
      "update-ruleset:inference:active",
    ],
  );
  assert.deepEqual(
    recovered.transaction.policyMutations.slice(-4).map(({ action, status }) => `${action}:${status}`),
    [
      "recover-disable-ruleset:created",
      "recover-disable-ruleset:created",
      "recover-activate-ruleset:created",
      "recover-activate-ruleset:created",
    ],
  );
  for (const key of Object.keys(REPOSITORIES)) {
    assert.equal(
      f.github.featureSha(REPOSITORIES[key].slug, interrupted.featureBranch),
      interrupted.repositories[key].plannedMainSha,
    );
    assert.equal(recovered.policies[key].exactQueue.status, "active");
  }
});

test("active-queue recovery reconciles ambiguous disable and refuses an unknown id", async (t) => {
  await t.test("ambiguous disable PUT", (t) => {
    const f = fixture(t);
    const planned = activePolicyRecoveryState(f);
    f.github.failAfterUpdateFor = "sceneworks";
    assert.throws(
      () => recover(f.statePath, { complete: true }, { github: f.github, git: f.git, clock }),
      /ambiguous ruleset update failure/,
    );
    const disabledBefore = f.github.mutations.filter(
      ({ type, repo, enforcement }) =>
        type === "update-ruleset" && repo === "sceneworks" && enforcement === "disabled",
    ).length;
    assert.equal(disabledBefore, 1);
    advanceMain(f.origins.sceneworks, "disable-main.txt");
    advanceMain(f.origins.inference, "disable-main.txt");
    f.github.failAfterUpdateFor = null;
    const recovered = recover(f.statePath, { complete: true }, { github: f.github, git: f.git, clock });
    assert.equal(recovered.phase, "ready");
    for (const key of Object.keys(REPOSITORIES)) {
      assert.equal(recovered.repositories[key].remoteRef, planned.repositories[key].plannedMainSha);
    }
    assert.equal(
      f.github.mutations.filter(
        ({ type, repo, enforcement }) =>
          type === "update-ruleset" && repo === "sceneworks" && enforcement === "disabled",
      ).length,
      1,
      "live disabled payload reconciled the ambiguous PUT without another disable",
    );
    assert.equal(
      recovered.transaction.policyMutations.find(
        ({ repo, action }) => repo === "sceneworks" && action === "recover-disable-ruleset",
      ).status,
      "reconciled",
    );
  });

  await t.test("known pre-call disable failure is retried", (t) => {
    const f = fixture(t);
    activePolicyRecoveryState(f);
    f.github.failUpdateFor = "sceneworks";
    assert.throws(
      () => recover(f.statePath, { complete: true }, { github: f.github, git: f.git, clock }),
      /ruleset update failure/,
    );
    assert.equal(
      auditFeaturePolicies(f.github, "sceneworks", loadState(f.statePath).featureBranch, "sc-18304")
        .exactQueue.enforcement,
      "active",
    );
    f.github.failUpdateFor = null;
    const recovered = recover(f.statePath, { complete: true }, { github: f.github, git: f.git, clock });
    assert.equal(recovered.phase, "ready");
    assert.equal(
      f.github.mutations.filter(
        ({ type, repo, enforcement }) =>
          type === "update-ruleset" && repo === "sceneworks" && enforcement === "disabled",
      ).length,
      2,
    );
  });

  await t.test("unknown active id", (t) => {
    const f = fixture(t);
    const state = activePolicyRecoveryState(f);
    const live = f.github.ruleSets.sceneworks.find(
      ({ name }) => name === state.policies.sceneworks.exactQueue.name,
    );
    live.id += 10000;
    const before = f.github.mutations.length;
    assert.throws(
      () => recover(f.statePath, { complete: true }, { github: f.github, git: f.git, clock }),
      /not owned by this transaction/,
    );
    assert.equal(f.github.mutations.length, before, "unknown id was never updated");
  });
});

test("bootstrap refuses stale main, pre-existing partial refs, divergent refs, and dirty owned clones", async (t) => {
  await t.test("stale main", (t) => {
    const f = fixture(t);
    planFixture(f);
    advanceMain(f.origins.sceneworks);
    assert.throws(
      () => bootstrap(f.statePath, { apply: true }, { github: f.github, git: f.git, clock }),
      /main moved/,
    );
  });
  await t.test("pre-existing partial", (t) => {
    const f = fixture(t);
    const state = planFixture(f);
    f.github.createFeatureRef(REPOSITORIES.sceneworks.slug, state.featureBranch, state.repositories.sceneworks.plannedMainSha);
    assert.throws(
      () => bootstrap(f.statePath, { apply: true }, { github: f.github, git: f.git, clock }),
      /exactly one mirrored/,
    );
  });
  await t.test("divergent feature ref", (t) => {
    const f = fixture(t);
    const state = planFixture(f);
    const newer = sideCommit(f.origins.sceneworks);
    run("git", ["update-ref", `refs/heads/${state.featureBranch}`, newer], { cwd: f.origins.sceneworks });
    assert.throws(
      () => bootstrap(f.statePath, { apply: true }, { github: f.github, git: f.git, clock }),
      /not planned/,
    );
  });
  await t.test("dirty state-owned clone", (t) => {
    const f = fixture(t);
    planFixture(f);
    const state = bootstrap(f.statePath, { apply: true }, { github: f.github, git: f.git, clock });
    writeFileSync(join(state.repositories.sceneworks.workspace.path, "dirty"), "x");
    assert.throws(
      () => bootstrap(f.statePath, { apply: true }, { github: f.github, git: f.git, clock }),
      /dirty or conflicts/,
    );
  });
});

test("remote failure persists the exact partial transaction and recover completes only its missing half", (t) => {
  const f = fixture(t);
  const planned = planFixture(f);
  f.github.failCreateFor = "inference";
  assert.throws(
    () => bootstrap(f.statePath, { apply: true }, { github: f.github, git: f.git, clock }),
    /injected create-ref failure/,
  );
  const partial = loadState(f.statePath);
  assert.equal(partial.phase, "recovery-required");
  assert.equal(partial.repositories.sceneworks.remoteRef, planned.repositories.sceneworks.plannedMainSha);
  assert.equal(partial.repositories.inference.remoteRef, null);
  const unsafe = auditState(f.statePath, f.reportPath, { github: f.github, shortcut: f.shortcut, clock });
  assert.equal(unsafe.bootstrap.policies.sceneworks.unsafeFeatureRef, false);
  assert.equal(unsafe.bootstrap.policies.sceneworks.exactQueue.enforcement, "active");
  assert.equal(unsafe.gates.featurePoliciesVerified, false);
  assert.throws(
    () => bootstrap(f.statePath, { apply: true }, { github: f.github, git: f.git, clock }),
    /recover --complete/,
  );
  advanceMain(f.origins.sceneworks, "ref-failure-main.txt");
  advanceMain(f.origins.inference, "ref-failure-main.txt");
  f.github.checks.set(`sceneworks:${planned.repositories.sceneworks.plannedMainSha}`, []);
  f.github.checks.set(`inference:${planned.repositories.inference.plannedMainSha}`, []);
  f.github.checkQueries = [];
  f.github.failCreateFor = null;
  const recovered = recover(f.statePath, { complete: true }, { github: f.github, git: f.git, clock });
  assert.equal(recovered.phase, "ready");
  assert.deepEqual(f.github.checkQueries, [], "journaled recovery uses the frozen check preflight");
  for (const key of Object.keys(REPOSITORIES)) {
    assert.equal(recovered.repositories[key].remoteRef, planned.repositories[key].plannedMainSha);
  }
  assert.deepEqual(
    recovered.transaction.remoteMutations.map(({ action, status }) => `${action}:${status}`),
    ["create-ref:created", "create-ref:failed", "recover-create-ref:created"],
  );
});

test("recover reconciles an ambiguous create-ref that succeeded after its durable intent", (t) => {
  const f = fixture(t);
  planFixture(f);
  f.github.failAfterCreateFor = "inference";
  assert.throws(
    () => bootstrap(f.statePath, { apply: true }, { github: f.github, git: f.git, clock }),
    /ambiguous create-ref failure/,
  );
  const interrupted = loadState(f.statePath);
  assert.equal(interrupted.phase, "recovery-required");
  assert.equal(interrupted.repositories.inference.remoteRef, null);
  assert.equal(
    f.github.featureSha(REPOSITORIES.inference.slug, interrupted.featureBranch),
    interrupted.repositories.inference.plannedMainSha,
  );
  f.github.failAfterCreateFor = null;
  const recovered = recover(f.statePath, { complete: true }, { github: f.github, git: f.git, clock });
  assert.equal(recovered.phase, "ready");
  assert.equal(recovered.repositories.inference.remoteRef, recovered.repositories.inference.plannedMainSha);
  assert.equal(recovered.transaction.remoteMutations.at(-1).status, "reconciled");
  assert.equal(recovered.transaction.remoteMutations.length, 2, "recovery did not issue another create-ref");
});

test("clone failure leaves truthful state and recover safely resumes without touching matching refs", (t) => {
  const f = fixture(t);
  const planned = planFixture(f);
  f.git.failCloneFor = "inference";
  assert.throws(
    () => bootstrap(f.statePath, { apply: true }, { github: f.github, git: f.git, clock }),
    /injected clone failure/,
  );
  const failed = loadState(f.statePath);
  assert.equal(failed.phase, "recovery-required");
  assert.equal(failed.repositories.sceneworks.workspace.status, "ready");
  assert.equal(failed.repositories.inference.workspace.status, "failed");
  assert.equal(failed.repositories.inference.workspace.failedStagingPaths.length, 1);
  const mutations = failed.transaction.remoteMutations.length;
  advanceMain(f.origins.sceneworks, "clone-failure-main.txt");
  advanceMain(f.origins.inference, "clone-failure-main.txt");
  const recovered = recover(f.statePath, { complete: true }, { github: f.github, git: f.git, clock });
  assert.equal(recovered.phase, "ready");
  assert.equal(recovered.transaction.remoteMutations.length, mutations);
  for (const key of Object.keys(REPOSITORIES)) {
    assert.equal(recovered.repositories[key].remoteRef, planned.repositories[key].plannedMainSha);
  }
  assert.ok(existsSync(failed.repositories.inference.workspace.failedStagingPaths[0]), "failed clone is not silently deleted");
});

test("recover claims a verified clone renamed after its durable provisioning intent", (t) => {
  const f = fixture(t);
  const planned = planFixture(f);
  installStagedPolicyPair(f, planned);
  for (const key of Object.keys(REPOSITORIES)) {
    f.github.createFeatureRef(REPOSITORIES[key].slug, planned.featureBranch, planned.repositories[key].plannedMainSha);
    planned.repositories[key].remoteRef = planned.repositories[key].plannedMainSha;
    planned.transaction.remoteMutations.push({
      action: "create-ref",
      repo: key,
      ref: `refs/heads/${planned.featureBranch}`,
      sha: planned.repositories[key].plannedMainSha,
      at: EPOCH,
      status: "created",
    });
  }
  activatePolicyPair(f, planned);
  const workspace = planned.repositories.sceneworks.workspace;
  workspace.status = "provisioning";
  workspace.activeStagingPath = `${workspace.path}.bootstrap-1`;
  f.git.cloneFeature(
    REPOSITORIES.sceneworks,
    planned.featureBranch,
    workspace.path,
    workspace.activeStagingPath,
  );
  planned.phase = "recovery-required";
  planned.transaction.recoveryReason = "simulated process stop after atomic clone rename";
  writeJsonAtomic(f.statePath, planned);
  const recovered = recover(f.statePath, { complete: true }, { github: f.github, git: f.git, clock });
  assert.equal(recovered.repositories.sceneworks.workspace.status, "ready");
  assert.equal(recovered.repositories.sceneworks.workspace.activeStagingPath, null);
  assert.equal(recovered.repositories.inference.workspace.status, "ready");
  assert.equal(recovered.transaction.remoteMutations.length, 2);
});

test("recover refuses a plan with no transaction or an unrecorded existing ref", async (t) => {
  await t.test("planned state", (t) => {
    const f = fixture(t);
    planFixture(f);
    assert.throws(
      () => recover(f.statePath, { complete: true }, { github: f.github, git: f.git, clock }),
      /planned state has no partial transaction/,
    );
  });
  await t.test("unrecorded half", (t) => {
    const f = fixture(t);
    const state = planFixture(f);
    installStagedPolicyPair(f, state);
    state.phase = "recovery-required";
    state.transaction.recoveryReason = "simulated policy-stage interruption";
    writeJsonAtomic(f.statePath, state);
    f.github.createFeatureRef(REPOSITORIES.sceneworks.slug, state.featureBranch, state.repositories.sceneworks.plannedMainSha);
    assert.throws(
      () => recover(f.statePath, { complete: true }, { github: f.github, git: f.git, clock }),
      /not a state-recorded matching ref/,
    );
  });
});

test("state rejects mismatched mirrored names and conflicting recorded refs", async (t) => {
  await t.test("mismatched names", (t) => {
    const f = fixture(t);
    planFixture(f);
    const state = JSON.parse(readFileSync(f.statePath, "utf8"));
    state.repositories.inference.featureBranch = "feature/sc-18304-other";
    writeJsonAtomic(f.statePath, state);
    assert.throws(() => loadState(f.statePath), /mismatched mirrored branch/);
  });
  await t.test("recorded ref conflicts live", (t) => {
    const f = fixture(t);
    planFixture(f);
    const state = bootstrap(f.statePath, { apply: true }, { github: f.github, git: f.git, clock });
    const newer = sideCommit(f.origins.inference);
    run("git", ["update-ref", `refs/heads/${state.featureBranch}`, newer], { cwd: f.origins.inference });
    assert.throws(
      () => bootstrap(f.statePath, { apply: true }, { github: f.github, git: f.git, clock }),
      /not planned|conflicts with live/,
    );
  });
});

test("PR topology enforces the epic-bearing train and leaves ordinary PRs alone", () => {
  assert.deepEqual(
    classifyPullRequest(
      "story/sc-18419-epic-18304-feature-automation",
      "feature/sc-18304-feature-automation",
    ),
    { kind: "story", epic: "sc-18304", slug: "feature-automation", story: "sc-18419" },
  );
  assert.equal(
    classifyPullRequest("sync/sc-18304-main-2026-08-10", "feature/sc-18304-feature-automation").kind,
    "sync",
  );
  assert.equal(
    classifyPullRequest(
      "sync/sc-18304-main-2026-08-10-2",
      "feature/sc-18304-feature-automation",
    ).kind,
    "sync",
  );
  assert.equal(classifyPullRequest("feature/sc-18304-feature-automation", "main").kind, "final");
  assert.equal(classifyPullRequest("fix/an-ordinary-bug", "main").kind, "ordinary");
  assert.equal(classifyPullRequest("story/sc-18419-ordinary-fix", "main").kind, "ordinary");
  assert.equal(classifyPullRequest("story/sc-18419-epicfix", "main").kind, "ordinary");
  assert.throws(
    () => classifyPullRequest("story/sc-18419-epic-18304-feature-automation", "main"),
    /must target/,
  );
  assert.throws(
    () => classifyPullRequest("feature/sc-18304-feature-automation", "feature/sc-1-other"),
    /main only/,
  );
  assert.throws(
    () => classifyPullRequest("feature/sc-18304-feature-automation", "release\/next"),
    /main only/,
  );
  assert.throws(
    () => classifyPullRequest("fix/no-epic-owner", "feature/sc-18304-feature-automation"),
    /epic-bearing story/,
  );
  assert.throws(() => classifyPullRequest("story/sc-18419-epic", "main"), /malformed feature-train/);
  assert.throws(() => classifyPullRequest("fix/no-owner", "feature/SC-18304-invalid"), /feature branch must/);
});

function prEvent(head, base, headRepo = REPOSITORIES.sceneworks.slug) {
  return {
    repository: { full_name: REPOSITORIES.sceneworks.slug },
    pull_request: {
      head: { ref: head, sha: "1".repeat(40), repo: { full_name: headRepo } },
      base: { ref: base, repo: { full_name: REPOSITORIES.sceneworks.slug } },
    },
  };
}

test("event policy rejects cross-repo train heads and validates queue target correspondence", () => {
  const canonical = "feature/sc-18304-feature-automation";
  const resolveFeature = (epic) => {
    assert.equal(epic, "sc-18304");
    return canonical;
  };
  assert.throws(
    () => evaluateEventTopology(
      "pull_request",
      prEvent(
        "story/sc-18419-epic-18304-feature-automation",
        "feature/sc-18304-feature-automation",
        "fork/SceneWorks",
      ),
    ),
    /same-repository/,
  );
  assert.equal(
    evaluateEventTopology(
      "pull_request",
      prEvent("story/sc-18419-epic-18304-feature-automation", canonical),
      REPOSITORIES.sceneworks.slug,
      resolveFeature,
    ).kind,
    "story",
  );
  assert.throws(
    () => evaluateEventTopology(
      "pull_request",
      prEvent(
        "story/sc-18419-epic-18304-different-slug",
        "feature/sc-18304-different-slug",
      ),
      REPOSITORIES.sceneworks.slug,
      resolveFeature,
    ),
    /not the unique live canonical feature branch/,
  );
  assert.equal(
    evaluateEventTopology(
      "pull_request",
      prEvent("story/sc-18419-ordinary-fix", "main", "fork/SceneWorks"),
      REPOSITORIES.sceneworks.slug,
      () => {
        throw new Error("ordinary fork unexpectedly resolved feature state");
      },
    ).kind,
    "ordinary",
  );
  assert.equal(
    evaluateEventTopology("merge_group", {
      merge_group: {
        base_ref: "refs/heads/main",
        head_ref: "refs/heads/gh-readonly-queue/main/pr-1-deadbeef",
        head_sha: "1".repeat(40),
      },
    }).validateFinalPin,
    true,
  );
  assert.throws(
    () => evaluateEventTopology("merge_group", {
      merge_group: {
        base_ref: "refs/heads/main",
        head_ref: "refs/heads/gh-readonly-queue/feature/sc-1-x/pr-1-deadbeef",
        head_sha: "1".repeat(40),
      },
    }),
    /does not correspond/,
  );
  assert.equal(
    evaluateEventTopology(
      "merge_group",
      {
        merge_group: {
          base_ref: `refs/heads/${canonical}`,
          head_ref: `refs/heads/gh-readonly-queue/${canonical}/pr-1-deadbeef`,
          head_sha: "1".repeat(40),
        },
      },
      REPOSITORIES.sceneworks.slug,
      resolveFeature,
    ).kind,
    "feature-merge-group",
  );
  const noncanonical = "feature/sc-18304-other";
  assert.throws(
    () => evaluateEventTopology(
      "merge_group",
      {
        merge_group: {
          base_ref: `refs/heads/${noncanonical}`,
          head_ref: `refs/heads/gh-readonly-queue/${noncanonical}/pr-2-deadbeef`,
          head_sha: "2".repeat(40),
        },
      },
      REPOSITORIES.sceneworks.slug,
      resolveFeature,
    ),
    /not the unique live canonical feature branch/,
  );
});

test("canonical feature resolution is fixed to origin and refuses zero, duplicate, or divergent candidates", () => {
  const root = "/tmp/canonical-feature-fixture";
  const branch = "feature/sc-18304-feature-automation";
  const calls = [];
  const process = {
    exec(command, args) {
      calls.push({ command, args });
      return `${"a".repeat(40)}\trefs/heads/${branch}`;
    },
  };
  assert.equal(resolveRemoteFeatureBranch("sc-18304", root, new GitRunner(process)), branch);
  assert.deepEqual(calls, [{
    command: "git",
    args: ["-C", root, "ls-remote", "--heads", "origin", "refs/heads/feature/sc-18304-*"],
  }]);

  const resolve = (output) => () => output;
  assert.throws(
    () => resolveRemoteFeatureBranch("sc-18304", root, { remoteFeatureRefs: resolve("") }),
    /no live feature branch/,
  );
  assert.throws(
    () => resolveRemoteFeatureBranch("sc-18304", root, {
      remoteFeatureRefs: resolve(
        `${"a".repeat(40)}\trefs/heads/${branch}\n${"b".repeat(40)}\trefs/heads/feature/sc-18304-other`,
      ),
    }),
    /multiple live feature branches/,
  );
  assert.throws(
    () => resolveRemoteFeatureBranch("sc-18304", root, {
      remoteFeatureRefs: resolve(
        `${"a".repeat(40)}\trefs/heads/${branch}\n${"b".repeat(40)}\trefs/heads/${branch}`,
      ),
    }),
    /divergent commits/,
  );
});

test("final CI policy checks every tracked Cargo pin and remote-main ancestry only at integration", (t) => {
  const f = fixture(t);
  const repoRoot = join(f.root, "ci-worktree");
  run("git", ["clone", f.origins.sceneworks, repoRoot]);
  f.github.createFeatureRef(
    REPOSITORIES.sceneworks.slug,
    "feature/sc-18304-feature-automation",
    f.sceneWorksMain,
  );
  const ordinary = enforceCiPolicy(
    { eventName: "pull_request", event: prEvent("fix/ordinary", "main"), repoRoot },
    { github: f.github, git: f.git },
  );
  assert.equal(ordinary.pin, null);
  const final = enforceCiPolicy(
    {
      eventName: "pull_request",
      event: prEvent("feature/sc-18304-feature-automation", "main"),
      repoRoot,
    },
    { github: f.github, git: f.git },
  );
  assert.equal(final.pin.sha, f.inferenceMain);
  writeFileSync(join(repoRoot, "Cargo.toml"), manifest("1".repeat(40)));
  assert.throws(
    () => enforceCiPolicy(
      { eventName: "push", event: { ref: "refs/heads/main", after: "2".repeat(40) }, repoRoot },
      { github: f.github, git: f.git },
    ),
    /not an ancestor/,
  );
});

test("record-pr separates pre-queue source-head runtime from post-merge deployment evidence", (t) => {
  const f = fixture(t);
  planFixture(f, { deploymentRequired: true });
  const head = f.sceneWorksMain;
  f.github.prs.set("sceneworks:19", {
    html_url: "https://github.com/SceneWorks/SceneWorks/pull/19",
    head: {
      ref: "story/sc-18419-epic-18304-pipeline-flexibility-mlx-perf",
      sha: head,
      repo: { full_name: REPOSITORIES.sceneworks.slug },
    },
    base: {
      ref: "feature/sc-18304-pipeline-flexibility-mlx-perf",
      repo: { full_name: REPOSITORIES.sceneworks.slug },
    },
  });
  f.github.runs.set("sceneworks:101", {
    head_sha: head,
    html_url: "https://github.com/SceneWorks/SceneWorks/actions/runs/101",
    path: ".github/workflows/windows-candle.yml",
    event: "workflow_dispatch",
    status: "completed",
    conclusion: "success",
  });
  const storyReceipt = recordEvidence(
    f.statePath,
    {
      repo: "sceneworks",
      number: 19,
    },
    { github: f.github, clock },
  );
  assert.equal(storyReceipt.story, "sc-18419");
  f.github.prs.set("sceneworks:20", {
    html_url: "https://github.com/SceneWorks/SceneWorks/pull/20",
    head: {
      ref: "feature/sc-18304-pipeline-flexibility-mlx-perf",
      sha: head,
      repo: { full_name: REPOSITORIES.sceneworks.slug },
    },
    base: { ref: "main", repo: { full_name: REPOSITORIES.sceneworks.slug } },
    merge_commit_sha: "e".repeat(40),
  });
  const finalReceipt = recordEvidence(
    f.statePath,
    {
      repo: "sceneworks",
      number: 20,
      runReceipts: [{ repo: "sceneworks", id: 101 }],
    },
    { github: f.github, clock },
  );
  assert.equal(finalReceipt.kind, "final");
  const state = loadState(f.statePath);
  assert.equal(state.pullRequests[0].headSha, head);
  assert.equal(state.finalPullRequests.sceneworks.number, 20);
  assert.equal(state.privilegedRuns[0].url, "https://github.com/SceneWorks/SceneWorks/actions/runs/101");
  assert.equal(state.privilegedRuns[0].evidencePhase, "pre-queue-source");
  f.github.timelines.set("sceneworks:20", [{ event: "added_to_merge_queue", created_at: EPOCH }]);
  assert.throws(
    () => recordEvidence(
      f.statePath,
      { repo: "sceneworks", number: 20, runReceipts: [{ repo: "sceneworks", id: 101 }] },
      { github: f.github, clock },
    ),
    /before merge-queue entry/,
  );
  f.github.deployments.set("sceneworks:201", {
    sha: "e".repeat(40),
    url: "https://api.example/deployments/201",
  });
  assert.throws(
    () => recordEvidence(
      f.statePath,
      { repo: "sceneworks", number: 20, deploymentIds: [{ repo: "sceneworks", id: 201 }] },
      { github: f.github, clock },
    ),
    /landed merge commit/,
  );
  const landed = "d".repeat(40);
  f.github.prs.get("sceneworks:20").merged_at = EPOCH;
  f.github.prs.get("sceneworks:20").merge_commit_sha = landed;
  f.github.deployments.set("sceneworks:201", {
    sha: landed,
    url: "https://api.example/deployments/201",
  });
  recordEvidence(
    f.statePath,
    { repo: "sceneworks", number: 20, deploymentIds: [{ repo: "sceneworks", id: 201 }] },
    { github: f.github, clock },
  );
  const completed = loadState(f.statePath);
  assert.equal(completed.deployments[0].sha, landed);
  assert.equal(completed.deployments[0].evidencePhase, "post-merge");
  assert.throws(
    () => recordEvidence(
      f.statePath,
      { repo: "sceneworks", number: 20, runReceipts: [{ repo: "sceneworks", id: 101 }] },
      { github: f.github, clock },
    ),
    /before merge-queue entry/,
  );
});

test("timeline pagination finds queue evidence after item 100 for recording and closeout", (t) => {
  const f = fixture(t);
  planFixture(f);
  const firstPage = Array.from({ length: 100 }, (_, index) => ({
    event: "commented",
    id: index + 1,
  }));
  const staleQueueEvent = {
    event: "added_to_merge_queue",
    id: 101,
    created_at: "2026-08-10T11:59:59Z",
  };
  const currentQueueEvent = {
    event: "added_to_merge_queue",
    id: 102,
    created_at: "2026-08-10T08:01:00.123456-04:00",
  };
  let secondPage = [staleQueueEvent];
  const process = new FakeGhProcess((args) => {
    assert.match(args[1], /\/issues\/(19|20)\/timeline\?per_page=100$/);
    return [firstPage, secondPage];
  });
  const client = new GitHubClient(process);
  f.github.timeline = (slug, number) => client.timeline(slug, number);
  const head = f.sceneWorksMain;
  f.github.prs.set("sceneworks:19", {
    html_url: "https://github.com/SceneWorks/SceneWorks/pull/19",
    head: {
      ref: "story/sc-18419-epic-18304-pipeline-flexibility-mlx-perf",
      sha: head,
      repo: { full_name: REPOSITORIES.sceneworks.slug },
    },
    base: {
      ref: "feature/sc-18304-pipeline-flexibility-mlx-perf",
      repo: { full_name: REPOSITORIES.sceneworks.slug },
    },
  });
  recordEvidence(f.statePath, { repo: "sceneworks", number: 19 }, { github: f.github, clock });
  const stale = auditState(f.statePath, f.reportPath, { github: f.github, shortcut: f.shortcut, clock });
  assert.equal(stale.pullRequests[0].queueEvidence, false);
  assert.equal(stale.pullRequests[0].queueEvidenceAt, null);
  assert.equal(stale.pullRequests[0].queueEvidenceVerifiable, true);

  secondPage = [{ event: "added_to_merge_queue", id: 102 }];
  const untimestamped = auditState(f.statePath, f.reportPath, { github: f.github, shortcut: f.shortcut, clock });
  assert.equal(untimestamped.pullRequests[0].queueEvidence, false);
  assert.equal(untimestamped.pullRequests[0].queueEvidenceAt, null);
  assert.equal(untimestamped.pullRequests[0].queueEvidenceVerifiable, false);

  secondPage = [{ event: "added_to_merge_queue", id: 102, created_at: "2027-02-31T12:00:00Z" }];
  const invalidCalendarDate = auditState(f.statePath, f.reportPath, { github: f.github, shortcut: f.shortcut, clock });
  assert.equal(invalidCalendarDate.pullRequests[0].queueEvidence, false);
  assert.equal(invalidCalendarDate.pullRequests[0].queueEvidenceAt, null);
  assert.equal(invalidCalendarDate.pullRequests[0].queueEvidenceVerifiable, false);

  secondPage = [{ ...currentQueueEvent, created_at: EPOCH }];
  const equal = auditState(f.statePath, f.reportPath, { github: f.github, shortcut: f.shortcut, clock });
  assert.equal(equal.pullRequests[0].queueEvidence, true);
  assert.equal(equal.pullRequests[0].queueEvidenceAt, EPOCH);

  secondPage = [staleQueueEvent, currentQueueEvent];
  const current = auditState(f.statePath, f.reportPath, { github: f.github, shortcut: f.shortcut, clock });
  assert.equal(current.pullRequests[0].queueEvidence, true);
  assert.equal(current.pullRequests[0].queueEvidenceAt, currentQueueEvent.created_at);

  f.github.prs.set("sceneworks:20", {
    html_url: "https://github.com/SceneWorks/SceneWorks/pull/20",
    head: {
      ref: "feature/sc-18304-pipeline-flexibility-mlx-perf",
      sha: head,
      repo: { full_name: REPOSITORIES.sceneworks.slug },
    },
    base: { ref: "main", repo: { full_name: REPOSITORIES.sceneworks.slug } },
  });
  f.github.runs.set("sceneworks:101", {
    head_sha: head,
    html_url: "https://github.com/SceneWorks/SceneWorks/actions/runs/101",
    path: ".github/workflows/windows-candle.yml",
    event: "workflow_dispatch",
    status: "completed",
    conclusion: "success",
  });
  secondPage = [currentQueueEvent];
  assert.throws(
    () => recordEvidence(
      f.statePath,
      { repo: "sceneworks", number: 20, runReceipts: [{ repo: "sceneworks", id: 101 }] },
      { github: f.github, clock },
    ),
    /before merge-queue entry/,
  );
  secondPage = [{ event: "added_to_merge_queue", id: 103 }];
  assert.throws(
    () => recordEvidence(
      f.statePath,
      { repo: "sceneworks", number: 20, runReceipts: [{ repo: "sceneworks", id: 101 }] },
      { github: f.github, clock },
    ),
    /before merge-queue entry/,
  );
  assert.ok(process.calls.every((args) => args.includes("--paginate") && args.includes("--slurp")));
});

test("audit writes a report and keeps merge, checks, queue, ancestry, pin, runtime, deployment, cleanup distinct", (t) => {
  const f = fixture(t);
  const state = planFixture(f, { stories: ["sc-18419"], deploymentRequired: true });
  // A deliberately incomplete state is useful here: every dimension must remain visible instead
  // of collapsing to one misleading Done boolean.
  const report = auditState(f.statePath, f.reportPath, { github: f.github, shortcut: f.shortcut, clock });
  assert.equal(report.ok, false);
  assert.equal(report.expectedStories["sc-18419"], false);
  assert.equal(report.finalPullRequests.sceneworks.merged, false);
  assert.equal(report.gates.allRecordedPrHeadsGreen, true);
  assert.equal(report.gates.allRecordedPrsHaveQueueEvidence, true);
  assert.equal(report.inferencePin.reachableFromInferenceMain, true);
  assert.equal(report.inferencePin.finalInferenceMergeCommit, null);
  assert.equal(report.inferencePin.matchesFinalInferenceMerge, false);
  assert.equal(report.gates.finalPinMatchesInferenceMerge, false);
  assert.equal(report.privilegedRuntime.every(({ recorded }) => !recorded), true);
  assert.equal(report.deployment.required, true);
  assert.equal(report.deployment.satisfied, false);
  assert.deepEqual(report.cleanup, {
    eligible: false,
    action: "report-only",
    refs: { sceneworks: null, inference: null },
    exactQueueRulesets: {
      sceneworks: {
        id: null,
        name: "Feature queue: sc-18304",
        digest: state.policies.sceneworks.exactQueue.digest,
      },
      inference: {
        id: null,
        name: "Feature queue: sc-18304",
        digest: state.policies.inference.exactQueue.digest,
      },
    },
    note: "This audit never deletes refs or opens a ruleset bypass window.",
  });
  assert.equal(JSON.parse(readFileSync(f.reportPath, "utf8")).featureBranch, state.featureBranch);
});

test("audit reaches cleanup eligibility only when every separately verified gate is green", (t) => {
  const f = fixture(t);
  const state = planFixture(f, { stories: ["sc-18419"], deploymentRequired: true });
  bootstrap(f.statePath, { apply: true }, { github: f.github, git: f.git, clock });
  const swHead = state.repositories.sceneworks.plannedMainSha;
  const inferenceHead = state.repositories.inference.plannedMainSha;
  const inferenceMerge = advanceMain(f.origins.inference, "landed-feature.txt");
  const swMerge = advanceMain(f.origins.sceneworks, "Cargo.toml", manifest(inferenceMerge));
  const makePr = (repo, number, head, base, sha, mergeSha = sha) => ({
    html_url: `https://github.com/${REPOSITORIES[repo].slug}/pull/${number}`,
    head: { ref: head, sha, repo: { full_name: REPOSITORIES[repo].slug } },
    base: { ref: base, repo: { full_name: REPOSITORIES[repo].slug } },
    merged_at: EPOCH,
    merge_commit_sha: mergeSha,
  });
  f.github.prs.set(
    "sceneworks:19",
    makePr(
      "sceneworks",
      19,
      "story/sc-18419-epic-18304-pipeline-flexibility-mlx-perf",
      state.featureBranch,
      swHead,
    ),
  );
  f.github.prs.set("sceneworks:20", {
    html_url: "https://github.com/SceneWorks/SceneWorks/pull/20",
    head: {
      ref: state.featureBranch,
      sha: swHead,
      repo: { full_name: REPOSITORIES.sceneworks.slug },
    },
    base: { ref: "main", repo: { full_name: REPOSITORIES.sceneworks.slug } },
    merge_commit_sha: "e".repeat(40),
  });
  f.github.prs.set(
    "inference:21",
    makePr("inference", 21, state.featureBranch, "main", inferenceHead, inferenceMerge),
  );
  for (const [repo, number, sha] of [
    ["sceneworks", 19, swHead],
    ["sceneworks", 20, swHead],
    ["inference", 21, inferenceHead],
  ]) {
    f.github.checks.set(
      `${repo}:${sha}`,
      state.requiredContexts[repo].map((name, index) =>
        trustedCheck(name, (repo === "sceneworks" ? 500 : 600) + index),
      ),
    );
    if (number !== 20) {
      f.github.timelines.set(`${repo}:${number}`, [{ event: "added_to_merge_queue", created_at: EPOCH }]);
    }
  }
  for (const [id, workflow] of [[101, "windows-candle.yml"], [102, "macos-mlx.yml"]]) {
    f.github.runs.set(`sceneworks:${id}`, {
      head_sha: swHead,
      html_url: `https://github.com/SceneWorks/SceneWorks/actions/runs/${id}`,
      path: `.github/workflows/${workflow}`,
      event: "workflow_dispatch",
      status: "completed",
      conclusion: "success",
    });
  }
  f.github.deployments.set("sceneworks:201", { sha: swMerge, url: "https://api.example/deployments/201" });
  f.github.statuses.set("sceneworks:201", [{ state: "success" }]);
  recordEvidence(f.statePath, { repo: "sceneworks", number: 19 }, { github: f.github, clock });
  recordEvidence(
    f.statePath,
    {
      repo: "sceneworks",
      number: 20,
      runReceipts: [{ repo: "sceneworks", id: 101 }, { repo: "sceneworks", id: 102 }],
    },
    { github: f.github, clock },
  );
  f.github.prs.get("sceneworks:20").merged_at = EPOCH;
  f.github.prs.get("sceneworks:20").merge_commit_sha = swMerge;
  f.github.timelines.set("sceneworks:20", [{ event: "added_to_merge_queue", created_at: EPOCH }]);
  recordEvidence(f.statePath, { repo: "inference", number: 21 }, { github: f.github, clock });
  recordEvidence(
    f.statePath,
    {
      repo: "sceneworks",
      number: 20,
      deploymentIds: [{ repo: "sceneworks", id: 201 }],
    },
    { github: f.github, clock },
  );
  const report = auditState(f.statePath, f.reportPath, { github: f.github, shortcut: f.shortcut, clock });
  assert.equal(report.ok, true);
  assert.equal(Object.values(report.gates).every(Boolean), true);
  assert.equal(report.cleanup.eligible, true);
  assert.equal(report.cleanup.action, "report-only");
  assert.equal(report.inferencePin.sha, inferenceMerge);
  assert.equal(report.inferencePin.finalInferenceMergeCommit, inferenceMerge);
  assert.equal(report.inferencePin.matchesFinalInferenceMerge, true);
  assert.equal(report.gates.finalPinMatchesInferenceMerge, true);

  f.github.prs.set("inference:98", {
    html_url: "https://github.com/SceneWorks/inference/pull/98",
    state: "open",
    head: {
      ref: "story/sc-18419-epic-18304-pipeline-flexibility-mlx-perf",
      sha: inferenceHead,
      repo: { full_name: REPOSITORIES.inference.slug },
    },
    base: { ref: state.featureBranch, repo: { full_name: REPOSITORIES.inference.slug } },
  });
  const unrecordedOpen = auditState(
    f.statePath,
    f.reportPath,
    { github: f.github, shortcut: f.shortcut, clock },
  );
  assert.equal(unrecordedOpen.gates.noOpenTrainPullRequests, false);
  assert.deepEqual(unrecordedOpen.livePullRequests.open.map(({ number }) => number), [98]);
  assert.equal(unrecordedOpen.cleanup.eligible, false);
  Object.assign(f.github.prs.get("inference:98"), {
    state: "closed",
    merged_at: EPOCH,
    merge_commit_sha: inferenceMerge,
  });
  const unrecordedMerged = auditState(
    f.statePath,
    f.reportPath,
    { github: f.github, shortcut: f.shortcut, clock },
  );
  assert.equal(unrecordedMerged.gates.noOpenTrainPullRequests, true);
  assert.equal(unrecordedMerged.gates.allMergedTrainPullRequestsRecorded, false);
  assert.deepEqual(unrecordedMerged.livePullRequests.unrecordedMerged.map(({ number }) => number), [98]);
  assert.equal(unrecordedMerged.cleanup.eligible, false);
  f.github.prs.delete("inference:98");

  f.github.prs.set("inference:97", {
    html_url: "https://github.com/SceneWorks/inference/pull/97",
    state: "closed",
    head: {
      ref: "story/sc-18419-epic-18304-pipeline-flexibility-mlx-perf",
      sha: inferenceHead,
      repo: { full_name: REPOSITORIES.inference.slug },
    },
    base: { ref: "main", repo: { full_name: REPOSITORIES.inference.slug } },
    merged_at: EPOCH,
    merge_commit_sha: inferenceMerge,
  });
  const wrongBaseMerged = auditState(
    f.statePath,
    f.reportPath,
    { github: f.github, shortcut: f.shortcut, clock },
  );
  assert.equal(wrongBaseMerged.gates.allMergedTrainPullRequestsRecorded, false);
  assert.equal(wrongBaseMerged.gates.noLiveTrainTopologyErrors, false);
  assert.deepEqual(wrongBaseMerged.livePullRequests.unrecordedMerged.map(({ number }) => number), [97]);
  assert.equal(wrongBaseMerged.cleanup.eligible, false);
  f.github.prs.delete("inference:97");

  const shortcutStory = f.shortcut.storyValues[0];
  Object.assign(shortcutStory, {
    workflow_state_id: 20,
    completed: false,
    blocked: true,
    blocker: true,
  });
  const unfinishedShortcut = auditState(
    f.statePath,
    f.reportPath,
    { github: f.github, shortcut: f.shortcut, clock },
  );
  assert.equal(unfinishedShortcut.gates.shortcutInventoryMatches, true);
  assert.equal(unfinishedShortcut.gates.shortcutStoriesDone, false);
  assert.equal(unfinishedShortcut.gates.shortcutNoBlockedStories, false);
  assert.equal(unfinishedShortcut.gates.shortcutNoUnresolvedBlockers, false);
  assert.equal(unfinishedShortcut.gates.allExpectedStoriesMerged, true);
  assert.equal(unfinishedShortcut.cleanup.eligible, false);
  Object.assign(shortcutStory, {
    workflow_state_id: 30,
    completed: true,
    blocked: false,
    blocker: false,
  });

  const stalePinMerge = advanceMain(
    f.origins.sceneworks,
    "Cargo.toml",
    manifest(inferenceHead),
  );
  f.github.prs.get("sceneworks:20").merge_commit_sha = stalePinMerge;
  const staleReachablePin = auditState(f.statePath, f.reportPath, { github: f.github, shortcut: f.shortcut, clock });
  assert.equal(staleReachablePin.inferencePin.reachableFromInferenceMain, true);
  assert.equal(staleReachablePin.inferencePin.matchesFinalInferenceMerge, false);
  assert.equal(staleReachablePin.gates.finalPinExactAndReachable, false);
  assert.equal(staleReachablePin.gates.finalPinMatchesInferenceMerge, false);
  assert.equal(staleReachablePin.cleanup.eligible, false);
  f.github.prs.get("sceneworks:20").merge_commit_sha = swMerge;

  const sceneWorksChecksKey = `sceneworks:${swHead}`;
  const baselineSceneWorksChecks = structuredClone(f.github.checks.get(sceneWorksChecksKey));
  const requiredName = baselineSceneWorksChecks[0].name;
  f.github.checks.set(sceneWorksChecksKey, baselineSceneWorksChecks.map((run, index) =>
    index === 0 ? { ...run, app: { id: 1 } } : run));
  const wrongApp = auditState(f.statePath, f.reportPath, { github: f.github, shortcut: f.shortcut, clock });
  assert.equal(wrongApp.gates.allRecordedPrHeadsGreen, false);
  assert.ok(wrongApp.pullRequests.find(({ repo }) => repo === "sceneworks").missingContexts.includes(requiredName));

  const laterFailure = trustedCheck(requiredName, 90000, {
    conclusion: "failure",
    started_at: "2026-08-10T13:00:00Z",
    completed_at: "2026-08-10T13:00:00Z",
  });
  const stillLaterWrongApp = trustedCheck(requiredName, 90001, {
    app: { id: 1 },
    started_at: "2026-08-10T14:00:00Z",
    completed_at: "2026-08-10T14:00:00Z",
  });
  f.github.checks.set(sceneWorksChecksKey, [
    ...baselineSceneWorksChecks,
    laterFailure,
    stillLaterWrongApp,
  ]);
  const staleSuccess = auditState(f.statePath, f.reportPath, { github: f.github, shortcut: f.shortcut, clock });
  assert.equal(staleSuccess.gates.allRecordedPrHeadsGreen, false);

  f.github.checks.get(sceneWorksChecksKey).push(trustedCheck(requiredName, 90002, {
    started_at: "2026-08-10T15:00:00Z",
    completed_at: "2026-08-10T15:00:00Z",
  }));
  const latestTrustedSuccess = auditState(f.statePath, f.reportPath, { github: f.github, shortcut: f.shortcut, clock });
  assert.equal(latestTrustedSuccess.gates.allRecordedPrHeadsGreen, true);
  f.github.checks.set(sceneWorksChecksKey, baselineSceneWorksChecks);

  const swDrift = sideCommit(f.origins.sceneworks, "feature-ref-drift.txt");
  run("git", ["update-ref", `refs/heads/${state.featureBranch}`, swDrift], { cwd: f.origins.sceneworks });
  const swDrifted = auditState(f.statePath, f.reportPath, { github: f.github, shortcut: f.shortcut, clock });
  assert.equal(swDrifted.gates.featureRefsMatchFinalPrHeads, false);
  assert.equal(swDrifted.cleanup.eligible, false);
  run("git", ["update-ref", `refs/heads/${state.featureBranch}`, swHead], { cwd: f.origins.sceneworks });

  const inferenceDrift = sideCommit(f.origins.inference, "feature-ref-drift.txt");
  run("git", ["update-ref", `refs/heads/${state.featureBranch}`, inferenceDrift], { cwd: f.origins.inference });
  const inferenceDrifted = auditState(f.statePath, f.reportPath, { github: f.github, shortcut: f.shortcut, clock });
  assert.equal(inferenceDrifted.gates.featureRefsMatchFinalPrHeads, false);
  assert.equal(inferenceDrifted.cleanup.eligible, false);
  run("git", ["update-ref", `refs/heads/${state.featureBranch}`, inferenceHead], { cwd: f.origins.inference });

  f.github.prs.set("inference:22", {
    html_url: "https://github.com/SceneWorks/inference/pull/22",
    head: {
      ref: "story/sc-18419-epic-18304-pipeline-flexibility-mlx-perf",
      sha: inferenceHead,
      repo: { full_name: REPOSITORIES.inference.slug },
    },
    base: { ref: state.featureBranch, repo: { full_name: REPOSITORIES.inference.slug } },
  });
  f.github.timelines.set("inference:22", [{ event: "added_to_merge_queue", created_at: EPOCH }]);
  recordEvidence(f.statePath, { repo: "inference", number: 22 }, { github: f.github, clock });
  const unmergedDuplicate = auditState(f.statePath, f.reportPath, { github: f.github, shortcut: f.shortcut, clock });
  assert.equal(unmergedDuplicate.expectedStories["sc-18419"], true);
  assert.equal(unmergedDuplicate.gates.allRecordedPrsMerged, false);
  assert.equal(unmergedDuplicate.cleanup.eligible, false);
});

test("pin parser recognizes Cargo dependency tables, multiline inline tables, targets, patches, and comments", () => {
  const exact = "a".repeat(40);
  const text = `
[package]
name = "fixture"
version = "0.0.0"

[package.metadata]
decoy = "https://github.com/SceneWorks/inference"

[dependencies]
inline = { git = "https://github.com/SceneWorks/inference", rev = "${exact}" } # accepted
multiline = {
  git = "https://github.com/SceneWorks/inference", # the URL is not a comment decoy
  rev = "${exact}",
}

[dependencies.detail]
git = "https://github.com/SceneWorks/inference"
rev = "${exact}"

[target.'cfg(windows)'.dependencies]
targeted = { git = "https://github.com/SceneWorks/inference", rev = "${exact}" }

[workspace.dependencies]
workspace-dep = { git = "https://github.com/SceneWorks/inference", rev = "${exact}" }

[patch."https://github.com/huggingface/candle"]
candle-kernels = {
  git = "https://github.com/SceneWorks/inference",
  rev = "${exact}",
}

# ignored = { git = "https://github.com/SceneWorks/inference", rev = "${"b".repeat(40)}" }
`;
  const parsed = extractInferencePin([{ path: "Cargo.toml", text }]);
  assert.equal(parsed.sha, exact);
  assert.equal(parsed.sources.length, 6);
});

test("pin parser rejects alternate inference URLs, branch or tag selectors, missing revs, and inconsistent pins", () => {
  const exact = "a".repeat(40);
  for (const url of [
    "https://github.com/SceneWorks/inference.git",
    "https://github.com/SceneWorks/inference/",
    "http://github.com/SceneWorks/inference",
    "git+https://github.com/SceneWorks/inference",
    "git://github.com/SceneWorks/inference",
    "ssh://git@github.com/SceneWorks/inference.git",
    "git@github.com:SceneWorks/inference.git",
    "https://GITHUB.com/sceneworks/INFERENCE",
    "https://www.github.com/SceneWorks/inference",
    "https://github.com./SceneWorks/inference",
    "https://github%2Ecom/SceneWorks/inference",
    "https://www%2Egithub%2Ecom./SceneWorks/inference",
    "https://www.github.com%2E/SceneWorks/inference.git",
    "ssh://git@www.github.com./SceneWorks/inference.git",
    "git@www.github.com.:SceneWorks/inference.git",
    "ssh://git@ssh.github.com:443/SceneWorks/inference.git",
    "git+ssh://git@ssh.github.com:443/SceneWorks/inference.git",
    "ssh://git@ssh.github.com.:443/SceneWorks/inference.git",
    "ssh://git@ssh%2Egithub%2Ecom%2E:443/SceneWorks/inference.git",
    "git@ssh.github.com:SceneWorks/inference.git",
    "git@github.com:/SceneWorks/inference.git",
    "https://github.com/SceneWorks/inference?branch=main",
    "https://github.com/SceneWorks%2Finference",
  ]) {
    assert.throws(
      () => extractInferencePin([{
        path: "Cargo.toml",
        text: manifest(exact).replace("https://github.com/SceneWorks/inference", url),
      }]),
      /canonical inference URL/,
      url,
    );
  }
  assert.throws(
    () => extractInferencePin([{ path: "Cargo.toml", text: manifest(exact).replace(`rev = "${exact}"`, 'tag = "runtime"') }]),
    /exact 40-hex rev/,
  );
  assert.throws(
    () => extractInferencePin([{
      path: "Cargo.toml",
      text: manifest(exact).replace(" }", ', branch = "main" }'),
    }]),
    /without branch or tag/,
  );
  assert.throws(
    () => extractInferencePin([{ path: "Cargo.toml", text: manifest(exact.slice(0, 12)) }]),
    /exact 40-hex rev/,
  );
  assert.throws(
    () => extractInferencePin([{
      path: "Cargo.toml",
      text: `[dependencies]\nruntime = {\n  git = "https://github.com/SceneWorks/inference",\n  # rev = "${exact}"\n}\n`,
    }]),
    /exact 40-hex rev/,
  );
  assert.throws(() => extractInferencePin([{ path: "Cargo.toml", text: "[workspace]\n" }]), /no tracked/);
  assert.throws(
    () => extractInferencePin([
      { path: "Cargo.toml", text: manifest(exact) },
      { path: "crates/a/Cargo.toml", text: manifest("b".repeat(40)) },
    ]),
    /inconsistent/,
  );
});

test("pin parser ignores comments, metadata, deceptive hosts, and repository-name prefixes", () => {
  const exact = "a".repeat(40);
  const parsed = extractInferencePin([{
    path: "Cargo.toml",
    text: `
[package.metadata]
canonical-looking = "https://github.com/SceneWorks/inference"
userinfo = "https://github.com@evil.example/SceneWorks/inference"

[dependencies]
host-attack = { git = "https://github.com@evil.example/SceneWorks/inference", branch = "main" }
ssh-host-attack = { git = "ssh://git@ssh.github.com.evil.example:443/SceneWorks/inference", branch = "main" }
prefix-attack = { git = "https://github.com/SceneWorks/inference-malicious", branch = "main" }
real = { git = "https://github.com/SceneWorks/inference", rev = "${exact}" }
# commented = { git = "https://github.com/SceneWorks/inference", branch = "main" }
`,
  }]);
  assert.equal(parsed.sha, exact);
  assert.equal(parsed.sources.length, 1);
});
