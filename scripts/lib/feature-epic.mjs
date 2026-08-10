import { execFileSync } from "node:child_process";
import { createHash, randomUUID } from "node:crypto";
import {
  closeSync,
  existsSync,
  fsyncSync,
  mkdirSync,
  openSync,
  readFileSync,
  readdirSync,
  renameSync,
  rmSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { hostname } from "node:os";
import { basename, dirname, isAbsolute, join, relative, resolve, sep } from "node:path";

export const STATE_VERSION = 3;
export const SHA_RE = /^[0-9a-f]{40}$/;
export const EPIC_RE = /^sc-([1-9][0-9]*)$/;
export const STORY_RE = /^sc-([1-9][0-9]*)$/;
export const SLUG_RE = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;
export const FEATURE_RE = /^feature\/sc-([1-9][0-9]*)-([a-z0-9]+(?:-[a-z0-9]+)*)$/;
export const STORY_BRANCH_RE =
  /^story\/sc-([1-9][0-9]*)-epic-([1-9][0-9]*)-([a-z0-9]+(?:-[a-z0-9]+)*)$/;
export const SYNC_BRANCH_RE = /^sync\/sc-([1-9][0-9]*)-main-(\d{4}-\d{2}-\d{2})$/;
export const EPIC_STORY_PREFIX_RE = /^story\/sc-[1-9][0-9]*-epic(?:-|$)/;
export const PRECANONICAL_STORY_BRANCH_RE =
  /^story\/(sc-[1-9][0-9]*)-(?!epic(?:-|$))[a-z0-9]+(?:-[a-z0-9]+)*$/;
export const GOVERNANCE_PROOF_BRANCH_RE =
  /^story\/(sc-[1-9][0-9]*)-epic-([1-9][0-9]*)-ruleset-proof$/;

export const REPOSITORIES = Object.freeze({
  sceneworks: Object.freeze({
    key: "sceneworks",
    slug: "SceneWorks/SceneWorks",
    cloneUrl: "https://github.com/SceneWorks/SceneWorks.git",
    directory: "SceneWorks",
  }),
  inference: Object.freeze({
    key: "inference",
    slug: "SceneWorks/inference",
    cloneUrl: "https://github.com/SceneWorks/inference.git",
    directory: "inference",
  }),
});

export const REQUIRED_CONTEXTS = Object.freeze({
  sceneworks: Object.freeze([
    "web",
    "parity",
    "candle",
    "build-windows",
    "check-linux",
    "check-macos",
    "macOS build, lint and workspace tests (hosted)",
  ]),
  inference: Object.freeze(["CI gate"]),
});

const DEFAULT_PRIVILEGED_WORKFLOWS = Object.freeze([
  Object.freeze({ repo: "sceneworks", workflow: "windows-candle.yml" }),
  Object.freeze({ repo: "sceneworks", workflow: "macos-mlx.yml" }),
]);

const INFERENCE_URL = "https://github.com/SceneWorks/inference";
const FEATURE_PATTERN = "refs/heads/feature/*";
const GITHUB_ACTIONS_APP_ID = 15368;
const ACCEPTED_CHECK_CONCLUSIONS = new Set(["success", "neutral", "skipped"]);
const MERGE_QUEUE_EVENTS = new Set(["added_to_merge_queue", "merge_queue_entry"]);
const SHORTCUT_API_ROOT = "https://api.app.shortcut.com/api/v3";
const STATE_LOCK_VERSION = 1;
const STATE_LOCK_OWNER = "owner.json";
const MIN_LOCK_STALE_AFTER_MS = 300_000;
const DIGEST_RE = /^[0-9a-f]{64}$/;
const SHORTCUT_WORKFLOW_STATE_TYPES = new Set(["backlog", "unstarted", "started", "done"]);
const PULL_REQUEST_LIST_JQ =
  '.[] | {number, state, html_url, head: {ref: .head.ref}, base: {ref: .base.ref}} | @base64';
const PLAN_MODES = new Set(["create", "adopt-existing"]);
const ADOPTION_DISPOSITIONS = new Set([
  "precanonical-story",
  "historical-train",
  "governance-proof",
]);

function pullRequestRule(requireCodeOwnerReview) {
  return {
    type: "pull_request",
    parameters: {
      required_approving_review_count: 0,
      dismiss_stale_reviews_on_push: false,
      required_reviewers: [],
      require_code_owner_review: requireCodeOwnerReview,
      dismissal_restriction: { enabled: false, allowed_actors: [] },
      require_last_push_approval: false,
      required_review_thread_resolution: false,
      allowed_merge_methods: ["merge"],
    },
  };
}

function statusRule(contexts) {
  return {
    type: "required_status_checks",
    parameters: {
      strict_required_status_checks_policy: true,
      do_not_enforce_on_create: false,
      required_status_checks: contexts.map((context) => ({
        context,
        integration_id: GITHUB_ACTIONS_APP_ID,
      })),
    },
  };
}

export function featurePolicyPayloads(repo, branch, epic) {
  normalizeRepoKey(repo);
  assertSafeFeatureBranch(branch);
  normalizeEpic(epic);
  if (!branch.startsWith(`feature/${epic}-`)) {
    refuse(`exact queue branch ${branch} does not belong to ${epic}`, "POLICY_EPIC_CONFLICT");
  }
  const exactQueue = {
    name: `Feature queue: ${epic}`,
    target: "branch",
    enforcement: "active",
    bypass_actors: [],
    conditions: { ref_name: { include: [`refs/heads/${branch}`], exclude: [] } },
    rules: [
      {
        type: "merge_queue",
        parameters: {
          merge_method: "MERGE",
          max_entries_to_build: 5,
          min_entries_to_merge: 1,
          max_entries_to_merge: 1,
          min_entries_to_merge_wait_minutes: 5,
          grouping_strategy: "ALLGREEN",
          check_response_timeout_minutes: 60,
        },
      },
    ],
  };
  return {
    wildcardBase: {
      name: "Feature integration base policy",
      target: "branch",
      enforcement: "active",
      bypass_actors: [],
      conditions: { ref_name: { include: [FEATURE_PATTERN], exclude: [] } },
      rules: [
        { type: "non_fast_forward" },
        pullRequestRule(repo === "sceneworks"),
        statusRule(REQUIRED_CONTEXTS[repo]),
      ],
    },
    deletionGuard: {
      name: "Feature integration deletion guard",
      target: "branch",
      enforcement: "active",
      bypass_actors: [],
      conditions: { ref_name: { include: [FEATURE_PATTERN], exclude: [] } },
      rules: [{ type: "deletion" }],
    },
    exactQueue,
    stagedExactQueue: { ...exactQueue, enforcement: "disabled" },
  };
}

function sortObject(value) {
  if (Array.isArray(value)) return value.map(sortObject);
  if (!value || typeof value !== "object") return value;
  return Object.fromEntries(
    Object.keys(value).sort().map((key) => [key, sortObject(value[key])]),
  );
}

function normalizedRule(rule) {
  const output = { type: rule.type };
  if (rule.type === "pull_request") {
    const parameters = rule.parameters ?? {};
    output.parameters = {
      required_approving_review_count: parameters.required_approving_review_count,
      dismiss_stale_reviews_on_push: parameters.dismiss_stale_reviews_on_push,
      required_reviewers: parameters.required_reviewers ?? [],
      require_code_owner_review: parameters.require_code_owner_review,
      dismissal_restriction: parameters.dismissal_restriction ?? { enabled: false, allowed_actors: [] },
      require_last_push_approval: parameters.require_last_push_approval,
      required_review_thread_resolution: parameters.required_review_thread_resolution,
      allowed_merge_methods: [...(parameters.allowed_merge_methods ?? [])].sort(),
    };
  } else if (rule.type === "required_status_checks") {
    const parameters = rule.parameters ?? {};
    output.parameters = {
      strict_required_status_checks_policy: parameters.strict_required_status_checks_policy,
      do_not_enforce_on_create: parameters.do_not_enforce_on_create,
      required_status_checks: [...(parameters.required_status_checks ?? [])]
        .map(({ context, integration_id }) => ({ context, integration_id }))
        .sort((a, b) => `${a.context}:${a.integration_id}`.localeCompare(`${b.context}:${b.integration_id}`)),
    };
  } else if (rule.type === "merge_queue") {
    const parameters = rule.parameters ?? {};
    output.parameters = {
      merge_method: parameters.merge_method,
      max_entries_to_build: parameters.max_entries_to_build,
      min_entries_to_merge: parameters.min_entries_to_merge,
      max_entries_to_merge: parameters.max_entries_to_merge,
      min_entries_to_merge_wait_minutes: parameters.min_entries_to_merge_wait_minutes,
      grouping_strategy: parameters.grouping_strategy,
      check_response_timeout_minutes: parameters.check_response_timeout_minutes,
    };
  }
  return output;
}

export function normalizeRuleset(ruleset) {
  return sortObject({
    name: ruleset?.name,
    target: ruleset?.target,
    enforcement: ruleset?.enforcement,
    bypass_actors: ruleset?.bypass_actors ?? [],
    conditions: {
      ref_name: {
        include: [...(ruleset?.conditions?.ref_name?.include ?? [])].sort(),
        exclude: [...(ruleset?.conditions?.ref_name?.exclude ?? [])].sort(),
      },
    },
    rules: [...(ruleset?.rules ?? [])].map(normalizedRule).sort((a, b) => a.type.localeCompare(b.type)),
  });
}

export function rulesetDigest(ruleset) {
  return createHash("sha256").update(JSON.stringify(normalizeRuleset(ruleset))).digest("hex");
}

function sameRuleset(actual, expected) {
  return rulesetDigest(actual) === rulesetDigest(expected);
}

export class FeatureEpicError extends Error {
  constructor(message, code = "FEATURE_EPIC_REFUSAL") {
    super(message);
    this.name = "FeatureEpicError";
    this.code = code;
  }
}

function refuse(message, code) {
  throw new FeatureEpicError(message, code);
}

export function assertAbsolutePath(value, label) {
  if (typeof value !== "string" || !isAbsolute(value)) {
    refuse(`${label} must be an absolute path`, "INVALID_PATH");
  }
  return resolve(value);
}

export function normalizeEpic(value) {
  if (!EPIC_RE.test(value ?? "")) {
    refuse(`epic must be a canonical sc-N id, got ${JSON.stringify(value)}`, "INVALID_EPIC");
  }
  return value;
}

export function normalizeSlug(value) {
  if (!SLUG_RE.test(value ?? "")) {
    refuse(`slug must be lower-case kebab-case, got ${JSON.stringify(value)}`, "INVALID_SLUG");
  }
  return value;
}

export function normalizeStories(values) {
  if (!Array.isArray(values) || values.length === 0) {
    refuse("at least one expected story id is required", "MISSING_STORIES");
  }
  const normalized = values.map((value) => {
    if (!STORY_RE.test(value ?? "")) {
      refuse(`story must be a canonical sc-N id, got ${JSON.stringify(value)}`, "INVALID_STORY");
    }
    return value;
  });
  const unique = [...new Set(normalized)];
  if (unique.length !== normalized.length) {
    refuse("expected story ids must be unique", "DUPLICATE_STORY");
  }
  return unique.sort((a, b) => Number(a.slice(3)) - Number(b.slice(3)));
}

export function featureBranch(epic, slug) {
  normalizeEpic(epic);
  normalizeSlug(slug);
  return `feature/${epic}-${slug}`;
}

export function storyBranch(story, epic, slug) {
  if (!STORY_RE.test(story ?? "")) refuse("invalid story id", "INVALID_STORY");
  normalizeEpic(epic);
  normalizeSlug(slug);
  return `story/${story}-epic-${epic.slice(3)}-${slug}`;
}

export function assertSafeFeatureBranch(branch) {
  const match = FEATURE_RE.exec(branch ?? "");
  if (!match) {
    refuse(
      `feature branch must be feature/sc-N-kebab-slug, got ${JSON.stringify(branch)}`,
      "INVALID_FEATURE_BRANCH",
    );
  }
  if (["main", "release", "next", "master", "default"].includes(match[2])) {
    refuse(`feature branch slug ${JSON.stringify(match[2])} is default-like`, "DEFAULT_LIKE_REF");
  }
  return { epic: `sc-${match[1]}`, slug: match[2] };
}

function pathInside(parent, child) {
  const rel = relative(parent, child);
  return rel === "" || (!rel.startsWith(`..${sep}`) && rel !== "..");
}

export function assertEmptyWorkspaceParent(path, statePath) {
  const parent = assertAbsolutePath(path, "workspace parent");
  if (!existsSync(parent) || !statSync(parent).isDirectory()) {
    refuse(`workspace parent must already exist as a directory: ${parent}`, "MISSING_WORKSPACE");
  }
  const state = assertAbsolutePath(statePath, "state path");
  if (pathInside(parent, state)) {
    refuse("state path must be outside the workspace parent", "STATE_IN_WORKSPACE");
  }
  const entries = readdirSync(parent);
  if (entries.length !== 0) {
    refuse(`workspace parent must be empty: ${parent}`, "REUSED_WORKSPACE");
  }
  return parent;
}

export function readJson(path, label = "JSON file") {
  try {
    return JSON.parse(readFileSync(path, "utf8"));
  } catch (error) {
    refuse(`${label} is not readable valid JSON: ${error.message}`, "INVALID_JSON");
  }
}

// The state file is the transaction journal. Flush the replacement before the rename and flush
// the containing directory after it, so a process or host failure cannot truthfully leave a
// remote mutation that the last acknowledged state record forgot.
export function writeJsonAtomic(path, value) {
  const target = assertAbsolutePath(path, "state/report path");
  const parent = dirname(target);
  mkdirSync(parent, { recursive: true });
  const temporary = join(parent, `.${basename(target)}.${process.pid}.${Date.now()}.tmp`);
  const fd = openSync(temporary, "wx", 0o600);
  try {
    writeFileSync(fd, `${JSON.stringify(value, null, 2)}\n`, "utf8");
    fsyncSync(fd);
  } finally {
    closeSync(fd);
  }
  renameSync(temporary, target);
  const dirFd = openSync(parent, "r");
  try {
    fsyncSync(dirFd);
  } finally {
    closeSync(dirFd);
  }
}

export function stateLockPath(statePath) {
  return `${assertAbsolutePath(statePath, "state path")}.lock`;
}

function lockOwnerPath(lockPath) {
  return join(lockPath, STATE_LOCK_OWNER);
}

function liveLocalProcess(pid) {
  if (!Number.isSafeInteger(pid) || pid <= 0) return null;
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    if (error?.code === "ESRCH") return false;
    return true;
  }
}

function readLockOwner(lockPath) {
  try {
    const owner = readJson(lockOwnerPath(lockPath), "feature epic lock owner");
    if (
      owner?.schemaVersion !== STATE_LOCK_VERSION ||
      typeof owner.token !== "string" ||
      !owner.token ||
      !Number.isSafeInteger(owner.pid) ||
      owner.pid <= 0 ||
      typeof owner.hostname !== "string" ||
      !owner.hostname ||
      typeof owner.operation !== "string" ||
      !owner.operation ||
      typeof owner.startedAt !== "string" ||
      !Number.isFinite(evidenceTimestamp(owner.startedAt))
    ) {
      return null;
    }
    return owner;
  } catch {
    return null;
  }
}

function lockAgeMs(lockPath, owner, clock) {
  const timestamp = owner ? evidenceTimestamp(owner.startedAt) : statSync(lockPath).mtimeMs;
  return Math.max(0, clock().valueOf() - timestamp);
}

function describeLockOwner(owner) {
  if (!owner) return "owner metadata is missing or invalid";
  return `pid ${owner.pid} on ${owner.hostname} running ${owner.operation} since ${owner.startedAt}`;
}

function reclaimableLock(lockPath, owner, staleAfterMs, clock) {
  const localHost = hostname();
  if (owner?.hostname === localHost) {
    const alive = liveLocalProcess(owner.pid);
    if (alive === true) return false;
    if (alive === false) return true;
  }
  return (
    Number.isFinite(staleAfterMs) &&
    staleAfterMs >= 0 &&
    lockAgeMs(lockPath, owner, clock) >= staleAfterMs
  );
}

function acquireStateLock(statePath, { operation, clock, staleAfterMs }) {
  if (
    staleAfterMs !== undefined &&
    (!Number.isSafeInteger(staleAfterMs) || staleAfterMs < MIN_LOCK_STALE_AFTER_MS)
  ) {
    refuse(
      `stale-lock recovery threshold must be at least ${MIN_LOCK_STALE_AFTER_MS} milliseconds`,
      "INVALID_STALE_LOCK_THRESHOLD",
    );
  }
  const lockPath = stateLockPath(statePath);
  mkdirSync(dirname(lockPath), { recursive: true });
  const localHost = hostname();
  for (let attempt = 0; attempt < 4; attempt += 1) {
    try {
      mkdirSync(lockPath, { mode: 0o700 });
      const owner = {
        schemaVersion: STATE_LOCK_VERSION,
        token: randomUUID(),
        pid: process.pid,
        hostname: localHost,
        operation,
        startedAt: nowIso(clock),
      };
      try {
        writeJsonAtomic(lockOwnerPath(lockPath), owner);
      } catch (error) {
        rmSync(lockPath, { recursive: true, force: true });
        throw error;
      }
      return { path: lockPath, owner };
    } catch (error) {
      if (error?.code !== "EEXIST") throw error;
    }

    const owner = readLockOwner(lockPath);
    let reclaimable;
    try {
      reclaimable = reclaimableLock(lockPath, owner, staleAfterMs, clock);
    } catch (error) {
      if (error?.code === "ENOENT") continue;
      throw error;
    }
    if (!reclaimable) {
      refuse(
        `state is locked by ${describeLockOwner(owner)}; wait for it to finish` +
          (owner ? "" : " or explicitly authorize stale-lock recovery after a safe age"),
        "STATE_LOCKED",
      );
    }
    const quarantine = `${lockPath}.stale-${process.pid}-${randomUUID()}`;
    try {
      renameSync(lockPath, quarantine);
    } catch (error) {
      if (error?.code === "ENOENT" || error?.code === "EEXIST") continue;
      throw error;
    }
    rmSync(quarantine, { recursive: true, force: true });
  }
  refuse("could not acquire the state lock after concurrent stale-lock recovery", "STATE_LOCK_RACE");
}

function releaseStateLock(lock) {
  const owner = readLockOwner(lock.path);
  if (!owner || owner.token !== lock.owner.token || owner.pid !== process.pid) {
    refuse("state lock ownership changed while the command was running", "STATE_LOCK_OWNERSHIP_LOST");
  }
  rmSync(lock.path, { recursive: true });
}

export function withStateLock(
  statePath,
  { operation = "feature-epic", clock = () => new Date(), staleAfterMs } = {},
  callback,
) {
  if (typeof callback !== "function") refuse("state lock callback is required", "INVALID_LOCK_CALLBACK");
  const lock = acquireStateLock(statePath, { operation, clock, staleAfterMs });
  let value;
  let operationError = null;
  try {
    value = callback();
  } catch (error) {
    operationError = error instanceof Error ? error : new Error(String(error));
  }
  try {
    releaseStateLock(lock);
  } catch (releaseError) {
    if (!operationError) throw releaseError;
    operationError.message += `; state lock release also failed: ${releaseError.message}`;
  }
  if (operationError) throw operationError;
  return value;
}

export function loadState(path) {
  const absolute = assertAbsolutePath(path, "state path");
  const state = readJson(absolute, "feature epic state");
  validateState(state);
  return state;
}

function validShortcutWorkflowState(value) {
  return (
    Number.isSafeInteger(value?.id) &&
    value.id > 0 &&
    Number.isSafeInteger(value.workflowId) &&
    value.workflowId > 0 &&
    typeof value.name === "string" &&
    value.name.length > 0 &&
    SHORTCUT_WORKFLOW_STATE_TYPES.has(value.type)
  );
}

function sameJson(left, right) {
  return JSON.stringify(sortObject(left)) === JSON.stringify(sortObject(right));
}

function hasExactKeys(value, keys) {
  return (
    value &&
    typeof value === "object" &&
    !Array.isArray(value) &&
    JSON.stringify(Object.keys(value).sort()) === JSON.stringify([...keys].sort())
  );
}

function validEvidenceTime(value) {
  return typeof value === "string" && Number.isFinite(evidenceTimestamp(value));
}

function expectedPullRequestUrl(repo, number) {
  return `https://github.com/${REPOSITORIES[repo].slug}/pull/${number}`;
}

function validateRequiredCheckReceipts(receipts, required, label) {
  if (!Array.isArray(required) || !Array.isArray(receipts) || receipts.length !== required.length) {
    refuse(`${label} must contain every required check exactly once`, "STATE_CHECK_RECEIPT_CONFLICT");
  }
  for (const [index, receipt] of receipts.entries()) {
    if (
      !hasExactKeys(receipt, ["name", "id", "appId", "status", "conclusion", "completedAt", "url"]) ||
      receipt.name !== required[index] ||
      !Number.isSafeInteger(receipt.id) ||
      receipt.id <= 0 ||
      receipt.appId !== GITHUB_ACTIONS_APP_ID ||
      receipt.status !== "completed" ||
      !ACCEPTED_CHECK_CONCLUSIONS.has(receipt.conclusion) ||
      !validEvidenceTime(receipt.completedAt) ||
      typeof receipt.url !== "string" ||
      receipt.url.length === 0
    ) {
      refuse(`${label} contains malformed or weakened required-check evidence`, "STATE_CHECK_RECEIPT_CONFLICT");
    }
  }
}

export function classifyAdoptedPullRequest(disposition, head, base) {
  if (!ADOPTION_DISPOSITIONS.has(disposition)) {
    refuse(`unknown adoption disposition ${JSON.stringify(disposition)}`, "INVALID_ADOPTION_DISPOSITION");
  }
  if (disposition === "precanonical-story") {
    const story = PRECANONICAL_STORY_BRANCH_RE.exec(head ?? "")?.[1] ?? null;
    if (!story || base !== "main") {
      refuse(
        `precanonical story adoption requires story/sc-N-slug -> main, got ${head} -> ${base}`,
        "INVALID_PRECANONICAL_STORY",
      );
    }
    return { kind: "story", story, epic: null, slug: null };
  }
  if (disposition === "governance-proof") {
    const proof = GOVERNANCE_PROOF_BRANCH_RE.exec(head ?? "");
    if (!proof || base !== `feature/ruleset-proof-${proof[1]}`) {
      refuse(
        `governance proof adoption requires the exact story/sc-N-epic-N-ruleset-proof topology`,
        "INVALID_GOVERNANCE_PROOF",
      );
    }
    return { kind: "governance-proof", story: proof[1], epic: `sc-${proof[2]}`, slug: null };
  }
  const classification = classifyPullRequest(head, base);
  if (!new Set(["story", "sync"]).has(classification.kind)) {
    refuse(
      `historical train adoption permits only strict story or sync topology`,
      "INVALID_HISTORICAL_TRAIN",
    );
  }
  return classification;
}

function validateCanonicalPullRequestReceipt(state, receipt) {
  if (
    !hasExactKeys(receipt, [
      "repo", "number", "url", "head", "base", "headSha", "kind", "story", "recordedAt",
    ]) ||
    !Object.hasOwn(REPOSITORIES, receipt.repo) ||
    !Number.isSafeInteger(receipt.number) ||
    receipt.number <= 0 ||
    receipt.url !== expectedPullRequestUrl(receipt.repo, receipt.number) ||
    typeof receipt.head !== "string" ||
    typeof receipt.base !== "string" ||
    !SHA_RE.test(receipt.headSha ?? "") ||
    !validEvidenceTime(receipt.recordedAt)
  ) {
    refuse("state contains malformed canonical PR evidence", "STATE_PR_RECEIPT_CONFLICT");
  }
  let classification;
  try {
    classification = classifyPullRequest(receipt.head, receipt.base);
  } catch {
    refuse("state canonical PR topology is not derivable from its refs", "STATE_PR_RECEIPT_CONFLICT");
  }
  if (
    classification.kind !== receipt.kind ||
    (classification.story ?? null) !== receipt.story ||
    classification.epic !== state.epic ||
    classification.slug !== state.slug ||
    (receipt.story !== null && !state.expectedStories.includes(receipt.story))
  ) {
    refuse("state canonical PR kind or story conflicts with its refs", "STATE_PR_RECEIPT_CONFLICT");
  }
}

function validateAdoptedPullRequestReceipt(state, receipt) {
  if (
    !hasExactKeys(receipt, [
      "repo", "number", "url", "disposition", "head", "base", "headSha", "mergeCommit",
      "mergedAt", "kind", "story", "queue", "requiredChecks", "canonicalFeatureSha", "adoptedAt",
    ]) ||
    !Object.hasOwn(REPOSITORIES, receipt.repo) ||
    !Number.isSafeInteger(receipt.number) ||
    receipt.number <= 0 ||
    receipt.url !== expectedPullRequestUrl(receipt.repo, receipt.number) ||
    !ADOPTION_DISPOSITIONS.has(receipt.disposition) ||
    typeof receipt.head !== "string" ||
    typeof receipt.base !== "string" ||
    !SHA_RE.test(receipt.headSha ?? "") ||
    !SHA_RE.test(receipt.mergeCommit ?? "") ||
    !validEvidenceTime(receipt.mergedAt) ||
    !validEvidenceTime(receipt.adoptedAt) ||
    evidenceTimestamp(receipt.adoptedAt) < evidenceTimestamp(receipt.mergedAt) ||
    !hasExactKeys(receipt.queue, ["event", "at", "headSha", "mergeCommit"]) ||
    !MERGE_QUEUE_EVENTS.has(receipt.queue.event) ||
    !validEvidenceTime(receipt.queue.at) ||
    evidenceTimestamp(receipt.queue.at) > evidenceTimestamp(receipt.mergedAt) ||
    receipt.queue.headSha !== receipt.headSha ||
    receipt.queue.mergeCommit !== receipt.mergeCommit
  ) {
    refuse("state contains malformed adopted PR evidence", "STATE_ADOPTION_RECEIPT_CONFLICT");
  }
  let classification;
  try {
    classification = classifyAdoptedPullRequest(receipt.disposition, receipt.head, receipt.base);
  } catch {
    refuse("state adopted PR topology is not derivable from its disposition", "STATE_ADOPTION_RECEIPT_CONFLICT");
  }
  if (
    classification.kind !== receipt.kind ||
    (classification.story ?? null) !== receipt.story ||
    (classification.epic !== null && classification.epic !== state.epic) ||
    (classification.slug !== null && classification.slug !== state.slug) ||
    (receipt.story !== null && !state.expectedStories.includes(receipt.story)) ||
    (receipt.disposition === "governance-proof"
      ? receipt.canonicalFeatureSha !== null
      : receipt.canonicalFeatureSha !== state.repositories?.[receipt.repo]?.adoptedFeatureSha)
  ) {
    refuse("state adopted PR kind, story, or feature binding conflicts with its refs", "STATE_ADOPTION_RECEIPT_CONFLICT");
  }
  validateRequiredCheckReceipts(
    receipt.requiredChecks,
    state.requiredContexts[receipt.repo],
    `state adopted PR ${receipt.repo}:${receipt.number}`,
  );
}

export function validateState(state) {
  if (!state || state.schemaVersion !== STATE_VERSION) {
    refuse(`unsupported or missing state schema version`, "INVALID_STATE");
  }
  const epic = normalizeEpic(state.epic);
  const slug = normalizeSlug(state.slug);
  const expected = featureBranch(epic, slug);
  if (state.featureBranch !== expected) {
    refuse(`state feature branch conflicts with epic and slug`, "STATE_BRANCH_CONFLICT");
  }
  assertSafeFeatureBranch(state.featureBranch);
  if (JSON.stringify(normalizeStories(state.expectedStories)) !== JSON.stringify(state.expectedStories)) {
    refuse("state expectedStories must be in canonical numeric order", "INVALID_STATE");
  }
  const workspaceParent = assertAbsolutePath(state.workspaceParent, "state workspace parent");
  if (
    !PLAN_MODES.has(state.mode) ||
    typeof state.deploymentRequired !== "boolean" ||
    !validEvidenceTime(state.createdAt) ||
    !validEvidenceTime(state.updatedAt) ||
    evidenceTimestamp(state.updatedAt) < evidenceTimestamp(state.createdAt)
  ) {
    refuse("state deploymentRequired must be boolean", "INVALID_STATE");
  }
  if (
    state.shortcut?.epicId !== epic ||
    (state.shortcut.epicUrl !== null && typeof state.shortcut.epicUrl !== "string") ||
    (state.shortcut.epicUpdatedAt !== null &&
      (typeof state.shortcut.epicUpdatedAt !== "string" ||
        !Number.isFinite(evidenceTimestamp(state.shortcut.epicUpdatedAt)))) ||
    state.shortcut.inventoryDigest !== storyInventoryDigest(state.expectedStories) ||
    !state.shortcut.workflowStates ||
    typeof state.shortcut.workflowStates !== "object" ||
    Array.isArray(state.shortcut.workflowStates) ||
    (state.shortcut.inventoryFrozenAt !== null &&
      (typeof state.shortcut.inventoryFrozenAt !== "string" ||
        !Number.isFinite(evidenceTimestamp(state.shortcut.inventoryFrozenAt)))) ||
    !Array.isArray(state.shortcut.refreshes)
  ) {
    refuse("state Shortcut inventory evidence conflicts with the expected stories", "STATE_SHORTCUT_CONFLICT");
  }
  for (const [id, workflowState] of Object.entries(state.shortcut.workflowStates)) {
    if (
      !/^[1-9][0-9]*$/.test(id) ||
      workflowState?.id !== Number(id) ||
      !validShortcutWorkflowState(workflowState)
    ) {
      refuse("state contains malformed Shortcut workflow-state evidence", "STATE_SHORTCUT_CONFLICT");
    }
  }
  let inventoryCursor = [...state.expectedStories];
  let workflowStateCursor = structuredClone(state.shortcut.workflowStates);
  let laterRefreshAt = Number.POSITIVE_INFINITY;
  for (let index = state.shortcut.refreshes.length - 1; index >= 0; index -= 1) {
    const refresh = state.shortcut.refreshes[index];
    if (
      typeof refresh?.at !== "string" ||
      !Number.isFinite(evidenceTimestamp(refresh.at)) ||
      evidenceTimestamp(refresh.at) > laterRefreshAt ||
      !DIGEST_RE.test(refresh.previousDigest ?? "") ||
      !DIGEST_RE.test(refresh.nextDigest ?? "") ||
      !Array.isArray(refresh.added) ||
      !Array.isArray(refresh.removed) ||
      refresh.removed.length !== 0 ||
      !Array.isArray(refresh.workflowStateChanges) ||
      (refresh.epicUpdatedAt !== null &&
        (typeof refresh.epicUpdatedAt !== "string" ||
          !Number.isFinite(evidenceTimestamp(refresh.epicUpdatedAt))))
    ) {
      refuse("state contains malformed Shortcut inventory-refresh evidence", "STATE_SHORTCUT_CONFLICT");
    }
    laterRefreshAt = evidenceTimestamp(refresh.at);
    const normalizedAdded = refresh.added.length > 0 ? normalizeStories(refresh.added) : [];
    if (
      JSON.stringify(normalizedAdded) !== JSON.stringify(refresh.added) ||
      refresh.nextDigest !== storyInventoryDigest(inventoryCursor) ||
      normalizedAdded.some((story) => !inventoryCursor.includes(story))
    ) {
      refuse("state Shortcut inventory-refresh chain is inconsistent", "STATE_SHORTCUT_CONFLICT");
    }
    const previousInventory = inventoryCursor.filter((story) => !normalizedAdded.includes(story));
    if (
      previousInventory.length === 0 ||
      refresh.previousDigest !== storyInventoryDigest(previousInventory)
    ) {
      refuse("state Shortcut inventory-refresh chain is inconsistent", "STATE_SHORTCUT_CONFLICT");
    }

    let previousChangeId = 0;
    for (const change of refresh.workflowStateChanges) {
      if (
        !Number.isSafeInteger(change?.id) ||
        change.id <= previousChangeId ||
        (change.before !== null &&
          (change.before?.id !== change.id || !validShortcutWorkflowState(change.before))) ||
        (change.after !== null &&
          (change.after?.id !== change.id || !validShortcutWorkflowState(change.after))) ||
        sameJson(change.before, change.after) ||
        !sameJson(workflowStateCursor[change.id] ?? null, change.after)
      ) {
        refuse("state Shortcut workflow-state refresh diff is inconsistent", "STATE_SHORTCUT_CONFLICT");
      }
      previousChangeId = change.id;
      if (change.before === null) delete workflowStateCursor[change.id];
      else workflowStateCursor[change.id] = structuredClone(change.before);
    }
    inventoryCursor = previousInventory;
  }
  if (!new Set([
    "planned",
    "staging-policies",
    "disabling-policies",
    "applying-remote-refs",
    "activating-policies",
    "recovery-required",
    "ready",
  ]).has(state.phase)) {
    refuse(`state has an unknown phase ${JSON.stringify(state.phase)}`, "INVALID_STATE");
  }
  for (const [key, fixed] of Object.entries(REPOSITORIES)) {
    const repo = state.repositories?.[key];
    if (!repo || repo.slug !== fixed.slug || repo.cloneUrl !== fixed.cloneUrl) {
      refuse(`state repository ${key} is not the fixed ${fixed.slug}`, "STATE_REPOSITORY_CONFLICT");
    }
    if (
      !SHA_RE.test(repo.plannedMainSha ?? "") ||
      (repo.adoptedFeatureSha !== null && !SHA_RE.test(repo.adoptedFeatureSha ?? "")) ||
      !Array.isArray(repo.adoptedRequiredChecks) ||
      (state.mode === "create" &&
        (repo.adoptedFeatureSha !== null || repo.adoptedRequiredChecks.length !== 0)) ||
      (state.mode === "adopt-existing" && repo.adoptedFeatureSha === null)
    ) {
      refuse(`state repository ${key} has invalid main SHA`, "INVALID_STATE_SHA");
    }
    if (state.mode === "adopt-existing") {
      validateRequiredCheckReceipts(
        repo.adoptedRequiredChecks,
        REQUIRED_CONTEXTS[key],
        `state ${key} adopted feature`,
      );
    }
    if (repo.featureBranch !== expected) {
      refuse(`state repository ${key} has a mismatched mirrored branch`, "MIRRORED_NAME_CONFLICT");
    }
    const expectedInitialFeature = repo.adoptedFeatureSha ?? repo.plannedMainSha;
    if (
      repo.remoteRef !== null &&
      (!SHA_RE.test(repo.remoteRef ?? "") || repo.remoteRef !== expectedInitialFeature)
    ) {
      refuse(`state repository ${key} has a conflicting recorded ref`, "STATE_REF_CONFLICT");
    }
    if (state.mode === "adopt-existing" && repo.remoteRef !== repo.adoptedFeatureSha) {
      refuse(`state adopted repository ${key} must record its immutable feature head`, "STATE_REF_CONFLICT");
    }
    if (repo.workspace?.path !== join(workspaceParent, fixed.directory)) {
      refuse(`state repository ${key} has a conflicting workspace path`, "STATE_WORKSPACE_CONFLICT");
    }
    if (!new Set(["pending", "provisioning", "failed", "ready"]).has(repo.workspace?.status)) {
      refuse(`state repository ${key} has an invalid workspace status`, "INVALID_STATE");
    }
    if (
      repo.workspace.head !== null && repo.workspace.head !== expectedInitialFeature
    ) {
      refuse(`state repository ${key} workspace head conflicts with its immutable baseline`, "STATE_WORKSPACE_CONFLICT");
    }
    const expectedPolicies = featurePolicyPayloads(key, expected, epic);
    const policies = state.policies?.[key];
    for (const [policyKey, expectedPayload] of [
      ["wildcardBase", expectedPolicies.wildcardBase],
      ["deletionGuard", expectedPolicies.deletionGuard],
    ]) {
      const policy = policies?.[policyKey];
      if (
        !Number.isSafeInteger(policy?.id) ||
        policy.id <= 0 ||
        policy.name !== expectedPayload.name ||
        policy.digest !== rulesetDigest(expectedPayload)
      ) {
        refuse(`state ${key} ${policyKey} policy conflicts with the required payload`, "STATE_POLICY_CONFLICT");
      }
    }
    const exact = policies?.exactQueue;
    if (
      exact?.name !== expectedPolicies.exactQueue.name ||
      exact?.digest !== rulesetDigest(expectedPolicies.exactQueue) ||
      !sameRuleset(exact?.payload, expectedPolicies.exactQueue) ||
      exact?.stagedDigest !== rulesetDigest(expectedPolicies.stagedExactQueue) ||
      !sameRuleset(exact?.stagedPayload, expectedPolicies.stagedExactQueue) ||
      (exact.id !== null && (!Number.isSafeInteger(exact.id) || exact.id <= 0)) ||
      !["pending", "disabled", "active", "failed"].includes(exact?.status) ||
      (exact?.status === "pending" && exact.id !== null) ||
      (["disabled", "active"].includes(exact?.status) && exact.id === null)
    ) {
      refuse(`state ${key} exact queue policy conflicts with the required payload`, "STATE_POLICY_CONFLICT");
    }
    if (
      state.mode === "adopt-existing" &&
      (exact.id === null || exact.status !== "active")
    ) {
      refuse(`state adopted repository ${key} lacks its exact active queue`, "STATE_POLICY_CONFLICT");
    }
  }
  if (!SHA_RE.test(state.inferencePin?.sha ?? "")) {
    refuse("state has no exact 40-hex inference pin", "INVALID_STATE_PIN");
  }
  if (
    !state.transaction ||
    !Array.isArray(state.transaction.remoteMutations) ||
    !Array.isArray(state.transaction.policyMutations)
  ) {
    refuse("state has no durable remote-mutation journals", "INVALID_STATE");
  }
  for (const key of [
    "pullRequests",
    "adoptedPullRequests",
    "privilegedWorkflows",
    "privilegedRuns",
    "deployments",
  ]) {
    if (!Array.isArray(state[key])) refuse(`state ${key} must be an array`, "INVALID_STATE");
  }
  for (const [key, required] of Object.entries(REQUIRED_CONTEXTS)) {
    if (JSON.stringify(state.requiredContexts?.[key]) !== JSON.stringify(required)) {
      refuse(`state required contexts for ${key} were weakened or changed`, "STATE_CONTEXT_CONFLICT");
    }
  }
  for (const required of DEFAULT_PRIVILEGED_WORKFLOWS) {
    if (!state.privilegedWorkflows?.some(
      ({ repo, workflow }) => repo === required.repo && workflow === required.workflow,
    )) {
      refuse(`state omits required privileged workflow ${required.repo}:${required.workflow}`, "STATE_RUNTIME_CONFLICT");
    }
  }
  const workflowKeys = new Set();
  for (const workflow of state.privilegedWorkflows) {
    const key = `${workflow?.repo}:${workflow?.workflow}`;
    if (
      !hasExactKeys(workflow, ["repo", "workflow"]) ||
      !Object.hasOwn(REPOSITORIES, workflow.repo) ||
      !/^[A-Za-z0-9][A-Za-z0-9._-]*\.ya?ml$/.test(workflow.workflow ?? "") ||
      workflowKeys.has(key)
    ) {
      refuse("state contains malformed or duplicate privileged workflow requirements", "STATE_RUNTIME_CONFLICT");
    }
    workflowKeys.add(key);
  }

  const pullRequestKeys = new Set();
  for (const receipt of state.pullRequests) {
    validateCanonicalPullRequestReceipt(state, receipt);
    const key = `${receipt.repo}:${receipt.number}`;
    if (pullRequestKeys.has(key)) {
      refuse("state contains duplicate canonical/adopted PR evidence", "STATE_PR_RECEIPT_CONFLICT");
    }
    pullRequestKeys.add(key);
  }
  for (const receipt of state.adoptedPullRequests) {
    validateAdoptedPullRequestReceipt(state, receipt);
    const key = `${receipt.repo}:${receipt.number}`;
    if (pullRequestKeys.has(key)) {
      refuse("state contains duplicate canonical/adopted PR evidence", "STATE_ADOPTION_RECEIPT_CONFLICT");
    }
    pullRequestKeys.add(key);
  }
  const finalIntentTimes = state.pullRequests
    .filter(({ kind }) => kind === "final")
    .map(({ recordedAt }) => recordedAt)
    .sort((left, right) => evidenceTimestamp(left) - evidenceTimestamp(right));
  const expectedInventoryFreeze = state.mode === "adopt-existing"
    ? state.createdAt
    : finalIntentTimes[0] ?? null;
  if (state.shortcut.inventoryFrozenAt !== expectedInventoryFreeze) {
    refuse(
      "state Shortcut inventory freeze does not match immutable adoption/final intent",
      "STATE_SHORTCUT_CONFLICT",
    );
  }

  if (!hasExactKeys(state.finalPullRequests, Object.keys(REPOSITORIES))) {
    refuse("state final PR pointers must cover the fixed repository pair", "STATE_FINAL_POINTER_CONFLICT");
  }
  for (const repo of Object.keys(REPOSITORIES)) {
    const pointer = state.finalPullRequests[repo];
    const finalReceipts = state.pullRequests.filter(
      (receipt) => receipt.repo === repo && receipt.kind === "final",
    );
    if (finalReceipts.length > 1) {
      refuse(`state contains multiple final PRs for ${repo}`, "STATE_FINAL_POINTER_CONFLICT");
    }
    if (pointer === null) {
      if (finalReceipts.length !== 0) {
        refuse(`state ${repo} final receipt lacks its exact pointer`, "STATE_FINAL_POINTER_CONFLICT");
      }
      continue;
    }
    if (
      !hasExactKeys(pointer, ["number", "headSha", "url", "mergeCommit", "mergedAt"]) ||
      !Number.isSafeInteger(pointer.number) ||
      pointer.number <= 0 ||
      !SHA_RE.test(pointer.headSha ?? "") ||
      pointer.url !== expectedPullRequestUrl(repo, pointer.number) ||
      ((pointer.mergeCommit === null || pointer.mergedAt === null)
        ? pointer.mergeCommit !== null || pointer.mergedAt !== null
        : !SHA_RE.test(pointer.mergeCommit ?? "") || !validEvidenceTime(pointer.mergedAt))
    ) {
      refuse(`state ${repo} final PR pointer is malformed`, "STATE_FINAL_POINTER_CONFLICT");
    }
    const receipt = finalReceipts[0];
    if (
      !receipt ||
      receipt.number !== pointer.number ||
      receipt.headSha !== pointer.headSha ||
      receipt.url !== pointer.url
    ) {
      refuse(`state ${repo} final PR pointer does not link its canonical receipt`, "STATE_FINAL_POINTER_CONFLICT");
    }
  }

  const runWorkflowKeys = new Set();
  const runIdKeys = new Set();
  for (const receipt of state.privilegedRuns) {
    const workflowKey = `${receipt?.repo}:${receipt?.workflow}`;
    const idKey = `${receipt?.repo}:${receipt?.id}`;
    const pointer = state.finalPullRequests?.[receipt?.finalPrRepo];
    if (
      !hasExactKeys(receipt, [
        "repo", "workflow", "id", "headSha", "url", "evidencePhase", "recordedAt",
        "finalPrRepo", "finalPrNumber",
      ]) ||
      !Object.hasOwn(REPOSITORIES, receipt.repo) ||
      !workflowKeys.has(workflowKey) ||
      !Number.isSafeInteger(receipt.id) ||
      receipt.id <= 0 ||
      !SHA_RE.test(receipt.headSha ?? "") ||
      typeof receipt.url !== "string" ||
      receipt.url.length === 0 ||
      receipt.evidencePhase !== "pre-queue-source" ||
      !validEvidenceTime(receipt.recordedAt) ||
      !Object.hasOwn(REPOSITORIES, receipt.finalPrRepo) ||
      !Number.isSafeInteger(receipt.finalPrNumber) ||
      receipt.finalPrNumber <= 0 ||
      pointer?.number !== receipt.finalPrNumber ||
      pointer?.headSha !== receipt.headSha ||
      runWorkflowKeys.has(workflowKey) ||
      runIdKeys.has(idKey)
    ) {
      refuse("state contains malformed, duplicate, or unlinked privileged runtime evidence", "STATE_RUNTIME_CONFLICT");
    }
    runWorkflowKeys.add(workflowKey);
    runIdKeys.add(idKey);
  }

  const deploymentKeys = new Set();
  for (const receipt of state.deployments) {
    const key = `${receipt?.repo}:${receipt?.id}`;
    const pointer = state.finalPullRequests?.[receipt?.finalPrRepo];
    if (
      !hasExactKeys(receipt, [
        "repo", "id", "sha", "url", "evidencePhase", "recordedAt", "finalPrRepo", "finalPrNumber",
      ]) ||
      !Object.hasOwn(REPOSITORIES, receipt.repo) ||
      !Number.isSafeInteger(receipt.id) ||
      receipt.id <= 0 ||
      !SHA_RE.test(receipt.sha ?? "") ||
      typeof receipt.url !== "string" ||
      receipt.url.length === 0 ||
      receipt.evidencePhase !== "post-merge" ||
      !validEvidenceTime(receipt.recordedAt) ||
      !Object.hasOwn(REPOSITORIES, receipt.finalPrRepo) ||
      !Number.isSafeInteger(receipt.finalPrNumber) ||
      receipt.finalPrNumber <= 0 ||
      pointer?.number !== receipt.finalPrNumber ||
      pointer?.mergeCommit !== receipt.sha ||
      deploymentKeys.has(key)
    ) {
      refuse("state contains malformed, duplicate, or unlinked deployment evidence", "STATE_DEPLOYMENT_CONFLICT");
    }
    deploymentKeys.add(key);
  }
  for (const mutation of state.transaction?.remoteMutations ?? []) {
    if (
      !["create-ref", "recover-create-ref"].includes(mutation.action) ||
      !Object.hasOwn(REPOSITORIES, mutation.repo) ||
      mutation.ref !== `refs/heads/${expected}` ||
      mutation.sha !== state.repositories[mutation.repo].plannedMainSha ||
      !["intent", "created", "failed", "reconciled"].includes(mutation.status)
    ) {
      refuse("state contains an invalid or destructive remote mutation record", "STATE_MUTATION_CONFLICT");
    }
  }
  for (const mutation of state.transaction.policyMutations) {
    const creation = ["create-ruleset", "recover-create-ruleset"].includes(mutation.action);
    const disabling = ["disable-ruleset", "recover-disable-ruleset"].includes(mutation.action);
    const activation = ["activate-ruleset", "recover-activate-ruleset"].includes(mutation.action);
    const policy = state.policies?.[mutation.repo]?.exactQueue;
    if (
      (!creation && !disabling && !activation) ||
      !Object.hasOwn(REPOSITORIES, mutation.repo) ||
      mutation.name !== policy?.name ||
      mutation.digest !== (creation || disabling ? policy?.stagedDigest : policy?.digest) ||
      !["intent", "created", "failed", "reconciled"].includes(mutation.status) ||
      (creation && mutation.id !== null && (!Number.isSafeInteger(mutation.id) || mutation.id <= 0)) ||
      ((disabling || activation) &&
        (!Number.isSafeInteger(mutation.id) || mutation.id <= 0 || mutation.id !== policy?.id))
    ) {
      refuse("state contains an invalid policy mutation record", "STATE_POLICY_MUTATION_CONFLICT");
    }
  }
  if (
    state.mode === "adopt-existing" &&
    (state.transaction.remoteMutations.length > 0 || state.transaction.policyMutations.length > 0)
  ) {
    refuse("adopt-existing state may never contain remote mutation intent", "STATE_ADOPTION_MUTATION_CONFLICT");
  }
  return state;
}

export class ProcessRunner {
  exec(command, args, options = {}) {
    return execFileSync(command, args, {
      cwd: options.cwd,
      encoding: "utf8",
      env: options.env,
      input: options.input,
      maxBuffer: options.maxBuffer ?? 16 * 1024 * 1024,
      stdio: options.stdio ?? [options.input === undefined ? "ignore" : "pipe", "pipe", "pipe"],
    }).trim();
  }
}

function curlConfigValue(value) {
  if (typeof value !== "string" || !value || /[\r\n\0]/.test(value)) {
    refuse("Shortcut API token is missing or malformed", "MISSING_SHORTCUT_TOKEN");
  }
  return value.replaceAll("\\", "\\\\").replaceAll('"', '\\"');
}

export class ShortcutClient {
  constructor(
    processRunner = new ProcessRunner(),
    token = process.env.SHORTCUT_API_TOKEN ?? process.env.CLAUDE_SHORTCUT_MCP_TOKEN,
  ) {
    this.process = processRunner;
    this.token = token;
  }

  api(path) {
    const token = curlConfigValue(this.token);
    const config = [
      'header = "Content-Type: application/json"',
      `header = "Shortcut-Token: ${token}"`,
      "",
    ].join("\n");
    let output;
    try {
      output = this.process.exec(
        "curl",
        ["--silent", "--show-error", "--fail-with-body", "--location", "--config", "-", `${SHORTCUT_API_ROOT}${path}`],
        { input: config },
      );
    } catch (error) {
      refuse(`Shortcut API request failed for ${path}: ${error.message}`, "SHORTCUT_API_FAILED");
    }
    try {
      return JSON.parse(output);
    } catch (error) {
      refuse(`Shortcut API returned invalid JSON for ${path}: ${error.message}`, "INVALID_SHORTCUT_RESPONSE");
    }
  }

  epic(id) {
    return this.api(`/epics/${id}`);
  }

  epicStories(id) {
    return this.api(`/epics/${id}/stories`);
  }

  workflows() {
    return this.api("/workflows");
  }
}

export class GitRunner {
  constructor(processRunner = new ProcessRunner()) {
    this.process = processRunner;
  }

  remoteSha(url, ref) {
    const output = this.process.exec("git", ["ls-remote", url, ref]);
    const lines = output.split(/\r?\n/).filter(Boolean);
    if (lines.length !== 1) return null;
    const [sha, actualRef] = lines[0].split(/\s+/);
    return SHA_RE.test(sha) && actualRef === ref ? sha : null;
  }

  cloneFeature(repo, branch, target, staging, expectedSha) {
    if (!SHA_RE.test(expectedSha ?? "")) refuse("clone requires an exact expected feature SHA", "INVALID_CLONE_HEAD");
    this.process.exec("git", ["clone", "--no-checkout", "--filter=blob:none", repo.cloneUrl, staging]);
    this.process.exec(
      "git",
      ["-C", staging, "fetch", "--no-tags", "origin", `refs/heads/${branch}:refs/remotes/origin/${branch}`],
    );
    this.process.exec(
      "git",
      ["-C", staging, "switch", "-c", branch, expectedSha],
    );
    this.process.exec("git", ["-C", staging, "branch", "--set-upstream-to", `origin/${branch}`, branch]);
    renameSync(staging, target);
  }

  inspectWorkspace(path) {
    if (!existsSync(path) || !statSync(path).isDirectory()) return null;
    try {
      return {
        origin: this.process.exec("git", ["-C", path, "remote", "get-url", "origin"]),
        branch: this.process.exec("git", ["-C", path, "branch", "--show-current"]),
        head: this.process.exec("git", ["-C", path, "rev-parse", "HEAD"]),
        dirty: this.process.exec("git", ["-C", path, "status", "--porcelain"]) !== "",
      };
    } catch {
      return { invalid: true };
    }
  }

  trackedCargoManifests(root) {
    const output = this.process.exec("git", ["-C", root, "ls-files", "-z", "--", "**/Cargo.toml", "Cargo.toml"]);
    return output.split("\0").filter(Boolean).sort();
  }

  remoteFeatureRefs(root, epic) {
    return this.process.exec("git", [
      "-C",
      root,
      "ls-remote",
      "--heads",
      "origin",
      `refs/heads/feature/${epic}-*`,
    ]);
  }
}

export function resolveRemoteFeatureBranch(epic, repoRoot, git = new GitRunner()) {
  epic = normalizeEpic(epic);
  repoRoot = assertAbsolutePath(repoRoot, "repository root");
  const output = git.remoteFeatureRefs(repoRoot, epic);
  const branches = new Map();
  for (const line of output.split(/\r?\n/).filter(Boolean)) {
    const fields = line.trim().split(/\s+/);
    if (fields.length !== 2 || !SHA_RE.test(fields[0])) {
      refuse(`origin returned a malformed feature ref line: ${JSON.stringify(line)}`, "MALFORMED_REMOTE_REF");
    }
    const refPrefix = "refs/heads/";
    if (!fields[1].startsWith(refPrefix)) {
      refuse(`origin returned a non-branch feature ref ${JSON.stringify(fields[1])}`, "MALFORMED_REMOTE_REF");
    }
    const branch = fields[1].slice(refPrefix.length);
    const parsed = assertSafeFeatureBranch(branch);
    if (parsed.epic !== epic) {
      refuse(`origin returned feature ref ${branch} for the wrong epic`, "MALFORMED_REMOTE_REF");
    }
    const previous = branches.get(branch);
    if (previous && previous !== fields[0]) {
      refuse(
        `origin returned divergent commits for canonical candidate ${branch}: ${previous} and ${fields[0]}`,
        "DIVERGENT_CANONICAL_REF",
      );
    }
    branches.set(branch, fields[0]);
  }
  if (branches.size === 0) {
    refuse(`no live feature branch exists on origin for ${epic}`, "MISSING_CANONICAL_REF");
  }
  if (branches.size !== 1) {
    refuse(
      `multiple live feature branches exist on origin for ${epic}: ${[...branches.keys()].sort().join(", ")}`,
      "DUPLICATE_CANONICAL_REF",
    );
  }
  return branches.keys().next().value;
}

export class GitHubClient {
  constructor(processRunner = new ProcessRunner()) {
    this.process = processRunner;
  }

  api(path, { method = "GET", fields = {}, headers = [], body } = {}) {
    const args = ["api", path, "--method", method];
    for (const header of headers) args.push("-H", header);
    for (const [key, value] of Object.entries(fields)) args.push("-f", `${key}=${value}`);
    if (body !== undefined) args.push("--input", "-");
    const output = this.process.exec("gh", args, {
      input: body === undefined ? undefined : JSON.stringify(body),
    });
    return output ? JSON.parse(output) : null;
  }

  apiPages(path, { headers = [] } = {}) {
    const args = ["api", path, "--method", "GET", "--paginate", "--slurp"];
    for (const header of headers) args.push("-H", header);
    const output = this.process.exec("gh", args);
    const pages = output ? JSON.parse(output) : [];
    if (!Array.isArray(pages)) {
      refuse(`GitHub returned a malformed paginated response for ${path}`, "INVALID_PAGINATION_RESPONSE");
    }
    return pages;
  }

  defaultBranch(slug) {
    return this.api(`repos/${slug}`).default_branch;
  }

  mainSha(slug) {
    const value = this.api(`repos/${slug}/git/ref/heads/main`);
    return value?.object?.sha ?? null;
  }

  featureSha(slug, branch) {
    try {
      const value = this.api(`repos/${slug}/git/ref/heads/${encodeURIComponent(branch)}`);
      return value?.object?.sha ?? null;
    } catch (error) {
      const status = error?.status ?? error?.code;
      if (status === 404 || `${error.message}`.includes("HTTP 404")) return null;
      throw error;
    }
  }

  createFeatureRef(slug, branch, sha) {
    return this.api(`repos/${slug}/git/refs`, {
      method: "POST",
      fields: { ref: `refs/heads/${branch}`, sha },
    });
  }

  rulesets(slug) {
    const path = `repos/${slug}/rulesets?includes_parents=false&per_page=100`;
    const summaries = this.apiPages(path).flatMap((page) => {
      if (!Array.isArray(page)) {
        refuse(`GitHub returned a malformed ruleset page for ${slug}`, "INVALID_PAGINATION_RESPONSE");
      }
      return page;
    });
    const seen = new Set();
    return summaries.map(({ id } = {}) => {
      if (!Number.isSafeInteger(id) || id <= 0 || seen.has(id)) {
        refuse(`GitHub returned an invalid or repeated ruleset id for ${slug}`, "INVALID_PAGINATION_RESPONSE");
      }
      seen.add(id);
      return this.api(`repos/${slug}/rulesets/${id}`);
    });
  }

  createRuleset(slug, payload) {
    return this.api(`repos/${slug}/rulesets`, { method: "POST", body: payload });
  }

  updateRuleset(slug, id, payload) {
    return this.api(`repos/${slug}/rulesets/${id}`, { method: "PUT", body: payload });
  }

  branchRules(slug, branch) {
    return this.api(`repos/${slug}/rules/branches/${encodeURIComponent(branch)}`) ?? [];
  }

  cargoManifests(slug, sha) {
    const tree = this.api(`repos/${slug}/git/trees/${sha}?recursive=1`);
    const paths = (tree?.tree ?? [])
      .filter(({ type, path }) => type === "blob" && /(^|\/)Cargo\.toml$/.test(path))
      .map(({ path }) => path)
      .sort();
    if (tree?.truncated || paths.length === 0) {
      refuse(`cannot prove the complete tracked Cargo manifest set at ${slug}@${sha}`, "MANIFEST_ENUMERATION_FAILED");
    }
    return paths.map((path) => {
      const content = this.api(`repos/${slug}/contents/${encodeURIComponent(path)}?ref=${sha}`);
      if (content?.encoding !== "base64" || typeof content.content !== "string") {
        refuse(`cannot read ${slug}/${path}@${sha}`, "MANIFEST_READ_FAILED");
      }
      return { path, text: Buffer.from(content.content.replaceAll("\n", ""), "base64").toString("utf8") };
    });
  }

  compare(slug, base, head) {
    return this.api(`repos/${slug}/compare/${base}...${head}`);
  }

  pullRequest(slug, number) {
    return this.api(`repos/${slug}/pulls/${number}`);
  }

  pullRequests(slug, { state = "all", base, head } = {}) {
    if (!new Set(["open", "closed", "all"]).has(state)) {
      refuse(`invalid pull-request state ${JSON.stringify(state)}`, "INVALID_PR_QUERY");
    }
    const query = new URLSearchParams({ state, per_page: "100", sort: "created", direction: "asc" });
    if (base) query.set("base", base);
    if (head) query.set("head", head);
    const path = `repos/${slug}/pulls?${query}`;
    const output = this.process.exec(
      "gh",
      ["api", path, "--method", "GET", "--paginate", "--jq", PULL_REQUEST_LIST_JQ],
    );
    const seen = new Set();
    return output.split(/\r?\n/).filter(Boolean).map((encoded) => {
      let pullRequest;
      try {
        if (!/^[A-Za-z0-9+/]+={0,2}$/.test(encoded)) throw new Error("invalid base64 record");
        pullRequest = JSON.parse(Buffer.from(encoded, "base64").toString("utf8"));
      } catch (error) {
        refuse(
          `GitHub returned a malformed projected pull-request record for ${slug}: ${error.message}`,
          "INVALID_PAGINATION_RESPONSE",
        );
      }
      if (
        !Number.isSafeInteger(pullRequest?.number) ||
        pullRequest.number <= 0 ||
        seen.has(pullRequest.number) ||
        !["open", "closed"].includes(pullRequest.state) ||
        typeof pullRequest.head?.ref !== "string" ||
        typeof pullRequest.base?.ref !== "string"
      ) {
        refuse(
          `GitHub returned an invalid or repeated pull-request number or shape for ${slug}`,
          "INVALID_PAGINATION_RESPONSE",
        );
      }
      seen.add(pullRequest.number);
      return pullRequest;
    });
  }

  checkRuns(slug, sha) {
    const path = `repos/${slug}/commits/${sha}/check-runs?per_page=100&filter=all`;
    return this.apiPages(path, {
      headers: ["Accept: application/vnd.github+json"],
    }).flatMap((page) => {
      if (!page || typeof page !== "object" || !Array.isArray(page.check_runs)) {
        refuse(`GitHub returned a malformed check-run page for ${slug}@${sha}`, "INVALID_PAGINATION_RESPONSE");
      }
      return page.check_runs;
    });
  }

  timeline(slug, number) {
    const path = `repos/${slug}/issues/${number}/timeline?per_page=100`;
    return this.apiPages(path, {
      headers: ["Accept: application/vnd.github+json"],
    }).flatMap((page) => {
      if (!Array.isArray(page)) {
        refuse(`GitHub returned a malformed timeline page for ${slug}#${number}`, "INVALID_PAGINATION_RESPONSE");
      }
      return page;
    });
  }

  workflowRun(slug, id) {
    return this.api(`repos/${slug}/actions/runs/${id}`);
  }

  deployment(slug, id) {
    return this.api(`repos/${slug}/deployments/${id}`);
  }

  deploymentStatuses(slug, id) {
    return this.api(`repos/${slug}/deployments/${id}/statuses?per_page=100`) ?? [];
  }
}

function onlyIncludedRef(ruleset, ref) {
  const include = ruleset?.conditions?.ref_name?.include ?? [];
  const exclude = ruleset?.conditions?.ref_name?.exclude ?? [];
  return include.length === 1 && include[0] === ref && exclude.length === 0;
}

function verifiedPolicyRecord(actual, expected, label) {
  if (!actual || !Number.isSafeInteger(actual.id) || actual.id <= 0) {
    refuse(`${label} has no immutable ruleset id`, "INVALID_RULESET_ID");
  }
  if (!sameRuleset(actual, expected)) {
    refuse(`${label} does not match the required semantic payload`, "MISMATCHED_RULESET");
  }
  return {
    id: actual.id,
    name: actual.name,
    digest: rulesetDigest(actual),
    enforcement: actual.enforcement,
  };
}

function exactPolicyRecord(actual, expected, staged, label) {
  if (!actual || !Number.isSafeInteger(actual.id) || actual.id <= 0) {
    refuse(`${label} has no immutable ruleset id`, "INVALID_RULESET_ID");
  }
  const active = sameRuleset(actual, expected);
  const disabled = sameRuleset(actual, staged);
  if (!active && !disabled) {
    refuse(`${label} does not match either required transaction payload`, "MISMATCHED_RULESET");
  }
  return {
    id: actual.id,
    name: actual.name,
    digest: rulesetDigest(actual),
    enforcement: active ? "active" : "disabled",
  };
}

export function auditFeaturePolicies(github, repo, branch, epic) {
  normalizeRepoKey(repo);
  const fixed = REPOSITORIES[repo];
  const expected = featurePolicyPayloads(repo, branch, epic);
  const all = github.rulesets(fixed.slug);
  if (!Array.isArray(all)) refuse(`${fixed.slug} ruleset listing is not an array`, "INVALID_RULESET_LIST");

  const wildcard = all.filter((ruleset) => onlyIncludedRef(ruleset, FEATURE_PATTERN));
  if (wildcard.length !== 2) {
    refuse(
      `${fixed.slug} must have exactly two wildcard feature policies; found ${wildcard.length}`,
      wildcard.length > 2 ? "DUPLICATE_RULESET" : "MISSING_RULESET",
    );
  }
  const baseCandidates = wildcard.filter(({ name }) => name === expected.wildcardBase.name);
  const deletionCandidates = wildcard.filter(({ name }) => name === expected.deletionGuard.name);
  if (baseCandidates.length !== 1 || deletionCandidates.length !== 1) {
    refuse(`${fixed.slug} wildcard policy names are missing, duplicated, or divergent`, "DUPLICATE_RULESET");
  }
  const wildcardBase = verifiedPolicyRecord(
    baseCandidates[0],
    expected.wildcardBase,
    `${fixed.slug} wildcard base policy`,
  );
  const deletionGuard = verifiedPolicyRecord(
    deletionCandidates[0],
    expected.deletionGuard,
    `${fixed.slug} wildcard deletion guard`,
  );

  const exactRef = `refs/heads/${branch}`;
  const exactCandidates = all.filter((ruleset) => onlyIncludedRef(ruleset, exactRef));
  const namedExactCandidates = all.filter(({ name }) => name === expected.exactQueue.name);
  if (
    namedExactCandidates.length > 1 ||
    (namedExactCandidates.length === 1 && !onlyIncludedRef(namedExactCandidates[0], exactRef))
  ) {
    refuse(
      `${fixed.slug} has a duplicate or divergent ${expected.exactQueue.name} policy`,
      namedExactCandidates.length > 1 ? "DUPLICATE_RULESET" : "MISMATCHED_RULESET",
    );
  }
  if (exactCandidates.length > 1) {
    refuse(`${fixed.slug} has duplicate exact queue policies for ${branch}`, "DUPLICATE_RULESET");
  }
  let exactQueue = null;
  if (exactCandidates.length === 1) {
    exactQueue = exactPolicyRecord(
      exactCandidates[0],
      expected.exactQueue,
      expected.stagedExactQueue,
      `${fixed.slug} exact feature queue`,
    );
  }
  return {
    wildcardBase,
    deletionGuard,
    exactQueue,
    expectedExactQueue: {
      name: expected.exactQueue.name,
      digest: rulesetDigest(expected.exactQueue),
      payload: expected.exactQueue,
      stagedDigest: rulesetDigest(expected.stagedExactQueue),
      stagedPayload: expected.stagedExactQueue,
    },
  };
}

function stripTomlComments(text) {
  let output = "";
  let quote = null;
  for (let index = 0; index < text.length; index += 1) {
    const character = text[index];
    if (quote) {
      output += character;
      if ((quote === '"' || quote === '"""') && character === "\\") {
        if (index + 1 < text.length) output += text[++index];
      } else if (quote.length === 3 && text.slice(index, index + 3) === quote) {
        output += quote.slice(1);
        index += 2;
        quote = null;
      } else if (quote.length === 1 && character === quote) {
        quote = null;
      }
      continue;
    }
    const candidate = text.slice(index, index + 3);
    if (candidate === '"""' || candidate === "'''") {
      quote = candidate;
      output += candidate;
      index += 2;
    } else if (character === '"' || character === "'") {
      quote = character;
      output += character;
    } else if (character === "#") {
      while (index < text.length && text[index] !== "\n") index += 1;
      if (index < text.length) output += "\n";
    } else {
      output += character;
    }
  }
  return output;
}

function tomlStructure(value) {
  let braces = 0;
  let brackets = 0;
  let quote = null;
  for (let index = 0; index < value.length; index += 1) {
    const character = value[index];
    if (quote) {
      if ((quote === '"' || quote === '"""') && character === "\\") {
        index += 1;
      } else if (quote.length === 3 && value.slice(index, index + 3) === quote) {
        index += 2;
        quote = null;
      } else if (quote.length === 1 && character === quote) {
        quote = null;
      }
      continue;
    }
    const candidate = value.slice(index, index + 3);
    if (candidate === '"""' || candidate === "'''") {
      quote = candidate;
      index += 2;
    } else if (character === '"' || character === "'") {
      quote = character;
    } else if (character === "{") {
      braces += 1;
    } else if (character === "}") {
      braces -= 1;
    } else if (character === "[") {
      brackets += 1;
    } else if (character === "]") {
      brackets -= 1;
    }
  }
  return { balanced: quote === null && braces === 0 && brackets === 0, braces, brackets, quote };
}

function tomlStatements(text, path) {
  const statements = [];
  let pending = "";
  let startLine = 0;
  for (const [index, line] of stripTomlComments(text).split(/\r?\n/).entries()) {
    if (!pending && line.trim() === "") continue;
    if (!pending) startLine = index + 1;
    pending += `${pending ? "\n" : ""}${line.trim()}`;
    const structure = tomlStructure(pending);
    if (structure.braces < 0 || structure.brackets < 0) {
      refuse(`${path}:${startLine} has malformed TOML structure`, "INVALID_CARGO_MANIFEST");
    }
    if (structure.balanced) {
      statements.push({ text: pending.trim(), line: startLine });
      pending = "";
    }
  }
  if (pending) refuse(`${path}:${startLine} has an unterminated TOML statement`, "INVALID_CARGO_MANIFEST");
  return statements;
}

function splitTomlTopLevel(value, delimiter) {
  const parts = [];
  let start = 0;
  let braces = 0;
  let brackets = 0;
  let quote = null;
  for (let index = 0; index < value.length; index += 1) {
    const character = value[index];
    if (quote) {
      if (quote === '"' && character === "\\") index += 1;
      else if (character === quote) quote = null;
      continue;
    }
    if (character === '"' || character === "'") quote = character;
    else if (character === "{") braces += 1;
    else if (character === "}") braces -= 1;
    else if (character === "[") brackets += 1;
    else if (character === "]") brackets -= 1;
    else if (character === delimiter && braces === 0 && brackets === 0) {
      parts.push(value.slice(start, index));
      start = index + 1;
    }
  }
  parts.push(value.slice(start));
  return parts;
}

function findTomlEquals(value) {
  let quote = null;
  let braces = 0;
  let brackets = 0;
  for (let index = 0; index < value.length; index += 1) {
    const character = value[index];
    if (quote) {
      if (quote === '"' && character === "\\") index += 1;
      else if (character === quote) quote = null;
      continue;
    }
    if (character === '"' || character === "'") quote = character;
    else if (character === "{") braces += 1;
    else if (character === "}") braces -= 1;
    else if (character === "[") brackets += 1;
    else if (character === "]") brackets -= 1;
    else if (character === "=" && braces === 0 && brackets === 0) return index;
  }
  return -1;
}

function parseTomlString(value) {
  value = value.trim();
  if (value.startsWith('"') && value.endsWith('"')) {
    try {
      return JSON.parse(value);
    } catch {
      return null;
    }
  }
  if (value.startsWith("'") && value.endsWith("'")) return value.slice(1, -1);
  return null;
}

function parseTomlKeyPath(value, path, line) {
  const segments = splitTomlTopLevel(value, ".").map((part) => {
    const trimmed = part.trim();
    const quoted = parseTomlString(trimmed);
    if (quoted !== null) return quoted;
    if (!/^[A-Za-z0-9_-]+$/.test(trimmed)) {
      refuse(`${path}:${line} has an unsupported TOML key`, "INVALID_CARGO_MANIFEST");
    }
    return trimmed;
  });
  if (segments.length === 0 || segments.some((segment) => !segment)) {
    refuse(`${path}:${line} has an empty TOML key`, "INVALID_CARGO_MANIFEST");
  }
  return segments;
}

function parseInlineTomlTable(value, path, line) {
  value = value.trim();
  if (!value.startsWith("{") || !value.endsWith("}")) return null;
  const fields = new Map();
  for (const item of splitTomlTopLevel(value.slice(1, -1), ",")) {
    if (!item.trim()) continue;
    const equals = findTomlEquals(item);
    if (equals < 0) refuse(`${path}:${line} has a malformed inline dependency`, "INVALID_CARGO_MANIFEST");
    const key = parseTomlKeyPath(item.slice(0, equals), path, line);
    if (key.length !== 1 || fields.has(key[0])) {
      refuse(`${path}:${line} has duplicate or nested inline dependency keys`, "INVALID_CARGO_MANIFEST");
    }
    fields.set(key[0], item.slice(equals + 1).trim());
  }
  return fields;
}

const DEPENDENCY_TABLES = new Set(["dependencies", "dev-dependencies", "build-dependencies"]);

function isCargoDependencyCollection(table) {
  const kind = table.at(-1);
  if (!DEPENDENCY_TABLES.has(kind)) return false;
  return (
    table.length === 1 ||
    (table.length === 2 && table[0] === "workspace") ||
    (table.length === 3 && table[0] === "target")
  );
}

function isCargoPatchCollection(table) {
  return (table.length === 2 && table[0] === "patch") || (table.length === 1 && table[0] === "replace");
}

function dependencyLocation(table, keys) {
  const fullPath = [...table, ...keys];
  const collection = fullPath.slice(0, -1);
  if (isCargoDependencyCollection(collection) || isCargoPatchCollection(collection)) {
    return { id: JSON.stringify(fullPath), field: null };
  }
  const detailCollection = fullPath.slice(0, -2);
  if (isCargoDependencyCollection(detailCollection) || isCargoPatchCollection(detailCollection)) {
    return { id: JSON.stringify(fullPath.slice(0, -1)), field: fullPath.at(-1) };
  }
  return null;
}

function isGitHubHost(value) {
  try {
    const hostname = decodeURIComponent(value).toLowerCase().replace(/\.+$/, "");
    return ["github.com", "www.github.com", "ssh.github.com"].includes(hostname);
  } catch {
    return false;
  }
}

function classifyInferenceUrl(value) {
  if (value === INFERENCE_URL) return "canonical";
  // Git accepts both relative (`host:owner/repo`) and absolute
  // (`host:/owner/repo`) SCP-style paths. GitHub resolves both spellings to the
  // same repository, so the optional leading slash must not hide an inference
  // dependency from the canonical-URL check.
  const scp = /^(?:[^@/:]+@)?([^/:]+)(?::\/?|\/)([^/?#]+)\/([^/?#]+)(.*)$/.exec(value);
  if (scp && isGitHubHost(scp[1])) {
    try {
      const owner = decodeURIComponent(scp[2]).toLowerCase();
      const repository = decodeURIComponent(scp[3]).replace(/\.git$/i, "").toLowerCase();
      if (owner === "sceneworks" && repository === "inference") return "alternate";
    } catch {
      return null;
    }
  }
  try {
    const parsed = new URL(value.replace(/^git\+/i, ""));
    if (!isGitHubHost(parsed.hostname)) return null;
    const parts = decodeURIComponent(parsed.pathname).split("/").filter(Boolean);
    if (
      parts.length >= 2 &&
      parts[0].toLowerCase() === "sceneworks" &&
      parts[1].replace(/\.git$/i, "").toLowerCase() === "inference"
    ) {
      return "alternate";
    }
  } catch {
    return null;
  }
  return null;
}

function cargoDependencyRecords(path, text) {
  const records = new Map();
  let table = [];
  for (const statement of tomlStatements(text, path)) {
    if (statement.text.startsWith("[") && statement.text.endsWith("]")) {
      const arrayTable = statement.text.startsWith("[[") && statement.text.endsWith("]]");
      const inner = statement.text.slice(arrayTable ? 2 : 1, arrayTable ? -2 : -1);
      table = parseTomlKeyPath(inner, path, statement.line);
      continue;
    }
    const equals = findTomlEquals(statement.text);
    if (equals < 0) continue;
    const keys = parseTomlKeyPath(statement.text.slice(0, equals), path, statement.line);
    const location = dependencyLocation(table, keys);
    if (!location) continue;
    const value = statement.text.slice(equals + 1).trim();
    const record = records.get(location.id) ?? { path, line: statement.line, fields: new Map() };
    if (location.field === null) {
      const inline = parseInlineTomlTable(value, path, statement.line);
      if (!inline) continue;
      for (const [field, fieldValue] of inline) {
        if (record.fields.has(field)) {
          refuse(`${path}:${statement.line} repeats dependency field ${field}`, "INVALID_CARGO_MANIFEST");
        }
        record.fields.set(field, fieldValue);
      }
    } else {
      if (record.fields.has(location.field)) {
        refuse(`${path}:${statement.line} repeats dependency field ${location.field}`, "INVALID_CARGO_MANIFEST");
      }
      record.fields.set(location.field, value);
    }
    records.set(location.id, record);
  }
  return [...records.values()];
}

export function extractInferencePin(manifests) {
  const pins = [];
  const sources = [];
  for (const { path, text } of manifests) {
    for (const record of cargoDependencyRecords(path, text)) {
      if (!record.fields.has("git")) continue;
      const gitUrl = parseTomlString(record.fields.get("git"));
      if (gitUrl === null) {
        refuse(`${path}:${record.line} dependency git must be a TOML string`, "INVALID_CARGO_MANIFEST");
      }
      const inferenceUrl = classifyInferenceUrl(gitUrl);
      if (inferenceUrl === null) continue;
      if (inferenceUrl !== "canonical") {
        refuse(
          `${path}:${record.line} must use canonical inference URL ${INFERENCE_URL}`,
          "INVALID_INFERENCE_URL",
        );
      }
      const rev = parseTomlString(record.fields.get("rev") ?? "");
      if (!rev || !SHA_RE.test(rev) || record.fields.has("tag") || record.fields.has("branch")) {
        refuse(
          `${path}:${record.line} must pin SceneWorks/inference by one exact 40-hex rev without branch or tag`,
          "INVALID_INFERENCE_PIN",
        );
      }
      pins.push(rev);
      sources.push({ path, line: record.line });
    }
  }
  if (pins.length === 0) {
    refuse("no tracked Cargo manifest pins SceneWorks/inference", "MISSING_INFERENCE_PIN");
  }
  const unique = [...new Set(pins)];
  if (unique.length !== 1) {
    refuse(`tracked Cargo manifests contain inconsistent inference pins: ${unique.join(", ")}`, "INCONSISTENT_INFERENCE_PIN");
  }
  return { sha: unique[0], sources };
}

export function isReachableFromMain(github, pin, mainSha) {
  if (!SHA_RE.test(pin) || !SHA_RE.test(mainSha)) return false;
  const comparison = github.compare(REPOSITORIES.inference.slug, pin, mainSha);
  return comparison?.status === "ahead" || comparison?.status === "identical";
}

function nowIso(clock) {
  return clock().toISOString();
}

function storyInventoryDigest(stories) {
  return createHash("sha256").update(JSON.stringify(normalizeStories(stories))).digest("hex");
}

export function inspectShortcutEpic(shortcut, epic, expectedStories) {
  epic = normalizeEpic(epic);
  const requestedStories = expectedStories === undefined ? null : normalizeStories(expectedStories);
  const epicId = Number(epic.slice(3));
  const liveEpic = shortcut.epic(epicId);
  if (
    !liveEpic ||
    liveEpic.id !== epicId ||
    typeof liveEpic.name !== "string" ||
    typeof liveEpic.app_url !== "string" ||
    typeof liveEpic.updated_at !== "string" ||
    !Number.isFinite(evidenceTimestamp(liveEpic.updated_at)) ||
    typeof liveEpic.archived !== "boolean" ||
    typeof liveEpic.completed !== "boolean"
  ) {
    refuse(`Shortcut returned malformed epic evidence for ${epic}`, "INVALID_SHORTCUT_EPIC");
  }
  const liveStories = shortcut.epicStories(epicId);
  if (!Array.isArray(liveStories)) {
    refuse(`Shortcut returned a malformed story inventory for ${epic}`, "INVALID_SHORTCUT_STORIES");
  }
  if (
    !Number.isSafeInteger(liveEpic.stats?.num_stories_total) ||
    liveEpic.stats.num_stories_total !== liveStories.length
  ) {
    refuse(
      `Shortcut epic ${epic} reports ${liveEpic.stats?.num_stories_total} stories but returned ${liveStories.length}`,
      "INCOMPLETE_SHORTCUT_INVENTORY",
    );
  }
  const workflows = shortcut.workflows();
  if (!Array.isArray(workflows)) {
    refuse("Shortcut returned a malformed workflow inventory", "INVALID_SHORTCUT_WORKFLOWS");
  }
  const workflowStates = new Map();
  for (const workflow of workflows) {
    if (!Number.isSafeInteger(workflow?.id) || workflow.id <= 0 || !Array.isArray(workflow.states)) {
      refuse("Shortcut returned a malformed workflow", "INVALID_SHORTCUT_WORKFLOWS");
    }
    for (const state of workflow.states) {
      const workflowState = {
        id: state?.id,
        name: state?.name,
        type: state?.type,
        workflowId: workflow.id,
      };
      if (
        !validShortcutWorkflowState(workflowState) ||
        workflowStates.has(state.id)
      ) {
        refuse("Shortcut returned a malformed or duplicate workflow state", "INVALID_SHORTCUT_WORKFLOWS");
      }
      workflowStates.set(state.id, workflowState);
    }
  }

  const seen = new Set();
  const stories = liveStories.map((story) => {
    if (
      !Number.isSafeInteger(story?.id) ||
      story.id <= 0 ||
      seen.has(story.id) ||
      story.epic_id !== epicId ||
      typeof story.name !== "string" ||
      typeof story.app_url !== "string" ||
      typeof story.updated_at !== "string" ||
      !Number.isFinite(evidenceTimestamp(story.updated_at)) ||
      !Number.isSafeInteger(story.workflow_state_id) ||
      story.workflow_state_id <= 0 ||
      typeof story.completed !== "boolean" ||
      typeof story.blocked !== "boolean" ||
      typeof story.blocker !== "boolean" ||
      typeof story.archived !== "boolean"
    ) {
      refuse(`Shortcut returned a malformed or duplicate story in ${epic}`, "INVALID_SHORTCUT_STORIES");
    }
    seen.add(story.id);
    const workflowState = workflowStates.get(story.workflow_state_id);
    if (!workflowState) {
      refuse(`Shortcut story sc-${story.id} references an unknown workflow state`, "UNKNOWN_SHORTCUT_STATE");
    }
    const done = story.completed && workflowState.type === "done" && !story.archived;
    return {
      id: `sc-${story.id}`,
      name: story.name,
      appUrl: story.app_url ?? null,
      updatedAt: story.updated_at ?? null,
      workflowState,
      completed: story.completed,
      blocked: story.blocked,
      blocker: story.blocker,
      archived: story.archived,
      done,
    };
  }).sort((left, right) => Number(left.id.slice(3)) - Number(right.id.slice(3)));
  if (stories.length === 0) refuse(`Shortcut epic ${epic} has no stories`, "EMPTY_SHORTCUT_EPIC");

  const liveIds = stories.map(({ id }) => id);
  expectedStories = requestedStories ?? liveIds;
  const missingFromEpic = expectedStories.filter((story) => !seen.has(Number(story.slice(3))));
  const unexpectedInEpic = liveIds.filter((story) => !expectedStories.includes(story));
  const inventoryMatches = missingFromEpic.length === 0 && unexpectedInEpic.length === 0;
  const allStoriesDone = stories.every(({ done }) => done);
  const noBlockedStories = stories.every(({ blocked }) => !blocked);
  const noUnresolvedBlockers = stories.every(({ blocker, done }) => !blocker || done);
  const epicNotArchived = !liveEpic.archived;
  return {
    epic: {
      id: epic,
      numericId: epicId,
      name: liveEpic.name,
      appUrl: liveEpic.app_url ?? null,
      state: liveEpic.state ?? null,
      completed: liveEpic.completed,
      archived: liveEpic.archived,
      updatedAt: liveEpic.updated_at ?? null,
      reportedStoryCount: liveEpic.stats.num_stories_total,
    },
    inventory: {
      expected: expectedStories,
      live: liveIds,
      missingFromEpic,
      unexpectedInEpic,
      matches: inventoryMatches,
      digest: storyInventoryDigest(liveIds),
    },
    stories: Object.fromEntries(stories.map((story) => [story.id, story])),
    workflowStates: Object.fromEntries(
      [...workflowStates.entries()]
        .sort(([left], [right]) => left - right)
        .map(([id, state]) => [id, state]),
    ),
    gates: {
      inventoryMatches,
      allStoriesDone,
      noBlockedStories,
      noUnresolvedBlockers,
      epicNotArchived,
    },
    ok:
      inventoryMatches &&
      allStoriesDone &&
      noBlockedStories &&
      noUnresolvedBlockers &&
      epicNotArchived,
  };
}

function plannedShortcutRecord(snapshot, inventoryFrozenAt = null) {
  return {
    epicId: snapshot.epic.id,
    epicUrl: snapshot.epic.appUrl,
    epicUpdatedAt: snapshot.epic.updatedAt,
    inventoryDigest: snapshot.inventory.digest,
    workflowStates: snapshot.workflowStates,
    inventoryFrozenAt,
    refreshes: [],
  };
}

function createPlanUnlocked(
  {
    epic,
    slug,
    stories,
    workspaceParent,
    statePath,
    deploymentRequired = false,
    privilegedWorkflows,
    adoptExisting = false,
  },
  { github = new GitHubClient(), shortcut = new ShortcutClient(), clock = () => new Date() } = {},
) {
  epic = normalizeEpic(epic);
  slug = normalizeSlug(slug);
  stories = normalizeStories(stories);
  if (typeof adoptExisting !== "boolean") refuse("adoptExisting must be boolean", "INVALID_PLAN_MODE");
  const mode = adoptExisting ? "adopt-existing" : "create";
  const branch = featureBranch(epic, slug);
  assertSafeFeatureBranch(branch);
  workspaceParent = assertEmptyWorkspaceParent(workspaceParent, statePath);
  statePath = assertAbsolutePath(statePath, "state path");
  if (existsSync(statePath)) refuse(`state path already exists: ${statePath}`, "STATE_EXISTS");

  const shortcutSnapshot = inspectShortcutEpic(shortcut, epic, stories);
  if (!shortcutSnapshot.inventory.matches) {
    refuse(
      `expected stories do not exactly match live Shortcut epic ${epic}; missing ${shortcutSnapshot.inventory.missingFromEpic.join(", ") || "none"}; unexpected ${shortcutSnapshot.inventory.unexpectedInEpic.join(", ") || "none"}`,
      "SHORTCUT_INVENTORY_CONFLICT",
    );
  }
  if (!shortcutSnapshot.gates.epicNotArchived) {
    refuse(`Shortcut epic ${epic} is archived`, "ARCHIVED_SHORTCUT_EPIC");
  }

  const createdAt = nowIso(clock);
  const existingRefs = Object.fromEntries(
    Object.entries(REPOSITORIES).map(([key, fixed]) => [key, github.featureSha(fixed.slug, branch)]),
  );
  const existingRefCount = Object.values(existingRefs).filter(Boolean).length;
  if (mode === "adopt-existing" && existingRefCount !== Object.keys(REPOSITORIES).length) {
    refuse(
      `adopt-existing requires both mirrored feature refs; found ${existingRefCount}`,
      "ADOPTION_REQUIRES_REF_PAIR",
    );
  }
  if (mode === "create" && existingRefCount !== 0) {
    refuse(
      "feature refs already exist; re-run plan with explicit --adopt-existing after verifying the protected train",
      "EXISTING_TRAIN_REQUIRES_ADOPTION",
    );
  }

  const repositories = {};
  for (const [key, fixed] of Object.entries(REPOSITORIES)) {
    const defaultBranch = github.defaultBranch(fixed.slug);
    if (defaultBranch !== "main") {
      refuse(`${fixed.slug} live default branch is ${JSON.stringify(defaultBranch)}, not main`, "NON_MAIN_DEFAULT");
    }
    const mainSha = github.mainSha(fixed.slug);
    if (!SHA_RE.test(mainSha ?? "")) {
      refuse(`cannot resolve exact ${fixed.slug} main SHA`, "MISSING_MAIN_SHA");
    }
    const adoptedFeatureSha = mode === "adopt-existing" ? existingRefs[key] : null;
    if (adoptedFeatureSha) {
      const ancestry = github.compare(fixed.slug, mainSha, adoptedFeatureSha);
      if (!new Set(["ahead", "identical"]).has(ancestry?.status)) {
        refuse(
          `${fixed.slug} adopted feature ${adoptedFeatureSha} does not contain planned main ${mainSha}`,
          "ADOPTED_FEATURE_MISSING_MAIN",
        );
      }
    }
    repositories[key] = {
      slug: fixed.slug,
      cloneUrl: fixed.cloneUrl,
      plannedMainSha: mainSha,
      adoptedFeatureSha,
      adoptedRequiredChecks: [],
      featureBranch: branch,
      remoteRef: adoptedFeatureSha,
      workspace: {
        path: join(workspaceParent, fixed.directory),
        status: "pending",
        head: null,
        activeStagingPath: null,
        failedStagingPaths: [],
      },
    };
  }

  if (mode === "create") {
    assertRequiredMainChecks(github, repositories, REQUIRED_CONTEXTS);
  } else {
    for (const [key, fixed] of Object.entries(REPOSITORIES)) {
      repositories[key].adoptedRequiredChecks = trustedRequiredCheckReceipts(
        github,
        fixed.slug,
        repositories[key].adoptedFeatureSha,
        REQUIRED_CONTEXTS[key],
        "adopted feature",
      );
    }
  }

  const pinRevision = mode === "adopt-existing"
    ? repositories.sceneworks.adoptedFeatureSha
    : repositories.sceneworks.plannedMainSha;
  const inferencePin = extractInferencePin(
    github.cargoManifests(REPOSITORIES.sceneworks.slug, pinRevision),
  );
  const inferenceTargets = mode === "adopt-existing"
    ? [repositories.inference.plannedMainSha, repositories.inference.adoptedFeatureSha]
    : [repositories.inference.plannedMainSha];
  if (inferenceTargets.some((target) => !isReachableFromMain(github, inferencePin.sha, target))) {
    refuse(
      `SceneWorks ${mode === "adopt-existing" ? "adopted feature" : "main"} pin ${inferencePin.sha} is not reachable from the paired inference baseline`,
      "UNREACHABLE_INFERENCE_PIN",
    );
  }

  const policyAudits = Object.fromEntries(
    Object.keys(REPOSITORIES).map((key) => [key, auditFeaturePolicies(github, key, branch, epic)]),
  );
  const existingExactQueues = Object.values(policyAudits).filter(({ exactQueue }) => exactQueue).length;
  if (existingExactQueues === 1) {
    refuse("exactly one repository already has the planned exact queue policy", "PARTIAL_POLICY_PAIR");
  }
  if (Object.values(policyAudits).some(({ exactQueue }) => exactQueue?.enforcement === "disabled")) {
    refuse(
      "a disabled exact queue exists outside a recorded feature-epic transaction",
      "UNOWNED_DISABLED_POLICY",
    );
  }
  if (mode === "create" && existingExactQueues !== 0) {
    refuse(
      "exact queue policies already exist; explicit adopt-existing mode is required",
      "EXISTING_TRAIN_REQUIRES_ADOPTION",
    );
  }
  if (
    mode === "adopt-existing" &&
    Object.values(policyAudits).some(({ exactQueue }) => exactQueue?.enforcement !== "active")
  ) {
    refuse("adopt-existing requires both exact queue policies active", "ADOPTION_POLICY_MISMATCH");
  }
  if (mode === "adopt-existing") {
    for (const key of Object.keys(REPOSITORIES)) {
      verifyEffectiveFeaturePolicies(github, key, branch, policyAudits[key]);
    }
  }
  const policies = Object.fromEntries(
    Object.entries(policyAudits).map(([key, audit]) => [
      key,
      {
        wildcardBase: audit.wildcardBase,
        deletionGuard: audit.deletionGuard,
        exactQueue: {
          ...audit.expectedExactQueue,
          id: audit.exactQueue?.id ?? null,
          status: audit.exactQueue?.enforcement ?? "pending",
        },
      },
    ]),
  );

  const requestedPrivileged = [...DEFAULT_PRIVILEGED_WORKFLOWS, ...(privilegedWorkflows ?? [])];
  const uniquePrivileged = [...new Map(
    requestedPrivileged.map(({ repo, workflow }) => {
      normalizeRepoKey(repo);
      if (!/^[A-Za-z0-9][A-Za-z0-9._-]*\.ya?ml$/.test(workflow ?? "")) {
        refuse(`invalid privileged workflow name ${JSON.stringify(workflow)}`, "INVALID_WORKFLOW");
      }
      return [`${repo}:${workflow}`, { repo, workflow }];
    }),
  ).values()];
  const state = {
    schemaVersion: STATE_VERSION,
    mode,
    epic,
    slug,
    featureBranch: branch,
    expectedStories: stories,
    workspaceParent,
    deploymentRequired: Boolean(deploymentRequired),
    shortcut: plannedShortcutRecord(
      shortcutSnapshot,
      mode === "adopt-existing" ? createdAt : null,
    ),
    createdAt,
    updatedAt: createdAt,
    phase: "planned",
    repositories,
    policies,
    inferencePin,
    requiredContexts: {
      sceneworks: [...REQUIRED_CONTEXTS.sceneworks],
      inference: [...REQUIRED_CONTEXTS.inference],
    },
    privilegedWorkflows: uniquePrivileged,
    pullRequests: [],
    adoptedPullRequests: [],
    finalPullRequests: { sceneworks: null, inference: null },
    privilegedRuns: [],
    deployments: [],
    transaction: {
      policyMutations: [],
      remoteMutations: [],
      recoveryReason: null,
      completedAt: null,
    },
  };
  validateState(state);
  writeJsonAtomic(statePath, state);
  return state;
}

export function createPlan(options, dependencies = {}) {
  const statePath = assertAbsolutePath(options?.statePath, "state path");
  const clock = dependencies.clock ?? (() => new Date());
  return withStateLock(
    statePath,
    { operation: "plan", clock, staleAfterMs: dependencies.lockStaleAfterMs },
    () => createPlanUnlocked({ ...options, statePath }, dependencies),
  );
}

function workflowStateDiff(previous, next) {
  const ids = [...new Set([...Object.keys(previous), ...Object.keys(next)])]
    .map(Number)
    .sort((left, right) => left - right);
  return ids.flatMap((id) => {
    const before = previous[id] ?? null;
    const after = next[id] ?? null;
    return JSON.stringify(before) === JSON.stringify(after) ? [] : [{ id, before, after }];
  });
}

function liveFinalIntegrationPullRequests(github, state) {
  return Object.entries(REPOSITORIES).flatMap(([repo, fixed]) => {
    const owner = fixed.slug.split("/")[0];
    return github
      .pullRequests(fixed.slug, { state: "all", head: `${owner}:${state.featureBranch}` })
      .map(({ number }) => github.pullRequest(fixed.slug, number))
      .map((pullRequest) => {
        const { number, state: prState, html_url, head, base } = pullRequest ?? {};
        if (
          !Number.isSafeInteger(number) ||
          !["open", "closed"].includes(prState) ||
          head?.ref !== state.featureBranch ||
          typeof base?.ref !== "string" ||
          head?.repo?.full_name !== fixed.slug ||
          base?.repo?.full_name !== fixed.slug
        ) {
          refuse(
            `${fixed.slug}#${number} returned malformed final-integration PR evidence`,
            "INVALID_PULL_REQUEST",
          );
        }
        return {
          repo,
          number,
          state: prState,
          base: base.ref,
          url: html_url ?? null,
        };
      });
  });
}

function refreshStoryInventoryUnlocked(
  statePath,
  { apply = false } = {},
  { github = new GitHubClient(), shortcut = new ShortcutClient(), clock = () => new Date() } = {},
) {
  if (!apply) refuse("refresh-stories is read-only unless --apply is explicit", "APPLY_REQUIRED");
  const state = loadState(statePath);
  const finalPrEvidence = state.pullRequests.filter(({ kind }) => kind === "final");
  const liveFinalPrs = liveFinalIntegrationPullRequests(github, state);
  if (
    state.shortcut.inventoryFrozenAt !== null ||
    Object.values(state.finalPullRequests).some(Boolean) ||
    finalPrEvidence.length > 0 ||
    liveFinalPrs.length > 0
  ) {
    refuse(
      "Shortcut inventory is frozen because final-integration PR intent or evidence exists",
      "SHORTCUT_INVENTORY_FROZEN",
    );
  }

  const snapshot = inspectShortcutEpic(shortcut, state.epic, undefined);
  if (!snapshot.gates.epicNotArchived) {
    refuse(`Shortcut epic ${state.epic} is archived`, "ARCHIVED_SHORTCUT_EPIC");
  }
  const previous = state.expectedStories;
  const next = snapshot.inventory.live;
  const added = next.filter((story) => !previous.includes(story));
  const removed = previous.filter((story) => !next.includes(story));
  if (removed.length > 0) {
    refuse(
      `Shortcut inventory refresh is additive-only; removed or cross-epic stories: ${removed.join(", ")}`,
      "SHORTCUT_STORY_REMOVAL",
    );
  }
  const recordedStories = new Set(
    state.pullRequests.map(({ story }) => story).filter(Boolean),
  );
  const missingRecorded = [...recordedStories].filter((story) => !next.includes(story));
  if (missingRecorded.length > 0) {
    refuse(
      `Shortcut inventory no longer contains recorded stories: ${missingRecorded.join(", ")}`,
      "SHORTCUT_STORY_REMOVAL",
    );
  }
  const workflowStateChanges = workflowStateDiff(
    state.shortcut.workflowStates,
    snapshot.workflowStates,
  );
  const refresh = {
    at: nowIso(clock),
    previousDigest: state.shortcut.inventoryDigest,
    nextDigest: snapshot.inventory.digest,
    added,
    removed,
    workflowStateChanges,
    epicUpdatedAt: snapshot.epic.updatedAt,
  };
  state.expectedStories = next;
  state.shortcut.epicUrl = snapshot.epic.appUrl;
  state.shortcut.epicUpdatedAt = snapshot.epic.updatedAt;
  state.shortcut.inventoryDigest = snapshot.inventory.digest;
  state.shortcut.workflowStates = snapshot.workflowStates;
  state.shortcut.refreshes.push(refresh);
  saveState(statePath, state, clock);
  return { state, refresh };
}

export function refreshStoryInventory(statePath, options = {}, dependencies = {}) {
  if (!options.apply) refuse("refresh-stories is read-only unless --apply is explicit", "APPLY_REQUIRED");
  statePath = assertAbsolutePath(statePath, "state path");
  const clock = dependencies.clock ?? (() => new Date());
  return withStateLock(
    statePath,
    { operation: "refresh-stories", clock, staleAfterMs: dependencies.lockStaleAfterMs },
    () => refreshStoryInventoryUnlocked(statePath, options, dependencies),
  );
}

function saveState(statePath, state, clock) {
  state.updatedAt = nowIso(clock);
  validateState(state);
  writeJsonAtomic(statePath, state);
}

function liveRefs(state, github) {
  return Object.fromEntries(
    Object.entries(REPOSITORIES).map(([key, fixed]) => [
      key,
      github.featureSha(fixed.slug, state.featureBranch),
    ]),
  );
}

function livePolicyAudits(state, github) {
  return Object.fromEntries(
    Object.keys(REPOSITORIES).map((key) => [
      key,
      auditFeaturePolicies(github, key, state.featureBranch, state.epic),
    ]),
  );
}

function assertWildcardPoliciesUnchanged(state, audits) {
  for (const key of Object.keys(REPOSITORIES)) {
    for (const policyKey of ["wildcardBase", "deletionGuard"]) {
      const recorded = state.policies[key][policyKey];
      const live = audits[key][policyKey];
      if (recorded.id !== live.id || recorded.digest !== live.digest) {
        refuse(
          `${REPOSITORIES[key].slug} ${policyKey} changed since plan`,
          "STALE_WILDCARD_POLICY",
        );
      }
    }
  }
}

function ownsPolicy(state, key, live) {
  if (!live) return false;
  if (state.policies[key].exactQueue.id !== null) {
    if (state.policies[key].exactQueue.id !== live.id) return false;
    return live.enforcement === "active" || policyWasCreatedByTransaction(state, key, live.id);
  }
  return state.transaction.policyMutations.some(
    (mutation) =>
      mutation.repo === key &&
      mutation.digest === live.digest &&
      mutation.id === null &&
      ["create-ruleset", "recover-create-ruleset"].includes(mutation.action) &&
      ["intent", "failed"].includes(mutation.status),
  );
}

function reconcilePolicyRecord(state, key, live, reconcileAttempts = true) {
  state.policies[key].exactQueue.id = live.id;
  state.policies[key].exactQueue.status = live.enforcement;
  if (reconcileAttempts) {
    for (const mutation of state.transaction.policyMutations) {
      if (
        mutation.repo === key &&
        mutation.digest === live.digest &&
        (mutation.id === null || mutation.id === live.id) &&
        mutation.status !== "created"
      ) {
        mutation.id = live.id;
        mutation.status = "reconciled";
      }
    }
  }
}

function installExactQueue(statePath, state, key, action, github, clock) {
  const policy = state.policies[key].exactQueue;
  const mutation = {
    action,
    repo: key,
    name: policy.name,
    digest: policy.stagedDigest,
    id: null,
    at: nowIso(clock),
    status: "intent",
  };
  state.transaction.policyMutations.push(mutation);
  saveState(statePath, state, clock);
  let created;
  try {
    created = github.createRuleset(REPOSITORIES[key].slug, policy.stagedPayload);
  } catch (error) {
    mutation.status = "failed";
    policy.status = "failed";
    markRecovery(statePath, state, `exact queue creation failed for ${REPOSITORIES[key].slug}: ${error.message}`, clock);
    throw error;
  }
  if (!Number.isSafeInteger(created?.id) || created.id <= 0) {
    markRecovery(statePath, state, `exact queue creation returned no id for ${REPOSITORIES[key].slug}`, clock);
    refuse(`created exact queue has no immutable id for ${REPOSITORIES[key].slug}`, "INVALID_RULESET_ID");
  }
  let audit;
  try {
    audit = auditFeaturePolicies(github, key, state.featureBranch, state.epic);
  } catch (error) {
    markRecovery(statePath, state, `exact queue verification failed for ${REPOSITORIES[key].slug}: ${error.message}`, clock);
    throw error;
  }
  if (
    !audit.exactQueue ||
    audit.exactQueue.id !== created.id ||
    audit.exactQueue.digest !== policy.stagedDigest ||
    audit.exactQueue.enforcement !== "disabled"
  ) {
    markRecovery(statePath, state, `staged exact queue creation was not verifiable for ${REPOSITORIES[key].slug}`, clock);
    refuse(`created exact queue did not match its disabled staging payload for ${REPOSITORIES[key].slug}`, "RULESET_VERIFY_FAILED");
  }
  mutation.id = created.id;
  mutation.status = "created";
  reconcilePolicyRecord(state, key, audit.exactQueue, false);
  saveState(statePath, state, clock);
}

function ensurePoliciesForBootstrap(statePath, state, github, clock, beforeFirstMutation) {
  let audits = livePolicyAudits(state, github);
  assertWildcardPoliciesUnchanged(state, audits);
  const present = Object.keys(REPOSITORIES).filter((key) => audits[key].exactQueue);
  if (present.length === 1) {
    refuse("exactly one repository has the planned exact queue policy; use recover --complete only for a state-owned partial", "PARTIAL_POLICY_PAIR");
  }
  if (present.length === Object.keys(REPOSITORIES).length) {
    for (const key of Object.keys(REPOSITORIES)) {
      const plannedId = state.policies[key].exactQueue.id;
      if (
        plannedId === null ||
        plannedId !== audits[key].exactQueue.id ||
        state.policies[key].exactQueue.status !== "active" ||
        audits[key].exactQueue.enforcement !== "active"
      ) {
        refuse(
          `${REPOSITORIES[key].slug} exact queue appeared, changed, or became disabled after plan`,
          "STALE_EXACT_POLICY",
        );
      }
    }
  }
  beforeFirstMutation();
  if (present.length === 0) {
    state.phase = "staging-policies";
    for (const key of Object.keys(REPOSITORIES)) {
      installExactQueue(statePath, state, key, "create-ruleset", github, clock);
    }
    audits = livePolicyAudits(state, github);
    assertWildcardPoliciesUnchanged(state, audits);
  } else {
    for (const key of Object.keys(REPOSITORIES)) {
      reconcilePolicyRecord(state, key, audits[key].exactQueue);
    }
    saveState(statePath, state, clock);
  }
  return audits;
}

function recoverPolicies(statePath, state, github, clock, beforeFirstMutation) {
  let audits = livePolicyAudits(state, github);
  assertWildcardPoliciesUnchanged(state, audits);
  const present = Object.keys(REPOSITORIES).filter((key) => audits[key].exactQueue);
  if (present.length === 0 && state.transaction.policyMutations.length === 0) {
    refuse("no state-owned exact queue transaction exists to recover", "UNOWNED_POLICY_RECOVERY");
  }
  for (const key of present) {
    if (!ownsPolicy(state, key, audits[key].exactQueue)) {
      refuse(`${REPOSITORIES[key].slug} exact queue is not owned by this transaction`, "UNOWNED_EXACT_POLICY");
    }
    reconcilePolicyRecord(state, key, audits[key].exactQueue);
  }
  beforeFirstMutation();
  saveState(statePath, state, clock);
  for (const key of Object.keys(REPOSITORIES).filter((candidate) => !audits[candidate].exactQueue)) {
    installExactQueue(statePath, state, key, "recover-create-ruleset", github, clock);
  }
  audits = livePolicyAudits(state, github);
  assertWildcardPoliciesUnchanged(state, audits);
  if (!Object.keys(REPOSITORIES).every((key) => audits[key].exactQueue)) {
    refuse("recovery did not produce a complete exact queue pair", "PARTIAL_POLICY_PAIR");
  }
  return audits;
}

function policyWasCreatedByTransaction(state, key, id) {
  return state.transaction.policyMutations.some(
    (mutation) =>
      mutation.repo === key &&
      mutation.id === id &&
      mutation.digest === state.policies[key].exactQueue.stagedDigest &&
      ["create-ruleset", "recover-create-ruleset"].includes(mutation.action) &&
      ["created", "reconciled"].includes(mutation.status),
  );
}

function retireUnresolvedTransitionIntents(state, key, actions, digest, id) {
  for (const previous of state.transaction.policyMutations) {
    if (
      previous.repo === key &&
      actions.includes(previous.action) &&
      previous.digest === digest &&
      previous.id === id &&
      previous.status === "intent"
    ) {
      previous.status = "failed";
    }
  }
}

function disableExactQueueForMissingRef(statePath, state, key, github, clock) {
  const fixed = REPOSITORIES[key];
  const policy = state.policies[key].exactQueue;
  const before = auditFeaturePolicies(github, key, state.featureBranch, state.epic).exactQueue;
  if (
    !before ||
    before.id !== policy.id ||
    before.enforcement !== "active" ||
    before.digest !== policy.digest
  ) {
    refuse(`${fixed.slug} exact queue is not the recorded active policy`, "UNSAFE_RULESET_UPDATE");
  }
  if (!policyWasCreatedByTransaction(state, key, before.id)) {
    refuse(
      `${fixed.slug} active exact queue cannot be disabled because this transaction did not create it`,
      "UNOWNED_EXACT_POLICY",
    );
  }
  retireUnresolvedTransitionIntents(
    state,
    key,
    ["disable-ruleset", "recover-disable-ruleset"],
    policy.stagedDigest,
    before.id,
  );
  const mutation = {
    action: "recover-disable-ruleset",
    repo: key,
    name: policy.name,
    digest: policy.stagedDigest,
    id: before.id,
    at: nowIso(clock),
    status: "intent",
  };
  state.transaction.policyMutations.push(mutation);
  state.phase = "disabling-policies";
  saveState(statePath, state, clock);
  try {
    github.updateRuleset(fixed.slug, before.id, policy.stagedPayload);
  } catch (error) {
    mutation.status = "failed";
    policy.status = "failed";
    markRecovery(statePath, state, `exact queue disable failed for ${fixed.slug}: ${error.message}`, clock);
    throw error;
  }
  let after;
  try {
    after = auditFeaturePolicies(github, key, state.featureBranch, state.epic).exactQueue;
  } catch (error) {
    markRecovery(statePath, state, `exact queue disable verification failed for ${fixed.slug}: ${error.message}`, clock);
    throw error;
  }
  if (
    !after ||
    after.id !== before.id ||
    after.enforcement !== "disabled" ||
    after.digest !== policy.stagedDigest
  ) {
    markRecovery(statePath, state, `exact queue disable was not verifiable for ${fixed.slug}`, clock);
    refuse(`disabled exact queue did not match its staging payload for ${fixed.slug}`, "RULESET_VERIFY_FAILED");
  }
  mutation.status = "created";
  reconcilePolicyRecord(state, key, after, false);
  saveState(statePath, state, clock);
}

function activateExactQueue(statePath, state, key, action, github, clock) {
  const fixed = REPOSITORIES[key];
  const policy = state.policies[key].exactQueue;
  const before = auditFeaturePolicies(github, key, state.featureBranch, state.epic).exactQueue;
  if (
    !before ||
    before.id !== policy.id ||
    before.enforcement !== "disabled" ||
    before.digest !== policy.stagedDigest
  ) {
    refuse(`${fixed.slug} exact queue is not the recorded disabled staging policy`, "UNSAFE_RULESET_UPDATE");
  }
  if (!policyWasCreatedByTransaction(state, key, before.id)) {
    refuse(`${fixed.slug} disabled exact queue is not owned by this transaction`, "UNOWNED_EXACT_POLICY");
  }
  retireUnresolvedTransitionIntents(
    state,
    key,
    ["activate-ruleset", "recover-activate-ruleset"],
    policy.digest,
    before.id,
  );
  const mutation = {
    action,
    repo: key,
    name: policy.name,
    digest: policy.digest,
    id: before.id,
    at: nowIso(clock),
    status: "intent",
  };
  state.transaction.policyMutations.push(mutation);
  state.phase = "activating-policies";
  saveState(statePath, state, clock);
  try {
    github.updateRuleset(fixed.slug, before.id, policy.payload);
  } catch (error) {
    mutation.status = "failed";
    policy.status = "failed";
    markRecovery(statePath, state, `exact queue activation failed for ${fixed.slug}: ${error.message}`, clock);
    throw error;
  }
  let after;
  try {
    after = auditFeaturePolicies(github, key, state.featureBranch, state.epic).exactQueue;
  } catch (error) {
    markRecovery(statePath, state, `exact queue activation verification failed for ${fixed.slug}: ${error.message}`, clock);
    throw error;
  }
  if (
    !after ||
    after.id !== before.id ||
    after.enforcement !== "active" ||
    after.digest !== policy.digest
  ) {
    markRecovery(statePath, state, `exact queue activation was not verifiable for ${fixed.slug}`, clock);
    refuse(`activated exact queue did not match its required payload for ${fixed.slug}`, "RULESET_VERIFY_FAILED");
  }
  mutation.status = "created";
  reconcilePolicyRecord(state, key, after, false);
  saveState(statePath, state, clock);
}

function disableActivePoliciesForMissingRefs(statePath, state, audits, missing, github, clock) {
  for (const key of missing) {
    const live = audits[key].exactQueue;
    if (!live) refuse(`${REPOSITORIES[key].slug} has no exact queue for ref recovery`, "MISSING_RULESET");
    if (live.enforcement === "active") {
      disableExactQueueForMissingRef(statePath, state, key, github, clock);
    } else if (
      live.id !== state.policies[key].exactQueue.id ||
      live.digest !== state.policies[key].exactQueue.stagedDigest
    ) {
      refuse(
        `${REPOSITORIES[key].slug} disabled exact queue is not the recorded staging policy`,
        "UNSAFE_RULESET_UPDATE",
      );
    }
  }
  const staged = livePolicyAudits(state, github);
  assertWildcardPoliciesUnchanged(state, staged);
  for (const key of missing) {
    if (staged[key].exactQueue?.enforcement !== "disabled") {
      refuse(
        `${REPOSITORIES[key].slug} exact queue is still active before create-ref`,
        "ACTIVE_QUEUE_BLOCKS_CREATE_REF",
      );
    }
  }
  return staged;
}

export function verifyEffectiveFeaturePolicies(github, repo, branch, audit) {
  normalizeRepoKey(repo);
  if (audit?.exactQueue?.enforcement !== "active") {
    refuse(`${REPOSITORIES[repo].slug} exact feature queue is not active`, "INACTIVE_EXACT_POLICY");
  }
  const rules = github.branchRules(REPOSITORIES[repo].slug, branch);
  if (!Array.isArray(rules)) {
    refuse(`${REPOSITORIES[repo].slug} effective branch rules are not an array`, "INVALID_EFFECTIVE_RULES");
  }
  const expected = new Map([
    [audit.wildcardBase.id, new Set(["non_fast_forward", "pull_request", "required_status_checks"])],
    [audit.deletionGuard.id, new Set(["deletion"])],
    [audit.exactQueue.id, new Set(["merge_queue"])],
  ]);
  for (const rule of rules) expected.get(rule?.ruleset_id)?.delete(rule?.type);
  const missing = [...expected.entries()]
    .flatMap(([id, types]) => [...types].map((type) => `${id}:${type}`));
  if (missing.length > 0) {
    refuse(
      `${REPOSITORIES[repo].slug} effective feature rules are missing ${missing.join(", ")}`,
      "INEFFECTIVE_FEATURE_POLICY",
    );
  }
  return {
    rulesetIds: [...expected.keys()],
    ruleTypes: [...new Set(rules.map(({ type }) => type))].sort(),
  };
}

function activateAndVerifyRepository(statePath, state, key, github, clock, recovering = false) {
  let audits = livePolicyAudits(state, github);
  assertWildcardPoliciesUnchanged(state, audits);
  let live = audits[key].exactQueue;
  if (!live) refuse(`${REPOSITORIES[key].slug} has no exact queue to activate`, "MISSING_RULESET");
  if (live.enforcement === "active") {
    const recorded = state.policies[key].exactQueue;
    if (recorded.id !== live.id || recorded.digest !== live.digest) {
      refuse(`${REPOSITORIES[key].slug} active exact queue is not the recorded ruleset`, "UNOWNED_EXACT_POLICY");
    }
    reconcilePolicyRecord(state, key, live);
    saveState(statePath, state, clock);
  } else {
    activateExactQueue(
      statePath,
      state,
      key,
      recovering ? "recover-activate-ruleset" : "activate-ruleset",
      github,
      clock,
    );
    audits = livePolicyAudits(state, github);
    assertWildcardPoliciesUnchanged(state, audits);
    live = audits[key].exactQueue;
  }
  try {
    verifyEffectiveFeaturePolicies(github, key, state.featureBranch, audits[key]);
  } catch (error) {
    markRecovery(
      statePath,
      state,
      `effective feature policy verification failed for ${REPOSITORIES[key].slug}: ${error.message}`,
      clock,
    );
    throw error;
  }
  return live;
}

function assertMainsUnchanged(state, github) {
  for (const [key, fixed] of Object.entries(REPOSITORIES)) {
    if (github.defaultBranch(fixed.slug) !== "main") {
      refuse(`${fixed.slug} live default branch is no longer main`, "NON_MAIN_DEFAULT");
    }
    const live = github.mainSha(fixed.slug);
    if (live !== state.repositories[key].plannedMainSha) {
      refuse(
        `${fixed.slug} main moved from planned ${state.repositories[key].plannedMainSha} to ${live}; re-plan`,
        "STALE_MAIN_HEAD",
      );
    }
  }
}

function transactionHasMutated(state) {
  return (
    state.transaction.policyMutations.length > 0 ||
    state.transaction.remoteMutations.length > 0 ||
    state.transaction.completedAt !== null ||
    Object.values(state.repositories).some(({ workspace }) => workspace.status !== "pending")
  );
}

function assertPreMutationPlanValid(state, github) {
  if (transactionHasMutated(state)) return;
  assertMainsUnchanged(state, github);
  assertRequiredMainChecks(github, state.repositories, state.requiredContexts);
}

function assertAdoptedPlanValid(state, github) {
  if (state.mode !== "adopt-existing") refuse("state is not an adopted train", "INVALID_PLAN_MODE");
  assertMainsUnchanged(state, github);
  for (const [key, fixed] of Object.entries(REPOSITORIES)) {
    const repo = state.repositories[key];
    const liveFeature = github.featureSha(fixed.slug, state.featureBranch);
    if (liveFeature !== repo.adoptedFeatureSha) {
      refuse(
        `${fixed.slug} adopted feature moved from ${repo.adoptedFeatureSha} to ${liveFeature}; re-plan before bootstrap`,
        "STALE_ADOPTED_FEATURE_HEAD",
      );
    }
    const ancestry = github.compare(fixed.slug, repo.plannedMainSha, repo.adoptedFeatureSha);
    if (!new Set(["ahead", "identical"]).has(ancestry?.status)) {
      refuse(`${fixed.slug} adopted feature no longer proves planned-main ancestry`, "ADOPTED_FEATURE_MISSING_MAIN");
    }
    const policies = auditFeaturePolicies(github, key, state.featureBranch, state.epic);
    if (
      policies.wildcardBase.id !== state.policies[key].wildcardBase.id ||
      policies.wildcardBase.digest !== state.policies[key].wildcardBase.digest ||
      policies.deletionGuard.id !== state.policies[key].deletionGuard.id ||
      policies.deletionGuard.digest !== state.policies[key].deletionGuard.digest ||
      policies.exactQueue?.id !== state.policies[key].exactQueue.id ||
      policies.exactQueue?.digest !== state.policies[key].exactQueue.digest ||
      policies.exactQueue?.enforcement !== "active"
    ) {
      refuse(`${fixed.slug} adopted protection differs from its immutable plan`, "ADOPTION_POLICY_MISMATCH");
    }
    verifyEffectiveFeaturePolicies(github, key, state.featureBranch, policies);
    const checkRuns = github.checkRuns(fixed.slug, repo.adoptedFeatureSha);
    if (
      missingRequiredChecks(checkRuns, state.requiredContexts[key]).length > 0 ||
      !checkReceiptsStillPresent(checkRuns, repo.adoptedRequiredChecks)
    ) {
      refuse(`${fixed.slug} adopted-head required checks no longer match the plan`, "ADOPTED_HEAD_CHECKS_NOT_GREEN");
    }
  }
  const pin = extractInferencePin(
    github.cargoManifests(
      REPOSITORIES.sceneworks.slug,
      state.repositories.sceneworks.adoptedFeatureSha,
    ),
  );
  if (
    !sameJson(pin, state.inferencePin) ||
    !isReachableFromMain(github, pin.sha, state.repositories.inference.plannedMainSha) ||
    !isReachableFromMain(github, pin.sha, state.repositories.inference.adoptedFeatureSha)
  ) {
    refuse("adopted SceneWorks/inference pin pairing changed or is unreachable", "ADOPTED_PIN_MISMATCH");
  }
}

function assertRecordedRefConsistency(state, refs) {
  for (const key of Object.keys(REPOSITORIES)) {
    const planned = state.repositories[key].plannedMainSha;
    const live = refs[key];
    const recorded = state.repositories[key].remoteRef;
    if (live && live !== planned) {
      refuse(
        `${REPOSITORIES[key].slug} ${state.featureBranch} is ${live}, not planned ${planned}`,
        "DIVERGENT_FEATURE_REF",
      );
    }
    if (recorded && recorded !== live) {
      refuse(
        `${REPOSITORIES[key].slug} recorded feature ref ${recorded} conflicts with live ${live}`,
        "STATE_REF_CONFLICT",
      );
    }
  }
}

function immutableFeatureBaseline(repoState) {
  return repoState.adoptedFeatureSha ?? repoState.plannedMainSha;
}

function markRecovery(statePath, state, message, clock) {
  state.phase = "recovery-required";
  state.transaction.recoveryReason = message;
  saveState(statePath, state, clock);
}

function provisionWorkspace(statePath, state, key, git, clock) {
  const repoState = state.repositories[key];
  const expectedHead = immutableFeatureBaseline(repoState);
  const fixed = REPOSITORIES[key];
  const target = repoState.workspace.path;
  const inspected = git.inspectWorkspace(target);
  if (inspected) {
    if (!new Set(["ready", "provisioning"]).has(repoState.workspace.status)) {
      refuse(`unrecorded or reused workspace path exists: ${target}`, "REUSED_WORKSPACE");
    }
    if (
      inspected.invalid ||
      inspected.dirty ||
      inspected.origin !== fixed.cloneUrl ||
      inspected.branch !== state.featureBranch ||
      inspected.head !== expectedHead
    ) {
      refuse(`state-owned workspace is dirty or conflicts with its recorded branch: ${target}`, "DIRTY_WORKSPACE");
    }
    if (repoState.workspace.status === "provisioning") {
      repoState.workspace.status = "ready";
      repoState.workspace.head = inspected.head;
      repoState.workspace.activeStagingPath = null;
      saveState(statePath, state, clock);
    }
    return;
  }
  if (repoState.workspace.status === "ready") {
    refuse(`recorded workspace disappeared: ${target}`, "MISSING_RECORDED_WORKSPACE");
  }
  if (repoState.workspace.status === "provisioning" && repoState.workspace.activeStagingPath) {
    if (existsSync(repoState.workspace.activeStagingPath)) {
      repoState.workspace.failedStagingPaths.push(repoState.workspace.activeStagingPath);
    }
    repoState.workspace.activeStagingPath = null;
    repoState.workspace.status = "failed";
    saveState(statePath, state, clock);
  }
  let attempt = repoState.workspace.failedStagingPaths.length + 1;
  let staging = `${target}.bootstrap-${attempt}`;
  while (existsSync(staging)) {
    attempt += 1;
    staging = `${target}.bootstrap-${attempt}`;
  }
  repoState.workspace.status = "provisioning";
  repoState.workspace.activeStagingPath = staging;
  saveState(statePath, state, clock);
  try {
    git.cloneFeature(fixed, state.featureBranch, target, staging, expectedHead);
  } catch (error) {
    if (existsSync(staging)) repoState.workspace.failedStagingPaths.push(staging);
    repoState.workspace.status = "failed";
    repoState.workspace.activeStagingPath = null;
    markRecovery(statePath, state, `local provisioning failed for ${fixed.slug}: ${error.message}`, clock);
    throw error;
  }
  const ready = git.inspectWorkspace(target);
  if (
    !ready ||
    ready.invalid ||
    ready.dirty ||
    ready.origin !== fixed.cloneUrl ||
    ready.branch !== state.featureBranch ||
    ready.head !== expectedHead
  ) {
    repoState.workspace.status = "failed";
    markRecovery(statePath, state, `provisioned workspace could not be verified for ${fixed.slug}`, clock);
    refuse(`provisioned workspace failed verification: ${target}`, "WORKSPACE_VERIFY_FAILED");
  }
  repoState.workspace.status = "ready";
  repoState.workspace.head = ready.head;
  repoState.workspace.activeStagingPath = null;
  saveState(statePath, state, clock);
}

function provisionAll(statePath, state, git, clock) {
  for (const key of Object.keys(REPOSITORIES)) provisionWorkspace(statePath, state, key, git, clock);
  state.phase = "ready";
  state.transaction.recoveryReason = null;
  state.transaction.completedAt ??= nowIso(clock);
  saveState(statePath, state, clock);
  return state;
}

function createFeatureRefWithJournal(statePath, state, key, action, github, clock) {
  const fixed = REPOSITORIES[key];
  const policy = auditFeaturePolicies(github, key, state.featureBranch, state.epic).exactQueue;
  if (
    !policy ||
    policy.id !== state.policies[key].exactQueue.id ||
    policy.enforcement !== "disabled" ||
    policy.digest !== state.policies[key].exactQueue.stagedDigest
  ) {
    refuse(
      `${fixed.slug} create-ref requires its recorded exact queue to be disabled`,
      "ACTIVE_QUEUE_BLOCKS_CREATE_REF",
    );
  }
  const mutation = {
    action,
    repo: key,
    ref: `refs/heads/${state.featureBranch}`,
    sha: state.repositories[key].plannedMainSha,
    at: nowIso(clock),
    status: "intent",
  };
  state.transaction.remoteMutations.push(mutation);
  saveState(statePath, state, clock);
  try {
    github.createFeatureRef(fixed.slug, state.featureBranch, state.repositories[key].plannedMainSha);
  } catch (error) {
    mutation.status = "failed";
    markRecovery(statePath, state, `remote create-ref failed for ${fixed.slug}: ${error.message}`, clock);
    throw error;
  }
  const created = github.featureSha(fixed.slug, state.featureBranch);
  if (created !== state.repositories[key].plannedMainSha) {
    markRecovery(statePath, state, `remote create-ref was not verifiable for ${fixed.slug}`, clock);
    refuse(`created ref did not resolve to the planned SHA for ${fixed.slug}`, "CREATE_REF_VERIFY_FAILED");
  }
  state.repositories[key].remoteRef = created;
  mutation.status = "created";
  saveState(statePath, state, clock);
}

function bootstrapUnlocked(
  statePath,
  { apply = false } = {},
  { github = new GitHubClient(), git = new GitRunner(), clock = () => new Date() } = {},
) {
  if (!apply) refuse("bootstrap is read-only unless --apply is explicit", "APPLY_REQUIRED");
  statePath = assertAbsolutePath(statePath, "state path");
  const state = loadState(statePath);
  if (state.mode === "adopt-existing") {
    if (!new Set(["planned", "ready"]).has(state.phase)) {
      refuse("adopted state with local recovery work requires recover --complete", "RECOVERY_REQUIRED");
    }
    assertAdoptedPlanValid(state, github);
    return provisionAll(statePath, state, git, clock);
  }
  if (state.phase === "recovery-required") {
    refuse("state requires `recover --complete`; bootstrap will not guess across a partial transaction", "RECOVERY_REQUIRED");
  }
  if (!["planned", "ready"].includes(state.phase)) {
    refuse(
      `state stopped during ${state.phase}; continue with recover --complete`,
      "RECOVERY_REQUIRED",
    );
  }
  const policies = ensurePoliciesForBootstrap(
    statePath,
    state,
    github,
    clock,
    () => assertPreMutationPlanValid(state, github),
  );
  const refs = liveRefs(state, github);
  assertRecordedRefConsistency(state, refs);
  const present = Object.values(refs).filter(Boolean).length;
  if (present === 1) {
    refuse("exactly one mirrored feature ref exists; use recover --complete", "PARTIAL_REMOTE_PAIR");
  }
  if (present === 0) {
    for (const key of Object.keys(REPOSITORIES)) {
      if (policies[key].exactQueue?.enforcement !== "disabled") {
        refuse(
          `${REPOSITORIES[key].slug} active exact queue blocks creation of its missing feature ref`,
          "ACTIVE_QUEUE_WITHOUT_REF",
        );
      }
    }
    state.phase = "applying-remote-refs";
    saveState(statePath, state, clock);
    for (const key of Object.keys(REPOSITORIES)) {
      createFeatureRefWithJournal(statePath, state, key, "create-ref", github, clock);
      activateAndVerifyRepository(statePath, state, key, github, clock);
    }
  } else {
    for (const key of Object.keys(REPOSITORIES)) {
      state.repositories[key].remoteRef = refs[key];
      saveState(statePath, state, clock);
      activateAndVerifyRepository(statePath, state, key, github, clock);
    }
  }
  return provisionAll(statePath, state, git, clock);
}

export function bootstrap(statePath, options = {}, dependencies = {}) {
  if (!options.apply) refuse("bootstrap is read-only unless --apply is explicit", "APPLY_REQUIRED");
  statePath = assertAbsolutePath(statePath, "state path");
  const clock = dependencies.clock ?? (() => new Date());
  return withStateLock(
    statePath,
    { operation: "bootstrap", clock, staleAfterMs: dependencies.lockStaleAfterMs },
    () => bootstrapUnlocked(statePath, options, dependencies),
  );
}

function recoverUnlocked(
  statePath,
  { complete = false } = {},
  { github = new GitHubClient(), git = new GitRunner(), clock = () => new Date() } = {},
) {
  if (!complete) refuse("recover is report-only unless --complete is explicit", "COMPLETE_REQUIRED");
  statePath = assertAbsolutePath(statePath, "state path");
  const state = loadState(statePath);
  if (state.mode === "adopt-existing") {
    if (state.phase === "planned") {
      refuse("adopted planned state has no partial operation; start with bootstrap --apply", "NOTHING_TO_RECOVER");
    }
    assertAdoptedPlanValid(state, github);
    return provisionAll(statePath, state, git, clock);
  }
  if (state.phase === "planned") {
    refuse("planned state has no partial transaction; start with bootstrap --apply", "NOTHING_TO_RECOVER");
  }
  const policies = recoverPolicies(
    statePath,
    state,
    github,
    clock,
    () => assertPreMutationPlanValid(state, github),
  );
  const refs = liveRefs(state, github);
  assertRecordedRefConsistency(state, refs);
  const missing = Object.keys(REPOSITORIES).filter((key) => !refs[key]);
  if (missing.length === 1) {
    const key = missing[0];
    const other = key === "sceneworks" ? "inference" : "sceneworks";
    const recordedIntent = state.transaction.remoteMutations.some(
      (mutation) =>
        mutation.repo === other &&
        mutation.sha === refs[other] &&
        ["intent", "created", "failed", "reconciled"].includes(mutation.status),
    );
    if (
      refs[other] !== state.repositories[other].plannedMainSha ||
      (!state.repositories[other].remoteRef && !recordedIntent)
    ) {
      refuse("the existing half is not a state-recorded matching ref", "UNOWNED_PARTIAL_PAIR");
    }
    state.repositories[other].remoteRef = refs[other];
    for (const mutation of state.transaction.remoteMutations) {
      if (mutation.repo === other && mutation.sha === refs[other] && mutation.status !== "created") {
        mutation.status = "reconciled";
      }
    }
    saveState(statePath, state, clock);
  }
  disableActivePoliciesForMissingRefs(
    statePath,
    state,
    policies,
    missing,
    github,
    clock,
  );
  for (const key of Object.keys(REPOSITORIES)) {
    if (missing.includes(key)) {
      createFeatureRefWithJournal(statePath, state, key, "recover-create-ref", github, clock);
    } else {
      state.repositories[key].remoteRef = refs[key];
    }
    for (const mutation of state.transaction.remoteMutations) {
      if (mutation.status === "intent" || mutation.status === "failed") {
        if (refs[mutation.repo] === mutation.sha) mutation.status = "reconciled";
      }
    }
    saveState(statePath, state, clock);
    activateAndVerifyRepository(statePath, state, key, github, clock, true);
  }
  return provisionAll(statePath, state, git, clock);
}

export function recover(statePath, options = {}, dependencies = {}) {
  if (!options.complete) refuse("recover is report-only unless --complete is explicit", "COMPLETE_REQUIRED");
  statePath = assertAbsolutePath(statePath, "state path");
  const clock = dependencies.clock ?? (() => new Date());
  return withStateLock(
    statePath,
    { operation: "recover", clock, staleAfterMs: dependencies.lockStaleAfterMs },
    () => recoverUnlocked(statePath, options, dependencies),
  );
}

export function classifyPullRequest(head, base) {
  const story = STORY_BRANCH_RE.exec(head);
  if (story) {
    const expectedBase = `feature/sc-${story[2]}-${story[3]}`;
    assertSafeFeatureBranch(expectedBase);
    if (base !== expectedBase) {
      refuse(`epic story ${head} must target ${expectedBase}, not ${base}`, "WRONG_STORY_BASE");
    }
    return { kind: "story", epic: `sc-${story[2]}`, slug: story[3], story: `sc-${story[1]}` };
  }
  const sync = SYNC_BRANCH_RE.exec(head);
  if (sync) {
    const date = new Date(`${sync[2]}T00:00:00Z`);
    if (Number.isNaN(date.valueOf()) || date.toISOString().slice(0, 10) !== sync[2]) {
      refuse(`sync branch ${head} has an invalid calendar date`, "INVALID_SYNC_DATE");
    }
    const feature = FEATURE_RE.exec(base);
    if (!feature || feature[1] !== sync[1]) {
      refuse(`sync branch ${head} must target its matching epic feature branch`, "WRONG_SYNC_BASE");
    }
    assertSafeFeatureBranch(base);
    return { kind: "sync", epic: `sc-${sync[1]}`, slug: feature[2] };
  }
  const feature = FEATURE_RE.exec(head);
  if (feature) {
    assertSafeFeatureBranch(head);
    if (base !== "main") {
      refuse(`feature branch ${head} may target main only, not ${base}`, "WRONG_FEATURE_BASE");
    }
    return { kind: "final", epic: `sc-${feature[1]}`, slug: feature[2] };
  }
  if (
    head.startsWith("sync/") ||
    head.startsWith("feature/") ||
    EPIC_STORY_PREFIX_RE.test(head)
  ) {
    refuse(`malformed feature-train head ${head}`, "MALFORMED_TRAIN_REF");
  }
  if (base.startsWith("feature/")) {
    assertSafeFeatureBranch(base);
    refuse(`only an epic-bearing story or matching sync branch may target ${base}`, "UNOWNED_FEATURE_PR");
  }
  return { kind: "ordinary" };
}

function refName(value) {
  return typeof value === "string" && value.startsWith("refs/heads/") ? value.slice(11) : null;
}

function canonicalFeatureFor(epic, actual, featureResolver) {
  if (typeof featureResolver !== "function") {
    refuse(`a live canonical feature-branch resolver is required for ${epic}`, "MISSING_CANONICAL_RESOLVER");
  }
  const canonical = featureResolver(epic);
  const parsed = assertSafeFeatureBranch(canonical);
  if (parsed.epic !== epic) {
    refuse(`canonical resolver returned ${canonical} for ${epic}`, "INVALID_CANONICAL_REF");
  }
  if (actual !== canonical) {
    refuse(
      `${actual} is not the unique live canonical feature branch for ${epic}: ${canonical}`,
      "NONCANONICAL_FEATURE_REF",
    );
  }
  return canonical;
}

export function evaluateEventTopology(
  eventName,
  event,
  repositorySlug = REPOSITORIES.sceneworks.slug,
  featureResolver,
) {
  if (eventName === "pull_request") {
    const pr = event?.pull_request;
    const head = pr?.head?.ref;
    const base = pr?.base?.ref;
    if (!head || !base || !pr?.head?.repo?.full_name || !event?.repository?.full_name) {
      refuse("pull_request event is missing head/base repository identity", "MALFORMED_EVENT");
    }
    const classification = classifyPullRequest(head, base);
    if (classification.kind !== "ordinary") {
      if (event.repository.full_name !== repositorySlug || pr.head.repo.full_name !== repositorySlug) {
        refuse("feature-train pull requests must use a same-repository head", "CROSS_REPOSITORY_HEAD");
      }
      canonicalFeatureFor(
        classification.epic,
        classification.kind === "final" ? head : base,
        featureResolver,
      );
    }
    return { ...classification, validateFinalPin: classification.kind === "final", head, base };
  }
  if (eventName === "merge_group") {
    const base = refName(event?.merge_group?.base_ref);
    const head = refName(event?.merge_group?.head_ref);
    if (!base || !head || !SHA_RE.test(event?.merge_group?.head_sha ?? "")) {
      refuse("merge_group event must carry full branch refs and an exact head SHA", "MALFORMED_EVENT");
    }
    if (!head.startsWith(`gh-readonly-queue/${base}/`)) {
      refuse(`merge_group head ${head} does not correspond to queue target ${base}`, "QUEUE_TARGET_CONFLICT");
    }
    const featureTarget = base.startsWith("feature/");
    if (featureTarget) {
      const parsed = assertSafeFeatureBranch(base);
      canonicalFeatureFor(parsed.epic, base, featureResolver);
    }
    return {
      kind: base === "main" ? "main-merge-group" : featureTarget ? "feature-merge-group" : "ordinary",
      validateFinalPin: base === "main",
      head,
      base,
    };
  }
  if (eventName === "push") {
    const branch = refName(event?.ref);
    if (!branch || !SHA_RE.test(event?.after ?? "")) {
      refuse("push event must carry a full branch ref and exact after SHA", "MALFORMED_EVENT");
    }
    return { kind: branch === "main" ? "main-push" : "ordinary", validateFinalPin: branch === "main", head: branch, base: null };
  }
  if (eventName === "workflow_dispatch") {
    return { kind: "ordinary", validateFinalPin: false, head: null, base: null };
  }
  refuse(`unsupported event ${JSON.stringify(eventName)}`, "UNSUPPORTED_EVENT");
}

export function localInferencePin(repoRoot, git = new GitRunner()) {
  const paths = git.trackedCargoManifests(repoRoot);
  if (paths.length === 0) refuse("git reports no tracked Cargo manifests", "MANIFEST_ENUMERATION_FAILED");
  return extractInferencePin(paths.map((path) => ({ path, text: readFileSync(join(repoRoot, path), "utf8") })));
}

export function enforceCiPolicy(
  { eventName, event, repoRoot },
  { github = new GitHubClient(), git = new GitRunner(), featureResolver } = {},
) {
  repoRoot = assertAbsolutePath(repoRoot, "repository root");
  const liveFeatureResolver = featureResolver ?? ((epic) => resolveRemoteFeatureBranch(epic, repoRoot, git));
  const topology = evaluateEventTopology(
    eventName,
    event,
    REPOSITORIES.sceneworks.slug,
    liveFeatureResolver,
  );
  let pin = null;
  if (topology.validateFinalPin) {
    pin = localInferencePin(repoRoot, git);
    if (github.defaultBranch(REPOSITORIES.inference.slug) !== "main") {
      refuse("inference live default branch is not main", "NON_MAIN_DEFAULT");
    }
    const mainSha = github.mainSha(REPOSITORIES.inference.slug);
    if (!SHA_RE.test(mainSha ?? "") || !isReachableFromMain(github, pin.sha, mainSha)) {
      refuse(`inference pin ${pin.sha} is not an ancestor of live inference main`, "UNREACHABLE_INFERENCE_PIN");
    }
  }
  return { ok: true, topology, pin };
}

function assertStatePullRequest(state, classification) {
  if (classification.kind === "ordinary") {
    refuse("record-pr records feature-train PRs only", "ORDINARY_PR");
  }
  if (classification.epic !== state.epic || classification.slug !== state.slug) {
    refuse("pull request does not belong to the state epic", "PR_EPIC_CONFLICT");
  }
  if (classification.kind === "story" && !state.expectedStories.includes(classification.story)) {
    refuse(`story ${classification.story} is not in the expected story set`, "UNEXPECTED_STORY");
  }
}

function normalizeRepoKey(value) {
  if (!Object.hasOwn(REPOSITORIES, value)) {
    refuse(`repo must be one of ${Object.keys(REPOSITORIES).join(", ")}`, "INVALID_REPOSITORY");
  }
  return value;
}

function evidenceTimestamp(value) {
  if (typeof value !== "string") return Number.NaN;
  const match = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.\d+)?(?:Z|[+-](\d{2}):(\d{2}))$/.exec(value);
  if (!match) return Number.NaN;
  const [, yearText, monthText, dayText, hourText, minuteText, secondText, offsetHourText, offsetMinuteText] = match;
  const year = Number(yearText);
  const month = Number(monthText);
  const day = Number(dayText);
  const hour = Number(hourText);
  const minute = Number(minuteText);
  const second = Number(secondText);
  const offsetHour = offsetHourText === undefined ? 0 : Number(offsetHourText);
  const offsetMinute = offsetMinuteText === undefined ? 0 : Number(offsetMinuteText);
  const leapYear = year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0);
  const daysInMonth = [31, leapYear ? 29 : 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
  if (
    month < 1 ||
    month > 12 ||
    day < 1 ||
    day > daysInMonth[month - 1] ||
    hour > 23 ||
    minute > 59 ||
    second > 59 ||
    offsetHour > 23 ||
    offsetMinute > 59
  ) {
    return Number.NaN;
  }
  return Date.parse(value);
}

function queueEvidenceSinceReceipt(timeline, recordedAt) {
  if (!Array.isArray(timeline)) refuse("GitHub PR timeline is not an array", "INVALID_TIMELINE");
  const recordedAtMs = evidenceTimestamp(recordedAt);
  if (!Number.isFinite(recordedAtMs)) return { evidenceAt: null, unverifiable: true };
  let evidenceAt = null;
  let evidenceAtMs = Number.POSITIVE_INFINITY;
  let unverifiable = false;
  for (const event of timeline) {
    if (!MERGE_QUEUE_EVENTS.has(event?.event)) continue;
    const createdAtMs = evidenceTimestamp(event?.created_at);
    if (!Number.isFinite(createdAtMs)) {
      unverifiable = true;
      continue;
    }
    if (createdAtMs >= recordedAtMs && createdAtMs < evidenceAtMs) {
      evidenceAt = event.created_at;
      evidenceAtMs = createdAtMs;
    }
  }
  return { evidenceAt, unverifiable };
}

function historicalQueueReceipt(timeline, headSha, mergeCommit, mergedAt) {
  if (!Array.isArray(timeline)) refuse("GitHub PR timeline is not an array", "INVALID_TIMELINE");
  const mergedAtMs = evidenceTimestamp(mergedAt);
  if (!Number.isFinite(mergedAtMs)) {
    refuse("merged PR lacks a verifiable merge time", "INVALID_ADOPTION_MERGE");
  }
  const candidates = [];
  for (const event of timeline) {
    if (!MERGE_QUEUE_EVENTS.has(event?.event)) continue;
    if (!validEvidenceTime(event.created_at)) {
      refuse("historical merge-queue evidence has no exact event time", "INVALID_ADOPTION_QUEUE");
    }
    if (evidenceTimestamp(event.created_at) <= mergedAtMs) {
      candidates.push({ event: event.event, at: event.created_at });
    }
  }
  if (candidates.length === 0) {
    refuse("adopted PR has no timestamped merge-queue event before merge", "MISSING_ADOPTION_QUEUE");
  }
  candidates.sort((left, right) => evidenceTimestamp(right.at) - evidenceTimestamp(left.at));
  const selected = candidates[0];
  if (
    candidates.slice(1).some(
      (candidate) => candidate.at === selected.at && candidate.event !== selected.event,
    )
  ) {
    refuse("historical merge-queue evidence is ambiguous at the selected time", "AMBIGUOUS_ADOPTION_QUEUE");
  }
  return { ...selected, headSha, mergeCommit };
}

function exactPullRequestIdentity(pr, fixed, number) {
  return Boolean(
    pr &&
    pr.number === number &&
    pr.html_url === `https://github.com/${fixed.slug}/pull/${number}` &&
    pr.base?.repo?.full_name === fixed.slug &&
    pr.head?.repo?.full_name === fixed.slug &&
    typeof pr.head?.ref === "string" &&
    typeof pr.base?.ref === "string" &&
    SHA_RE.test(pr.head?.sha ?? "")
  );
}

function pullRequestMergeEvidence(pr) {
  if (pr?.merged_at == null) {
    return { mergeCommit: null, mergedAt: null };
  }
  if (
    pr.state !== "closed" ||
    !validEvidenceTime(pr.merged_at) ||
    !SHA_RE.test(pr.merge_commit_sha ?? "")
  ) {
    refuse("pull request has malformed merge evidence", "INVALID_PR_MERGE");
  }
  return { mergeCommit: pr.merge_commit_sha, mergedAt: pr.merged_at };
}

function sameReceiptExceptTimestamp(left, right, timestampKey) {
  if (!left || !right) return false;
  const leftComparable = { ...left, [timestampKey]: null };
  const rightComparable = { ...right, [timestampKey]: null };
  return sameJson(leftComparable, rightComparable);
}

function recordEvidenceUnlocked(
  statePath,
  { repo, number, runReceipts = [], deploymentIds = [] },
  { github = new GitHubClient(), clock = () => new Date() } = {},
) {
  statePath = assertAbsolutePath(statePath, "state path");
  const state = loadState(statePath);
  repo = normalizeRepoKey(repo);
  if (!Number.isSafeInteger(number) || number <= 0) refuse("PR number must be a positive integer", "INVALID_PR_NUMBER");
  const fixed = REPOSITORIES[repo];
  const pr = github.pullRequest(fixed.slug, number);
  if (!exactPullRequestIdentity(pr, fixed, number)) {
    refuse("recorded feature-train PR must have exact same-repository identity", "INVALID_PULL_REQUEST");
  }
  if (!new Set(["open", "closed"]).has(pr.state)) {
    refuse("recorded feature-train PR has invalid live state", "INVALID_PULL_REQUEST");
  }
  const classification = classifyPullRequest(pr.head.ref, pr.base.ref);
  assertStatePullRequest(state, classification);
  if (state.adoptedPullRequests.some((item) => item.repo === repo && item.number === number)) {
    refuse(`PR ${fixed.slug}#${number} is already an immutable adoption receipt`, "PR_RECEIPT_CONFLICT");
  }
  const mergeEvidence = pullRequestMergeEvidence(pr);
  const existingReceipt = state.pullRequests.find(
    (item) => item.repo === repo && item.number === number,
  );
  const receipt = {
    repo,
    number,
    url: expectedPullRequestUrl(repo, number),
    head: pr.head.ref,
    base: pr.base.ref,
    headSha: pr.head.sha,
    kind: classification.kind,
    story: classification.story ?? null,
    recordedAt: existingReceipt?.recordedAt ?? nowIso(clock),
  };
  if (existingReceipt && !sameJson(existingReceipt, receipt)) {
    refuse(`PR ${fixed.slug}#${number} conflicts with its recorded head/base`, "PR_RECEIPT_CONFLICT");
  }
  if (!existingReceipt) state.pullRequests.push(receipt);
  if (classification.kind === "final") {
    const pointer = {
      number,
      headSha: pr.head.sha,
      url: expectedPullRequestUrl(repo, number),
      mergeCommit: mergeEvidence.mergeCommit,
      mergedAt: mergeEvidence.mergedAt,
    };
    const existingPointer = state.finalPullRequests[repo];
    if (
      existingPointer &&
      (existingPointer.number !== number ||
        existingPointer.headSha !== pointer.headSha ||
        existingPointer.url !== pointer.url)
    ) {
      refuse(`${fixed.slug} already has a different immutable final PR`, "FINAL_PR_CONFLICT");
    }
    if (
      existingPointer?.mergeCommit &&
      (pointer.mergeCommit !== existingPointer.mergeCommit ||
        pointer.mergedAt !== existingPointer.mergedAt)
    ) {
      refuse(`${fixed.slug} final PR merge evidence changed`, "FINAL_PR_CONFLICT");
    }
    state.finalPullRequests[repo] = existingPointer?.mergeCommit ? existingPointer : pointer;
    state.shortcut.inventoryFrozenAt ??= receipt.recordedAt;
  }

  if (runReceipts.length > 0) {
    if (classification.kind !== "final") {
      refuse("privileged run receipts must be attached to a final feature PR", "RUN_REQUIRES_FINAL_PR");
    }
    const alreadyQueued = github.timeline(fixed.slug, number)
      .some((event) => MERGE_QUEUE_EVENTS.has(event?.event));
    if (pr.state !== "open" || pr.merged_at || alreadyQueued) {
      refuse(
        "privileged runtime receipts must be recorded from the final source head before merge-queue entry",
        "RUN_REQUIRES_OPEN_FINAL_PR",
      );
    }
  }
  for (const item of runReceipts) {
    const runRepo = normalizeRepoKey(item.repo ?? repo);
    if (!Number.isSafeInteger(item.id) || item.id <= 0) refuse("workflow run id must be a positive integer", "INVALID_RUN_ID");
    const run = github.workflowRun(REPOSITORIES[runRepo].slug, item.id);
    if (!run || !SHA_RE.test(run.head_sha ?? "") || !run.html_url || !run.path) {
      refuse(`workflow run ${item.id} lacks exact-head evidence`, "INVALID_RUN_RECEIPT");
    }
    const expected = state.privilegedWorkflows.find(
      ({ repo: expectedRepo, workflow }) => expectedRepo === runRepo && run.path.endsWith(`/${workflow}`),
    );
    if (!expected) refuse(`workflow run ${item.id} is not a required privileged workflow`, "UNEXPECTED_RUN");
    if (run.event !== "workflow_dispatch" || run.status !== "completed" || run.conclusion !== "success") {
      refuse(`workflow run ${item.id} is not a successful explicit dispatch`, "UNSUCCESSFUL_RUN");
    }
    if (run.head_sha !== pr.head.sha) {
      refuse(`workflow run ${item.id} did not execute the final PR head ${pr.head.sha}`, "INEXACT_RUN_HEAD");
    }
    const runReceipt = {
      repo: runRepo,
      workflow: expected.workflow,
      id: item.id,
      headSha: run.head_sha,
      url: run.html_url,
      evidencePhase: "pre-queue-source",
      recordedAt: nowIso(clock),
      finalPrRepo: repo,
      finalPrNumber: number,
    };
    const conflicts = state.privilegedRuns.filter(
      (existing) =>
        (existing.repo === runRepo && existing.workflow === expected.workflow) ||
        (existing.repo === runRepo && existing.id === item.id),
    );
    if (conflicts.length === 0) state.privilegedRuns.push(runReceipt);
    else if (!conflicts.every((existing) => sameReceiptExceptTimestamp(existing, runReceipt, "recordedAt"))) {
      refuse(`workflow run ${item.id} conflicts with immutable runtime evidence`, "RUN_RECEIPT_CONFLICT");
    }
  }

  if (deploymentIds.length > 0) {
    if (classification.kind !== "final") {
      refuse("deployment receipts must be attached to a final feature PR", "DEPLOYMENT_REQUIRES_FINAL_PR");
    }
    if (!pr.merged_at || !SHA_RE.test(pr.merge_commit_sha ?? "")) {
      refuse(
        "deployment receipts require the landed merge commit of a merged final PR",
        "DEPLOYMENT_REQUIRES_MERGED_FINAL_PR",
      );
    }
  }
  for (const item of deploymentIds) {
    const deploymentRepo = normalizeRepoKey(item.repo ?? repo);
    if (!Number.isSafeInteger(item.id) || item.id <= 0) refuse("deployment id must be a positive integer", "INVALID_DEPLOYMENT_ID");
    const deployment = github.deployment(REPOSITORIES[deploymentRepo].slug, item.id);
    if (!deployment || !SHA_RE.test(deployment.sha ?? "")) refuse("deployment lacks an exact SHA", "INVALID_DEPLOYMENT");
    const expectedDeploymentSha = pr.merge_commit_sha;
    if (deployment.sha !== expectedDeploymentSha) {
      refuse(
        `deployment ${item.id} is ${deployment.sha}, not final revision ${expectedDeploymentSha}`,
        "INEXACT_DEPLOYMENT_HEAD",
      );
    }
    const deploymentReceipt = {
      repo: deploymentRepo,
      id: item.id,
      sha: deployment.sha,
      url: deployment.url ?? deployment.statuses_url,
      evidencePhase: "post-merge",
      recordedAt: nowIso(clock),
      finalPrRepo: repo,
      finalPrNumber: number,
    };
    if (typeof deploymentReceipt.url !== "string" || deploymentReceipt.url.length === 0) {
      refuse(`deployment ${item.id} has no evidence URL`, "INVALID_DEPLOYMENT");
    }
    const conflicts = state.deployments.filter(
      (existing) => existing.repo === deploymentRepo && existing.id === item.id,
    );
    if (conflicts.length === 0) state.deployments.push(deploymentReceipt);
    else if (!conflicts.every((existing) => sameReceiptExceptTimestamp(existing, deploymentReceipt, "recordedAt"))) {
      refuse(`deployment ${item.id} conflicts with immutable deployment evidence`, "DEPLOYMENT_RECEIPT_CONFLICT");
    }
  }
  saveState(statePath, state, clock);
  return existingReceipt ?? receipt;
}

export function recordEvidence(statePath, options, dependencies = {}) {
  statePath = assertAbsolutePath(statePath, "state path");
  const clock = dependencies.clock ?? (() => new Date());
  return withStateLock(
    statePath,
    { operation: "record-pr", clock, staleAfterMs: dependencies.lockStaleAfterMs },
    () => recordEvidenceUnlocked(statePath, options, dependencies),
  );
}

function compareCheckRunRecency(left, right) {
  const leftTime = left.started_at ?? left.created_at ?? left.completed_at ?? "";
  const rightTime = right.started_at ?? right.created_at ?? right.completed_at ?? "";
  if (leftTime && rightTime && leftTime !== rightTime) return leftTime.localeCompare(rightTime);
  const leftId = Number.isSafeInteger(left.id) ? left.id : -1;
  const rightId = Number.isSafeInteger(right.id) ? right.id : -1;
  if (leftId !== rightId) return leftId - rightId;
  return leftTime.localeCompare(rightTime);
}

function latestChecksByName(checkRuns) {
  if (!Array.isArray(checkRuns)) refuse("GitHub check runs are not an array", "INVALID_CHECK_RUNS");
  const latest = new Map();
  for (const run of checkRuns) {
    if (run?.app?.id !== GITHUB_ACTIONS_APP_ID || typeof run.name !== "string") continue;
    const current = latest.get(run.name);
    if (!current) {
      latest.set(run.name, run);
      continue;
    }
    const recency = compareCheckRunRecency(run, current);
    if (recency > 0) latest.set(run.name, run);
    if (recency === 0 && JSON.stringify(sortObject(run)) !== JSON.stringify(sortObject(current))) {
      refuse(`GitHub returned ambiguous latest runs for ${run.name}`, "AMBIGUOUS_CHECK_RUN");
    }
  }
  return latest;
}

function missingRequiredChecks(checkRuns, required) {
  const checks = latestChecksByName(checkRuns);
  return required.filter((name) => {
    const check = checks.get(name);
    return check?.status !== "completed" || !ACCEPTED_CHECK_CONCLUSIONS.has(check?.conclusion);
  });
}

function trustedRequiredCheckReceipts(github, slug, sha, required, label) {
  const checkRuns = github.checkRuns(slug, sha);
  const latest = latestChecksByName(checkRuns);
  const missing = missingRequiredChecks(checkRuns, required);
  if (missing.length > 0) {
    refuse(
      `${slug} ${label} ${sha} lacks terminal trusted required checks: ${missing.join(", ")}`,
      "ADOPTED_HEAD_CHECKS_NOT_GREEN",
    );
  }
  return required.map((name) => {
    const run = latest.get(name);
    if (
      !Number.isSafeInteger(run?.id) ||
      run.id <= 0 ||
      !validEvidenceTime(run.completed_at) ||
      typeof run.details_url !== "string" ||
      run.details_url.length === 0
    ) {
      refuse(`${slug} ${label} ${sha} has malformed evidence for ${name}`, "INVALID_CHECK_RECEIPT");
    }
    return {
      name,
      id: run.id,
      appId: run.app.id,
      status: run.status,
      conclusion: run.conclusion,
      completedAt: run.completed_at,
      url: run.details_url,
    };
  });
}

function checkReceiptsStillPresent(checkRuns, receipts) {
  if (!Array.isArray(checkRuns)) return false;
  return receipts.every((receipt) => checkRuns.some((run) =>
    run?.id === receipt.id &&
    run.name === receipt.name &&
    run.app?.id === receipt.appId &&
    run.status === receipt.status &&
    run.conclusion === receipt.conclusion &&
    run.completed_at === receipt.completedAt &&
    run.details_url === receipt.url
  ));
}

function comparisonContains(github, slug, ancestor, descendant) {
  if (!SHA_RE.test(ancestor ?? "") || !SHA_RE.test(descendant ?? "")) return false;
  return new Set(["ahead", "identical"]).has(github.compare(slug, ancestor, descendant)?.status);
}

function assertAdoptedReceiptLive(github, state, receipt) {
  const fixed = REPOSITORIES[receipt.repo];
  const pr = github.pullRequest(fixed.slug, receipt.number);
  if (!exactPullRequestIdentity(pr, fixed, receipt.number)) {
    refuse(`${fixed.slug}#${receipt.number} no longer has its exact repository identity`, "ADOPTION_PR_DRIFT");
  }
  const merge = pullRequestMergeEvidence(pr);
  if (
    pr.state !== "closed" ||
    pr.head.ref !== receipt.head ||
    pr.base.ref !== receipt.base ||
    pr.head.sha !== receipt.headSha ||
    merge.mergeCommit !== receipt.mergeCommit ||
    merge.mergedAt !== receipt.mergedAt
  ) {
    refuse(`${fixed.slug}#${receipt.number} no longer matches its immutable adoption receipt`, "ADOPTION_PR_DRIFT");
  }
  const classification = classifyAdoptedPullRequest(receipt.disposition, pr.head.ref, pr.base.ref);
  if (
    classification.kind !== receipt.kind ||
    (classification.story ?? null) !== receipt.story ||
    (classification.epic !== null && classification.epic !== state.epic) ||
    (classification.slug !== null && classification.slug !== state.slug)
  ) {
    refuse(`${fixed.slug}#${receipt.number} adoption topology changed`, "ADOPTION_TOPOLOGY_DRIFT");
  }
  const timeline = github.timeline(fixed.slug, receipt.number);
  if (
    !Array.isArray(timeline) ||
    !timeline.some(
      (event) => event?.event === receipt.queue.event && event.created_at === receipt.queue.at,
    )
  ) {
    refuse(`${fixed.slug}#${receipt.number} exact merge-queue event is no longer present`, "ADOPTION_QUEUE_DRIFT");
  }
  const checkRuns = github.checkRuns(fixed.slug, receipt.headSha);
  if (
    missingRequiredChecks(checkRuns, state.requiredContexts[receipt.repo]).length > 0 ||
    !checkReceiptsStillPresent(checkRuns, receipt.requiredChecks)
  ) {
    refuse(`${fixed.slug}#${receipt.number} required checks no longer match its receipt`, "ADOPTION_CHECK_DRIFT");
  }
  if (receipt.disposition !== "governance-proof") {
    const adoptedFeatureSha = state.repositories[receipt.repo].adoptedFeatureSha;
    if (
      receipt.canonicalFeatureSha !== adoptedFeatureSha ||
      !comparisonContains(github, fixed.slug, receipt.mergeCommit, adoptedFeatureSha)
    ) {
      refuse(`${fixed.slug}#${receipt.number} is not in the adopted canonical feature history`, "ADOPTION_ANCESTRY_DRIFT");
    }
    if (
      receipt.disposition === "precanonical-story" &&
      !comparisonContains(
        github,
        fixed.slug,
        receipt.mergeCommit,
        state.repositories[receipt.repo].plannedMainSha,
      )
    ) {
      refuse(`${fixed.slug}#${receipt.number} is not in planned main history`, "ADOPTION_MAIN_ANCESTRY_DRIFT");
    }
  }
  return pr;
}

function adoptPullRequestUnlocked(
  statePath,
  { repo, number, disposition },
  { github = new GitHubClient(), clock = () => new Date() } = {},
) {
  statePath = assertAbsolutePath(statePath, "state path");
  const state = loadState(statePath);
  if (state.mode !== "adopt-existing") {
    refuse("adopt-pr requires an explicit adopt-existing plan", "ADOPTION_MODE_REQUIRED");
  }
  repo = normalizeRepoKey(repo);
  if (!Number.isSafeInteger(number) || number <= 0) {
    refuse("PR number must be a positive integer", "INVALID_PR_NUMBER");
  }
  if (!ADOPTION_DISPOSITIONS.has(disposition)) {
    refuse(`unknown adoption disposition ${JSON.stringify(disposition)}`, "INVALID_ADOPTION_DISPOSITION");
  }
  if (state.pullRequests.some((item) => item.repo === repo && item.number === number)) {
    refuse("canonical PR evidence cannot be reclassified as adoption evidence", "ADOPTION_RECEIPT_CONFLICT");
  }
  // Adoption records history but never weakens or changes the protected train itself.
  assertAdoptedPlanValid(state, github);
  const existing = state.adoptedPullRequests.find(
    (item) => item.repo === repo && item.number === number,
  );
  if (existing) {
    if (existing.disposition !== disposition) {
      refuse("an adopted PR disposition is immutable", "ADOPTION_RECEIPT_CONFLICT");
    }
    assertAdoptedReceiptLive(github, state, existing);
    return existing;
  }

  const fixed = REPOSITORIES[repo];
  const pr = github.pullRequest(fixed.slug, number);
  if (!exactPullRequestIdentity(pr, fixed, number)) {
    refuse("adopted PR must have exact same-repository identity", "INVALID_ADOPTION_PR");
  }
  const merge = pullRequestMergeEvidence(pr);
  if (!merge.mergeCommit || pr.state !== "closed") {
    refuse("only an already-merged PR may be adopted", "UNMERGED_ADOPTION_PR");
  }
  const classification = classifyAdoptedPullRequest(disposition, pr.head.ref, pr.base.ref);
  if (
    (classification.epic !== null && classification.epic !== state.epic) ||
    (classification.slug !== null && classification.slug !== state.slug) ||
    (classification.story && !state.expectedStories.includes(classification.story))
  ) {
    refuse("adopted PR does not belong to the state epic and story inventory", "ADOPTION_EPIC_CONFLICT");
  }
  const canonicalFeatureSha = disposition === "governance-proof"
    ? null
    : state.repositories[repo].adoptedFeatureSha;
  if (
    canonicalFeatureSha &&
    !comparisonContains(github, fixed.slug, merge.mergeCommit, canonicalFeatureSha)
  ) {
    refuse("adopted PR merge is not an ancestor of the canonical feature head", "ADOPTION_MISSING_FEATURE_ANCESTRY");
  }
  if (
    disposition === "precanonical-story" &&
    !comparisonContains(
      github,
      fixed.slug,
      merge.mergeCommit,
      state.repositories[repo].plannedMainSha,
    )
  ) {
    refuse("precanonical story merge is not an ancestor of planned main", "ADOPTION_MISSING_MAIN_ANCESTRY");
  }
  const adoptedAt = nowIso(clock);
  if (evidenceTimestamp(adoptedAt) < evidenceTimestamp(merge.mergedAt)) {
    refuse("adoption time predates the historical merge", "INVALID_ADOPTION_TIME");
  }
  const receipt = {
    repo,
    number,
    url: expectedPullRequestUrl(repo, number),
    disposition,
    head: pr.head.ref,
    base: pr.base.ref,
    headSha: pr.head.sha,
    mergeCommit: merge.mergeCommit,
    mergedAt: merge.mergedAt,
    kind: classification.kind,
    story: classification.story ?? null,
    queue: historicalQueueReceipt(
      github.timeline(fixed.slug, number),
      pr.head.sha,
      merge.mergeCommit,
      merge.mergedAt,
    ),
    requiredChecks: trustedRequiredCheckReceipts(
      github,
      fixed.slug,
      pr.head.sha,
      state.requiredContexts[repo],
      `adopted PR #${number}`,
    ),
    canonicalFeatureSha,
    adoptedAt,
  };
  state.adoptedPullRequests.push(receipt);
  saveState(statePath, state, clock);
  return receipt;
}

export function adoptPullRequest(statePath, options, dependencies = {}) {
  statePath = assertAbsolutePath(statePath, "state path");
  const clock = dependencies.clock ?? (() => new Date());
  return withStateLock(
    statePath,
    { operation: "adopt-pr", clock, staleAfterMs: dependencies.lockStaleAfterMs },
    () => adoptPullRequestUnlocked(statePath, options, dependencies),
  );
}

function assertRequiredMainChecks(github, repositories, requiredContexts) {
  const failures = [];
  for (const [key, fixed] of Object.entries(REPOSITORIES)) {
    const sha = repositories[key].plannedMainSha;
    const missing = missingRequiredChecks(github.checkRuns(fixed.slug, sha), requiredContexts[key]);
    if (missing.length > 0) {
      failures.push(`${fixed.slug} main ${sha}: ${missing.join(", ")}`);
    }
  }
  if (failures.length > 0) {
    refuse(
      `exact main commits lack terminal trusted required checks: ${failures.join("; ")}`,
      "MAIN_CHECKS_NOT_GREEN",
    );
  }
}

function auditPr(github, state, receipt) {
  const slug = REPOSITORIES[receipt.repo].slug;
  const pr = github.pullRequest(slug, receipt.number);
  const unchanged =
    pr?.head?.sha === receipt.headSha && pr?.head?.ref === receipt.head && pr?.base?.ref === receipt.base;
  const merged = Boolean(unchanged && pr?.merged_at && SHA_RE.test(pr?.merge_commit_sha ?? ""));
  const required = state.requiredContexts[receipt.repo];
  const missing = unchanged ? missingRequiredChecks(github.checkRuns(slug, receipt.headSha), required) : required;
  const timeline = unchanged ? github.timeline(slug, receipt.number) : [];
  const queue = queueEvidenceSinceReceipt(timeline, receipt.recordedAt);
  const queueEvidence = Boolean(queue.evidenceAt) && !queue.unverifiable;
  return {
    repo: receipt.repo,
    number: receipt.number,
    url: receipt.url,
    head: receipt.head,
    base: receipt.base,
    headSha: receipt.headSha,
    unchanged,
    merged,
    mergeCommit: merged ? pr.merge_commit_sha : null,
    headGreen: missing.length === 0,
    missingContexts: missing,
    queueEvidence,
    queueEvidenceAt: queueEvidence ? queue.evidenceAt : null,
    queueEvidenceVerifiable: !queue.unverifiable,
  };
}

function auditAdoptedPr(github, state, receipt) {
  let error = null;
  try {
    assertAdoptedReceiptLive(github, state, receipt);
  } catch (caught) {
    error = caught.message;
  }
  return {
    repo: receipt.repo,
    number: receipt.number,
    url: receipt.url,
    disposition: receipt.disposition,
    head: receipt.head,
    base: receipt.base,
    headSha: receipt.headSha,
    mergeCommit: receipt.mergeCommit,
    mergedAt: receipt.mergedAt,
    kind: receipt.kind,
    story: receipt.story,
    canonicalFeatureSha: receipt.canonicalFeatureSha,
    immutableEvidenceValid: error === null,
    merged: error === null,
    headGreen: error === null,
    queueEvidence: error === null,
    error,
  };
}

function refClaimsEpicTrain(state, ref) {
  if (typeof ref !== "string") return false;
  if (ref === state.featureBranch) return true;
  const epicNumber = state.epic.slice(3);
  const epicClaim = new RegExp(`(?:^|[^0-9])(?:sc-)?${epicNumber}(?:[^0-9]|$)`);
  if (/^(?:feature|story|sync)\//.test(ref) && epicClaim.test(ref)) return true;
  if (!ref.startsWith("story/")) return false;
  return state.expectedStories.some((story) => {
    const storyNumber = story.slice(3);
    return new RegExp(`(?:^|[^0-9])sc-${storyNumber}(?:[^0-9]|$)`).test(ref);
  });
}

function possiblyBelongsToEpicTrain(state, pullRequest) {
  return (
    refClaimsEpicTrain(state, pullRequest?.head?.ref) ||
    refClaimsEpicTrain(state, pullRequest?.base?.ref)
  );
}

function discoverLiveTrainPullRequests(github, state) {
  const discovered = [];
  for (const [repo, fixed] of Object.entries(REPOSITORIES)) {
    const owner = fixed.slug.split("/")[0];
    const baseMatches = github.pullRequests(fixed.slug, { state: "all", base: state.featureBranch });
    const headMatches = github.pullRequests(fixed.slug, {
      state: "all",
      head: `${owner}:${state.featureBranch}`,
    });
    const repositoryWide = github.pullRequests(fixed.slug, { state: "all" });
    const numbers = new Set([
      ...baseMatches.map(({ number }) => number),
      ...headMatches.map(({ number }) => number),
      ...repositoryWide.filter((summary) => possiblyBelongsToEpicTrain(state, summary))
        .map(({ number }) => number),
    ]);
    for (const number of numbers) {
      const pullRequest = github.pullRequest(fixed.slug, number);
      if (
        !pullRequest ||
        pullRequest.number !== number ||
        typeof pullRequest.state !== "string" ||
        !["open", "closed"].includes(pullRequest.state) ||
        typeof pullRequest.head?.ref !== "string" ||
        typeof pullRequest.base?.ref !== "string" ||
        typeof pullRequest.head?.repo?.full_name !== "string" ||
        typeof pullRequest.base?.repo?.full_name !== "string" ||
        (pullRequest.merged_at !== null &&
          pullRequest.merged_at !== undefined &&
          (!Number.isFinite(evidenceTimestamp(pullRequest.merged_at)) ||
            !SHA_RE.test(pullRequest.merge_commit_sha ?? ""))) ||
        (pullRequest.state === "open" && pullRequest.merged_at != null)
      ) {
        refuse(
          `${fixed.slug}#${number} returned malformed live pull-request evidence`,
          "INVALID_PULL_REQUEST",
        );
      }
      let classification = null;
      let topologyError = null;
      const canonicalReceipt = state.pullRequests.find(
        (item) => item.repo === repo && item.number === pullRequest.number,
      );
      const adoptedReceipt = state.adoptedPullRequests.find(
        (item) => item.repo === repo && item.number === pullRequest.number,
      );
      try {
        classification = adoptedReceipt
          ? classifyAdoptedPullRequest(
            adoptedReceipt.disposition,
            pullRequest.head.ref,
            pullRequest.base.ref,
          )
          : classifyPullRequest(pullRequest.head.ref, pullRequest.base.ref);
        if (
          pullRequest.head.repo.full_name !== fixed.slug ||
          pullRequest.base.repo.full_name !== fixed.slug
        ) {
          topologyError = "pull request head and base must belong to the fixed same repository";
        } else if (
          adoptedReceipt &&
          (pullRequest.head.sha !== adoptedReceipt.headSha ||
            pullRequest.head.ref !== adoptedReceipt.head ||
            pullRequest.base.ref !== adoptedReceipt.base)
        ) {
          topologyError = "pull request differs from its explicit immutable adoption receipt";
        } else if (
          classification.kind === "ordinary" ||
          (classification.epic !== null && classification.epic !== state.epic) ||
          (classification.slug !== null && classification.slug !== state.slug) ||
          (classification.story && !state.expectedStories.includes(classification.story))
        ) {
          topologyError = "pull request is not a valid member of this feature train";
        }
      } catch (error) {
        topologyError = error.message;
      }
      const receipt = canonicalReceipt ?? adoptedReceipt;
      const merged = Boolean(pullRequest.merged_at && SHA_RE.test(pullRequest.merge_commit_sha ?? ""));
      discovered.push({
        repo,
        number: pullRequest.number,
        url: pullRequest.html_url ?? null,
        head: pullRequest.head.ref,
        base: pullRequest.base.ref,
        state: pullRequest.state,
        merged,
        mergeCommit: merged ? pullRequest.merge_commit_sha : null,
        kind: classification?.kind ?? null,
        story: classification?.story ?? null,
        recorded: Boolean(receipt),
        receiptType: canonicalReceipt ? "canonical" : adoptedReceipt ? "adopted" : null,
        disposition: adoptedReceipt?.disposition ?? null,
        topologyError,
      });
    }
  }
  discovered.sort((left, right) => left.repo.localeCompare(right.repo) || left.number - right.number);
  const discoveredKeys = new Set(discovered.map(({ repo, number }) => `${repo}:${number}`));
  const missingReceipts = [...state.pullRequests, ...state.adoptedPullRequests]
    .filter(({ repo, number }) => !discoveredKeys.has(`${repo}:${number}`))
    .map(({ repo, number }) => ({ repo, number }));
  const open = discovered.filter(({ state: prState }) => prState === "open");
  const unrecordedMerged = discovered.filter(({ merged, recorded }) => merged && !recorded);
  const liveTopologyErrors = discovered.filter(
    ({ state: prState, merged, topologyError }) => topologyError && (prState === "open" || merged),
  );
  return {
    pullRequests: discovered,
    open,
    unrecordedMerged,
    missingReceipts,
    liveTopologyErrors,
    gates: {
      noOpenTrainPullRequests: open.length === 0,
      allMergedTrainPullRequestsRecorded: unrecordedMerged.length === 0,
      allReceiptsDiscovered: missingReceipts.length === 0,
      noLiveTopologyErrors: liveTopologyErrors.length === 0,
    },
  };
}

function auditStateUnlocked(
  statePath,
  reportPath,
  { github = new GitHubClient(), shortcut = new ShortcutClient(), clock = () => new Date() } = {},
) {
  statePath = assertAbsolutePath(statePath, "state path");
  reportPath = assertAbsolutePath(reportPath, "report path");
  if (resolve(statePath) === resolve(reportPath)) refuse("report path must differ from state path", "REPORT_STATE_CONFLICT");
  const state = loadState(statePath);
  const shortcutStatus = inspectShortcutEpic(shortcut, state.epic, state.expectedStories);
  const shortcutWorkflowMapMatchesFrozen = sameJson(
    shortcutStatus.workflowStates,
    state.shortcut.workflowStates,
  );
  const refs = liveRefs(state, github);
  const mains = Object.fromEntries(
    Object.entries(REPOSITORIES).map(([key, fixed]) => [key, github.mainSha(fixed.slug)]),
  );
  const policies = {};
  for (const key of Object.keys(REPOSITORIES)) {
    try {
      const live = auditFeaturePolicies(github, key, state.featureBranch, state.epic);
      const wildcardBaseMatches =
        live.wildcardBase.id === state.policies[key].wildcardBase.id &&
        live.wildcardBase.digest === state.policies[key].wildcardBase.digest;
      const deletionGuardMatches =
        live.deletionGuard.id === state.policies[key].deletionGuard.id &&
        live.deletionGuard.digest === state.policies[key].deletionGuard.digest;
      const exactQueueMatches =
        Boolean(live.exactQueue) &&
        live.exactQueue.id === state.policies[key].exactQueue.id &&
        live.exactQueue.enforcement === "active" &&
        live.exactQueue.digest === state.policies[key].exactQueue.digest;
      let effective = null;
      let effectiveError = null;
      if (refs[key] && exactQueueMatches) {
        try {
          effective = verifyEffectiveFeaturePolicies(github, key, state.featureBranch, live);
        } catch (error) {
          effectiveError = error.message;
        }
      }
      const unsafeFeatureRef = Boolean(refs[key] && !exactQueueMatches);
      policies[key] = {
        wildcardBase: { ...live.wildcardBase, matchesPlan: wildcardBaseMatches },
        deletionGuard: { ...live.deletionGuard, matchesPlan: deletionGuardMatches },
        exactQueue: live.exactQueue
          ? { ...live.exactQueue, matchesPlan: exactQueueMatches }
          : { id: null, name: state.policies[key].exactQueue.name, digest: null, matchesPlan: false },
        effective,
        unsafeFeatureRef,
        ok:
          wildcardBaseMatches &&
          deletionGuardMatches &&
          exactQueueMatches &&
          Boolean(refs[key]) &&
          Boolean(effective) &&
          !unsafeFeatureRef,
        error: effectiveError,
      };
    } catch (error) {
      policies[key] = {
        wildcardBase: null,
        deletionGuard: null,
        exactQueue: null,
        effective: null,
        unsafeFeatureRef: Boolean(refs[key]),
        ok: false,
        error: error.message,
      };
    }
  }
  const prs = state.pullRequests.map((receipt) => auditPr(github, state, receipt));
  const adoptedPrs = state.adoptedPullRequests.map(
    (receipt) => auditAdoptedPr(github, state, receipt),
  );
  const livePullRequests = discoverLiveTrainPullRequests(github, state);
  const storyCoverage = Object.fromEntries(
    state.expectedStories.map((story) => [
      story,
      prs.some((pr) => pr.merged && state.pullRequests.some(
        (receipt) => receipt.repo === pr.repo && receipt.number === pr.number && receipt.story === story,
      )) || adoptedPrs.some(
        (pr) =>
          pr.immutableEvidenceValid &&
          pr.merged &&
          pr.kind === "story" &&
          pr.disposition !== "governance-proof" &&
          pr.story === story,
      ),
    ]),
  );
  const finalPrs = Object.fromEntries(
    Object.keys(REPOSITORIES).map((key) => {
      const record = state.finalPullRequests[key];
      if (!record) return [key, { required: true, recorded: false, merged: false }];
      const audit = prs.find((item) => item.repo === key && item.number === record.number);
      return [key, { required: true, recorded: true, ...audit }];
    }),
  );
  const ancestry = {};
  for (const [key, fixed] of Object.entries(REPOSITORIES)) {
    const featureSha = refs[key];
    const mainSha = mains[key];
    const immutableBaselineSha = immutableFeatureBaseline(state.repositories[key]);
    const plannedMainSha = state.repositories[key].plannedMainSha;
    let currentMainIncluded = false;
    let plannedMainIncluded = false;
    let immutableBaselineIncluded = false;
    if (featureSha && mainSha) {
      const comparison = github.compare(fixed.slug, mainSha, featureSha);
      currentMainIncluded = comparison?.status === "ahead" || comparison?.status === "identical";
    }
    if (featureSha) {
      plannedMainIncluded = comparisonContains(github, fixed.slug, plannedMainSha, featureSha);
      immutableBaselineIncluded = comparisonContains(
        github,
        fixed.slug,
        immutableBaselineSha,
        featureSha,
      );
    }
    ancestry[key] = {
      mainSha,
      plannedMainSha,
      immutableBaselineSha,
      featureSha,
      currentMainIncluded,
      plannedMainIncluded,
      immutableBaselineIncluded,
    };
  }

  const adoptionBaseline = {
    required: state.mode === "adopt-existing",
    repositories: {},
    pin: { exact: state.mode !== "adopt-existing", error: null },
  };
  for (const [key, fixed] of Object.entries(REPOSITORIES)) {
    if (state.mode !== "adopt-existing") {
      adoptionBaseline.repositories[key] = { required: false, ok: true };
      continue;
    }
    const repoState = state.repositories[key];
    const checkRuns = github.checkRuns(fixed.slug, repoState.adoptedFeatureSha);
    const missing = missingRequiredChecks(checkRuns, state.requiredContexts[key]);
    const exactReceipts = checkReceiptsStillPresent(checkRuns, repoState.adoptedRequiredChecks);
    adoptionBaseline.repositories[key] = {
      required: true,
      plannedMainSha: repoState.plannedMainSha,
      adoptedFeatureSha: repoState.adoptedFeatureSha,
      liveFeatureSha: refs[key],
      plannedMainIncluded: comparisonContains(
        github,
        fixed.slug,
        repoState.plannedMainSha,
        repoState.adoptedFeatureSha,
      ),
      adoptedHeadIncluded: ancestry[key].immutableBaselineIncluded,
      requiredChecksExact: missing.length === 0 && exactReceipts,
      missingContexts: missing,
      policiesExactAndEffective: policies[key].ok,
    };
    adoptionBaseline.repositories[key].ok = Object.entries(adoptionBaseline.repositories[key])
      .filter(([field]) => [
        "plannedMainIncluded",
        "adoptedHeadIncluded",
        "requiredChecksExact",
        "policiesExactAndEffective",
      ].includes(field))
      .every(([, value]) => value === true);
  }
  if (state.mode === "adopt-existing") {
    try {
      const baselinePin = extractInferencePin(
        github.cargoManifests(
          REPOSITORIES.sceneworks.slug,
          state.repositories.sceneworks.adoptedFeatureSha,
        ),
      );
      adoptionBaseline.pin = {
        sha: baselinePin.sha,
        matchesReceipt: sameJson(baselinePin, state.inferencePin),
        reachableFromPlannedInferenceMain: isReachableFromMain(
          github,
          baselinePin.sha,
          state.repositories.inference.plannedMainSha,
        ),
        reachableFromAdoptedInferenceHead: isReachableFromMain(
          github,
          baselinePin.sha,
          state.repositories.inference.adoptedFeatureSha,
        ),
        exact: false,
        error: null,
      };
      adoptionBaseline.pin.exact =
        adoptionBaseline.pin.matchesReceipt &&
        adoptionBaseline.pin.reachableFromPlannedInferenceMain &&
        adoptionBaseline.pin.reachableFromAdoptedInferenceHead;
    } catch (error) {
      adoptionBaseline.pin = { exact: false, error: error.message };
    }
  }
  adoptionBaseline.ok =
    Object.values(adoptionBaseline.repositories).every(({ ok }) => ok) &&
    adoptionBaseline.pin.exact;

  const pinRevision = finalPrs.sceneworks?.merged
    ? finalPrs.sceneworks.mergeCommit
    : state.finalPullRequests.sceneworks?.headSha ?? refs.sceneworks ?? mains.sceneworks;
  const liveManifests = github.cargoManifests(REPOSITORIES.sceneworks.slug, pinRevision);
  let pin;
  let pinReachable = false;
  let pinError = null;
  try {
    pin = extractInferencePin(liveManifests);
    pinReachable = isReachableFromMain(github, pin.sha, mains.inference);
  } catch (error) {
    pinError = error.message;
  }
  const finalInferenceMergeCommit = finalPrs.inference?.merged
    ? finalPrs.inference.mergeCommit
    : null;
  const pinMatchesFinalInferenceMerge = Boolean(
    pin?.sha && finalInferenceMergeCommit && pin.sha === finalInferenceMergeCommit,
  );

  const runtime = state.privilegedWorkflows.map((requirement) => {
    const receipt = state.privilegedRuns.find(
      (item) => item.repo === requirement.repo && item.workflow === requirement.workflow,
    );
    if (!receipt) return { ...requirement, recorded: false, exactHead: false, successful: false, url: null };
    const run = github.workflowRun(REPOSITORIES[receipt.repo].slug, receipt.id);
    return {
      ...requirement,
      recorded: true,
      evidencePhase: receipt.evidencePhase ?? null,
      exactHead: run?.head_sha === state.finalPullRequests[receipt.finalPrRepo]?.headSha,
      successful: run?.status === "completed" && run?.conclusion === "success",
      explicitDispatch: run?.event === "workflow_dispatch",
      url: receipt.url,
    };
  });
  const deployments = state.deployments.map((receipt) => {
    const deployment = github.deployment(REPOSITORIES[receipt.repo].slug, receipt.id);
    const statuses = github.deploymentStatuses(REPOSITORIES[receipt.repo].slug, receipt.id);
    return {
      ...receipt,
      exactHead: deployment?.sha === state.finalPullRequests[receipt.finalPrRepo]?.mergeCommit,
      successful: statuses[0]?.state === "success",
    };
  });
  const deployment = {
    required: state.deploymentRequired,
    satisfied: !state.deploymentRequired || deployments.some((item) => item.exactHead && item.successful),
    receipts: deployments,
  };
  const gates = {
    shortcutInventoryMatches: shortcutStatus.gates.inventoryMatches,
    shortcutWorkflowMapMatchesFrozen,
    shortcutStoriesDone: shortcutStatus.gates.allStoriesDone,
    shortcutNoBlockedStories: shortcutStatus.gates.noBlockedStories,
    shortcutNoUnresolvedBlockers: shortcutStatus.gates.noUnresolvedBlockers,
    shortcutEpicNotArchived: shortcutStatus.gates.epicNotArchived,
    noOpenTrainPullRequests: livePullRequests.gates.noOpenTrainPullRequests,
    allMergedTrainPullRequestsRecorded: livePullRequests.gates.allMergedTrainPullRequestsRecorded,
    allReceiptsDiscovered: livePullRequests.gates.allReceiptsDiscovered,
    noLiveTrainTopologyErrors: livePullRequests.gates.noLiveTopologyErrors,
    featurePoliciesVerified: Object.values(policies).every(({ ok }) => ok),
    immutableFeatureBaselinesIncluded: Object.values(ancestry).every(
      ({ immutableBaselineIncluded }) => immutableBaselineIncluded,
    ),
    adoptedBaselineVerified: adoptionBaseline.ok,
    allExpectedStoriesMerged: Object.values(storyCoverage).every(Boolean),
    allRecordedPrsMerged: [...prs, ...adoptedPrs].every((pr) => pr.merged),
    allRecordedPrHeadsGreen: [...prs, ...adoptedPrs].every((pr) => pr.headGreen),
    allRecordedPrsHaveQueueEvidence: [...prs, ...adoptedPrs].every((pr) => pr.queueEvidence),
    allAdoptedReceiptsExact: adoptedPrs.every((pr) => pr.immutableEvidenceValid),
    finalPullRequestsMerged: Object.values(finalPrs).every((pr) => pr.merged),
    featureRefsMatchFinalPrHeads: Object.keys(REPOSITORIES).every(
      (key) => finalPrs[key]?.merged && refs[key] === state.finalPullRequests[key]?.headSha,
    ),
    featureBranchesIncludeCurrentMain: Object.values(ancestry).every(
      (item) => item.currentMainIncluded || Object.values(finalPrs).every((pr) => pr.merged),
    ),
    finalPinExactAndReachable: Boolean(pin?.sha && pinReachable && pinMatchesFinalInferenceMerge),
    finalPinMatchesInferenceMerge: pinMatchesFinalInferenceMerge,
    privilegedRuntimeExactHead: runtime.every(
      (item) => item.recorded && item.exactHead && item.successful && item.explicitDispatch,
    ),
    deploymentSatisfied: deployment.satisfied,
  };
  const cleanupEligible = Object.values(gates).every(Boolean);
  const report = {
    schemaVersion: STATE_VERSION,
    mode: state.mode,
    epic: state.epic,
    featureBranch: state.featureBranch,
    generatedAt: nowIso(clock),
    bootstrap: {
      plannedMainShas: Object.fromEntries(
        Object.entries(state.repositories).map(([key, repo]) => [key, repo.plannedMainSha]),
      ),
      adoptedFeatureShas: Object.fromEntries(
        Object.entries(state.repositories).map(([key, repo]) => [key, repo.adoptedFeatureSha]),
      ),
      refs,
      workspaces: Object.fromEntries(
        Object.entries(state.repositories).map(([key, repo]) => [key, repo.workspace]),
      ),
      policies,
    },
    shortcut: shortcutStatus,
    expectedStories: storyCoverage,
    pullRequests: prs,
    adoptedPullRequests: adoptedPrs,
    livePullRequests,
    finalPullRequests: finalPrs,
    requiredContexts: state.requiredContexts,
    ancestry,
    adoptionBaseline,
    inferencePin: {
      revision: pinRevision,
      sha: pin?.sha ?? null,
      reachableFromInferenceMain: pinReachable,
      finalInferenceMergeCommit,
      matchesFinalInferenceMerge: pinMatchesFinalInferenceMerge,
      error: pinError,
    },
    privilegedRuntime: runtime,
    deployment,
    gates,
    cleanup: {
      eligible: cleanupEligible,
      action: "report-only",
      refs,
      exactQueueRulesets: Object.fromEntries(
        Object.keys(REPOSITORIES).map((key) => [
          key,
          {
            id: policies[key].exactQueue?.id ?? state.policies[key].exactQueue.id,
            name: state.policies[key].exactQueue.name,
            digest: state.policies[key].exactQueue.digest,
          },
        ]),
      ),
      note: "This audit never deletes refs or opens a ruleset bypass window.",
    },
    ok: cleanupEligible,
  };
  writeJsonAtomic(reportPath, report);
  return report;
}

export function auditState(statePath, reportPath, dependencies = {}) {
  statePath = assertAbsolutePath(statePath, "state path");
  const clock = dependencies.clock ?? (() => new Date());
  return withStateLock(
    statePath,
    { operation: "audit", clock, staleAfterMs: dependencies.lockStaleAfterMs },
    () => auditStateUnlocked(statePath, reportPath, dependencies),
  );
}

export function parseReceipt(value, label) {
  const match = /^(sceneworks|inference):([1-9][0-9]*)$/.exec(value ?? "");
  if (!match) refuse(`${label} must be repo:positive-id`, "INVALID_RECEIPT");
  return { repo: match[1], id: Number(match[2]) };
}
