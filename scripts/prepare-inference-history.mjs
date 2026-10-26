// Cache only immutable Git objects and named revision refs; test worktrees are rebuilt per run.
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { appendFileSync, copyFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { inferencePinFromCargo } from "./inference-closure-digest.mjs";

const ORIGIN = "https://github.com/SceneWorks/inference.git";
export function historyRequirements(root, run = execFileSync) {
  const pin = inferencePinFromCargo(readFileSync(path.join(root, "Cargo.toml"), "utf8"));
  const contract = JSON.parse(readFileSync(path.join(root, "release/starvector-terminal-campaign-v1.json"), "utf8")).inference_contract;
  const terminal = contract.validator_source?.revision ?? contract.revision;
  const anchorRevisions = run(process.execPath, ["scripts/anchor-loader-closure.mjs", "--anchor-revisions"], { cwd: root, encoding: "utf8" }).trim().split(/\s+/);
  const revisions = [...new Set([pin, terminal, contract.revision, contract.validator_source?.base_revision ?? contract.revision, ...anchorRevisions])].sort();
  if (revisions.some((revision) => !/^[0-9a-f]{40}$/.test(revision))) throw new Error("invalid immutable inference revision");
  return { pin, terminal, revisions };
}
export function historyKey(requirements) {
  return createHash("sha256").update(JSON.stringify(requirements)).digest("hex");
}

export function prepareHistory({ requirements, cache, inference, terminal, origin = ORIGIN, run = execFileSync }) {
  const git = (args) => run("git", args, { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] }).trim();
  mkdirSync(path.dirname(cache), { recursive: true });
  if (!existsSync(cache)) git(["init", "--bare", cache]);
  if (git(["-C", cache, "rev-parse", "--is-bare-repository"]) !== "true") throw new Error("history cache must be a bare repository");
  // Never execute hooks or resolve a URL/config inherited from an older cache entry.
  const cachedGit = (args) => git(["-c", "core.hooksPath=/dev/null", "-c", "transfer.fsckObjects=true", "-C", cache, ...args]);
  const missing = requirements.revisions.filter((revision) => {
    try { return cachedGit(["rev-parse", "--verify", `refs/ci-history/${revision}^{commit}`]) !== revision; }
    catch { return true; }
  });
  if (missing.length) {
    // Each fetch may start detached auto-GC, whose shallow pruning can race the next fetch's
    // shallow-file stat check (Git 2.55). One atomic fetch owns all boundary/ref updates; disable
    // its automatic maintenance so no background writer races verification or checkout creation.
    cachedGit(["fetch", "--no-tags", "--no-auto-maintenance", "--atomic", "--depth=2", origin,
      ...missing.map((revision) => `+${revision}:refs/ci-history/${revision}`)]);
  }
  for (const revision of requirements.revisions) {
    if (cachedGit(["rev-parse", "--verify", `refs/ci-history/${revision}^{commit}`]) !== revision) throw new Error(`history ref mismatch: ${revision}`);
    cachedGit(["cat-file", "-e", `${revision}^{tree}`]);
  }
  cachedGit(["fsck", "--connectivity-only", "--no-dangling"]);
  for (const [destination, revision] of [[inference, requirements.pin], [terminal, requirements.terminal]]) {
    if (existsSync(destination)) throw new Error(`test checkout already exists: ${destination}`);
    // clone --shared cannot preserve arbitrary retained refs from a shallow bare source. Create
    // a clean local checkout explicitly, sharing only its object database and shallow boundaries.
    git(["init", "-q", destination]);
    writeFileSync(path.join(destination, ".git/objects/info/alternates"), `${path.resolve(cache, "objects").replaceAll("\\", "/")}\n`);
    if (existsSync(path.join(cache, "shallow"))) copyFileSync(path.join(cache, "shallow"), path.join(destination, ".git/shallow"));
    for (const required of requirements.revisions) git(["-C", destination, "update-ref", `refs/ci-history/${required}`, required]);
    git(["-c", "core.hooksPath=/dev/null", "-C", destination, "checkout", "--detach", revision]);
    if (git(["-C", destination, "rev-parse", "HEAD"]) !== revision) throw new Error("test checkout revision mismatch");
    for (const required of requirements.revisions) git(["-C", destination, "cat-file", "-e", `${required}^{commit}`]);
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const root = process.cwd();
  const requirements = historyRequirements(root);
  if (process.argv.includes("--key")) {
    const output = `key=${historyKey(requirements)}\n`;
    if (process.env.GITHUB_OUTPUT) appendFileSync(process.env.GITHUB_OUTPUT, output);
    else process.stdout.write(output);
  } else {
    const temporary = process.env.RUNNER_TEMP;
    if (!temporary) throw new Error("RUNNER_TEMP is required");
    prepareHistory({ requirements, cache: path.join(temporary, "inference-history.git"), inference: path.join(temporary, "inference"), terminal: path.join(temporary, "starvector-pinned-inference") });
    if (process.env.GITHUB_ENV) appendFileSync(process.env.GITHUB_ENV, `INFERENCE_REPO=${path.join(temporary, "inference")}\nSTARVECTOR_TERMINAL_INFERENCE_TEST_ROOT=${path.join(temporary, "starvector-pinned-inference")}\n`);
  }
}
