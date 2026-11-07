import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";
import test from "node:test";
import { historyKey, historyRequirements, prepareHistory } from "./prepare-inference-history.mjs";

const git = (args) => execFileSync("git", args, { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] }).trim();
test("cache request binds current pin, all anchor revisions and both StarVector validator revisions", () => {
  const requested = historyRequirements(process.cwd());
  const anchors = execFileSync(process.execPath, ["scripts/anchor-loader-closure.mjs", "--anchor-revisions"], { encoding: "utf8" }).trim().split(/\s+/);
  const contract = JSON.parse(readFileSync("release/starvector-terminal-campaign-v1.json", "utf8")).inference_contract;
  for (const revision of [requested.pin, contract.revision, contract.validator_source.revision, contract.validator_source.base_revision, ...anchors]) assert.ok(requested.revisions.includes(revision), revision);
  assert.equal(requested.terminal, "0b084cbf46b5b4f4f561305ef116268b57eecf7f");
  assert.notEqual(historyKey(requested), historyKey({ ...requested, revisions: requested.revisions.slice(1) }));
  assert.equal(new Set(requested.revisions).size, requested.revisions.length);
});
test("cold and warm history keep all requested refs; a new set fetches only missing revisions", () => {
  const root = mkdtempSync(path.join(tmpdir(), "check-history-"));
  try {
    const origin = path.join(root, "origin"); mkdirSync(origin);
    git(["init", "-q", origin]); git(["-C", origin, "config", "user.email", "fixture@example.invalid"]); git(["-C", origin, "config", "user.name", "Fixture"]);
    const revisions = [];
    for (const value of ["one", "two", "three", "four"]) {
      writeFileSync(path.join(origin, "source"), value); git(["-C", origin, "add", "source"]); git(["-C", origin, "commit", "-qm", value]); revisions.push(git(["-C", origin, "rev-parse", "HEAD"]));
    }
    const cache = path.join(root, "cache.git");
    const options = { cache, origin: pathToFileURL(origin).href, requirements: { pin: revisions[3], terminal: revisions[1], revisions: [revisions[0], revisions[1], revisions[3]] } };
    // Force pack-based fetches and aggressive detached maintenance. Real Git trace events expose
    // actual housekeeping children, rather than merely asserting that a flag was constructed.
    git(["init", "--bare", "-q", cache]);
    for (const [key, value] of [["fetch.unpackLimit", "1"], ["gc.autoPackLimit", "1"], ["gc.auto", "1"], ["gc.autoDetach", "true"], ["maintenance.auto", "true"]]) git(["-C", cache, "config", key, value]);
    const trace = path.join(root, "git-trace.jsonl");
    let fetches = 0;
    const run = (command, args, rest) => {
      if (args.includes("fetch")) fetches++;
      return execFileSync(command, args, { ...rest, env: { ...process.env, GIT_TRACE2_EVENT: trace } });
    };
    function prepare(label, requirements = options.requirements) {
      const inference = path.join(root, `inference-${label}`), terminal = path.join(root, `terminal-${label}`);
      prepareHistory({ ...options, requirements, inference, terminal, run });
      assert.equal(git(["-C", inference, "rev-parse", "HEAD"]), requirements.pin);
      assert.equal(git(["-C", terminal, "rev-parse", "HEAD"]), requirements.terminal);
      for (const revision of requirements.revisions) {
        assert.equal(git(["-C", cache, "rev-parse", `refs/ci-history/${revision}`]), revision);
        git(["-C", inference, "show", `${revision}:source`]);
      }
    }
    prepare("cold"); assert.equal(fetches, 1, "fetch all missing exact refs in one operation");
    prepare("warm"); assert.equal(fetches, 1, "valid cache must do no network fetch");
    const expanded = { ...options.requirements, revisions: [...revisions] };
    prepare("expanded", expanded); assert.equal(fetches, 2);
    // A stale/wrong ref cannot be trusted just because the cache key matched.
    git(["-C", cache, "update-ref", `refs/ci-history/${revisions[0]}`, revisions[1]]);
    prepare("repaired", expanded); assert.equal(fetches, 3);
    const housekeeping = readFileSync(trace, "utf8").trim().split("\n").map((line) => JSON.parse(line))
      .filter((event) => event.event === "child_start" && event.argv.some((arg) => arg === "maintenance" || arg === "gc"));
    assert.deepEqual(housekeeping, [], "history preparation must not launch a shallow-file maintenance writer");
    // Positive control: the trace must detect Git's actual default post-fetch maintenance. Keep
    // its GC threshold disabled so this control has no detached process to race fixture cleanup.
    const control = path.join(root, "control.git"), controlTrace = path.join(root, "control-trace.jsonl");
    git(["init", "--bare", "-q", control]);
    execFileSync("git", ["-c", "core.hooksPath=/dev/null", "-c", "gc.auto=0", "-c", "maintenance.auto=true", "-C", control,
      "fetch", "--no-tags", "--depth=2", options.origin, revisions[3]],
    { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"], env: { ...process.env, GIT_TRACE2_EVENT: controlTrace } });
    const controlChildren = readFileSync(controlTrace, "utf8").trim().split("\n").map((line) => JSON.parse(line));
    assert.ok(controlChildren.some((event) => event.event === "child_start" && event.argv.some((arg) => arg === "maintenance" || arg === "gc")),
      "the real Git trace control must observe default post-fetch housekeeping");
  } finally { rmSync(root, { recursive: true, force: true }); }
});
