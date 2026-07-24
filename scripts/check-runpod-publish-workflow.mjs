import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

const workflowPath = ".github/workflows/publish-runpod.yml";
const [workflow, dockerfile, serverCandle, desktopWindows] = await Promise.all([
  readFile(workflowPath, "utf8"),
  readFile("docker/rust.Dockerfile", "utf8"),
  readFile(".github/workflows/server-candle-linux.yml", "utf8"),
  readFile(".github/workflows/desktop-windows.yml", "utf8"),
]);

function requireText(body, text, message = `missing ${text}`) {
  assert.ok(body.includes(text), message);
}

function topLevelSection(name, nextNames) {
  const next = nextNames.join("|");
  const match = new RegExp(`^${name}:\\r?\\n([\\s\\S]*?)(?=^(?:${next}):)`, "m").exec(workflow);
  assert.ok(match, `${workflowPath} is missing top-level ${name}`);
  return match[1];
}

const triggers = topLevelSection("on", ["permissions", "concurrency", "env", "jobs"]);
requireText(triggers, "push:");
requireText(triggers, '"v*"');
requireText(triggers, "workflow_dispatch:");
requireText(triggers, "image_tag:");
assert.ok(!triggers.includes("pull_request:"), "heavy publication workflow must not run on PRs");
assert.ok(!/^\s{2}branches:/m.test(triggers), "heavy publication workflow must not run on branches");

const permissions = topLevelSection("permissions", ["concurrency", "env", "jobs"]);
assert.match(permissions, /^\s{2}contents: read$/m);
assert.match(permissions, /^\s{2}packages: write$/m);
assert.equal(
  permissions.match(/^\s{2}[a-z-]+:/gm)?.length,
  2,
  "publication workflow permissions must remain minimal",
);

for (const contract of [
  "ghcr.io/sceneworks/sceneworks-runpod",
  "docker/login-action@",
  "registry: ghcr.io",
  "username: ${{ github.actor }}",
  "password: ${{ secrets.GITHUB_TOKEN }}",
  "docker/setup-buildx-action@",
  "docker/metadata-action@",
  "docker/build-push-action@",
  "file: docker/rust.Dockerfile",
  "target: runpod",
  "platforms: linux/amd64",
  "push: true",
  "inference_token=${{ secrets.SCENEWORKS_INFERENCE_READ_TOKEN || github.token }}",
  "x-access-token:${{ secrets.SCENEWORKS_INFERENCE_READ_TOKEN || github.token }}@github.com",
  "npm run check",
  "npm run check:docker:runpod",
  "scripts/check-gen-core-skew.sh --self-test",
  "sceneworks-worker --features backend-candle",
  "sceneworks-rust-api --features embed-web,backend-candle",
]) {
  requireText(workflow, contract);
}

for (const [action, version, sha] of [
  ["actions/checkout", "v7.0.1", "3d3c42e5aac5ba805825da76410c181273ba90b1"],
  ["actions/setup-node", "v7.0.0", "820762786026740c76f36085b0efc47a31fe5020"],
  ["docker/login-action", "v4.5.1", "abd2ef45e78c5afb21d64d4ca52ee8550d9572c7"],
  ["docker/setup-buildx-action", "v4.2.0", "bb05f3f5519dd87d3ba754cc423b652a5edd6d2c"],
  ["docker/metadata-action", "v6.2.0", "dc802804100637a589fabce1cb79ff13a1411302"],
  ["docker/build-push-action", "v7.3.0", "53b7df96c91f9c12dcc8a07bcb9ccacbed38856a"],
]) {
  requireText(
    workflow,
    `uses: ${action}@${sha} # ${version}`,
    `${action} must use the reviewed immutable ${version} commit`,
  );
}

assert.match(workflow, /\^refs\/tags\/v\(\[0-9\]\+\\\.\[0-9\]\+\\\.\[0-9\]\+\)\$/);
requireText(workflow, 'type=raw,value=latest,enable=${{ steps.publication.outputs.release == \'true\' }}');
requireText(workflow, "^manual-[a-z0-9][a-z0-9._-]{0,119}$");
requireText(workflow, 'tag="manual-${GITHUB_SHA:0:12}"');
requireText(workflow, "flavor: latest=false");
assert.equal(
  workflow.match(/value=latest/g)?.length,
  1,
  "latest must have exactly one guarded metadata rule",
);
assert.ok(
  !/build-args:[\s\S]*SCENEWORKS_INFERENCE_READ_TOKEN/.test(workflow),
  "private inference token must be a BuildKit secret, never a build arg",
);
assert.equal(
  workflow.match(/secrets\.SCENEWORKS_INFERENCE_READ_TOKEN \|\| github\.token/g)?.length,
  2,
  "host fetch and BuildKit must both use the dedicated-token/GITHUB_TOKEN fallback",
);
assert.ok(
  !/(?:echo|printf)[^\n]*\$\{\{\s*(?:secrets\.|github\.token)/.test(workflow),
  "publication workflow must never print an inference credential",
);

const runtimeCuda = /FROM nvidia\/cuda:([0-9.]+)-runtime-ubuntu24\.04 AS rust-worker-candle/.exec(
  dockerfile,
)?.[1];
const builderCuda = /FROM nvidia\/cuda:([0-9.]+)-devel-ubuntu22\.04 AS candle-builder/.exec(
  dockerfile,
)?.[1];
const serverCuda = /^\s+cuda:\s*"([^"]+)"/m.exec(serverCandle)?.[1];
assert.equal(runtimeCuda, "12.9.1", "RunPod runtime must retain the CUDA 12.9.1 pin");
assert.equal(builderCuda, runtimeCuda, "RunPod builder/runtime CUDA pins must stay aligned");
assert.equal(serverCuda, runtimeCuda, "RunPod and server candle CUDA pins must stay aligned");
requireText(desktopWindows, "CUDA 12.9 Toolkit");

console.log("SceneWorks RunPod publication workflow contract check passed.");
