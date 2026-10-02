import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

import { findActionPinViolations } from "./lib/action-pins.mjs";

const workflowPath = ".github/workflows/publish-runpod.yml";
const [workflow, dockerfile, serverCandle, desktopWindows, candleSmoke] = await Promise.all([
  readFile(workflowPath, "utf8"),
  readFile("docker/rust.Dockerfile", "utf8"),
  readFile(".github/workflows/server-candle-linux.yml", "utf8"),
  readFile(".github/workflows/desktop-windows.yml", "utf8"),
  readFile("scripts/check-docker-candle-runtime.sh", "utf8"),
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
  "actions/checkout@",
  "actions/setup-node@",
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
  "npm run check",
  "npm run check:runpod:image-contract",
  "scripts/check-gen-core-skew.sh --self-test",
  "sceneworks-worker --features backend-candle",
  "sceneworks-rust-api --features embed-web,backend-candle",
]) {
  requireText(workflow, contract);
}

function validateCandleRuntimeGate(candidate) {
  const steps = candidate.split(/(?=^      - name: )/m);
  const byId = (id) => {
    const matches = steps.filter((step) => new RegExp(`^        id: ${id}$`, "m").test(step));
    assert.equal(matches.length, 1, `expected one ${id} step`);
    return matches[0];
  };
  const combined = byId("build_staging");
  const stagedCheck = byId("verify_staging");
  const candle = byId("build_candle_runtime");
  const smoke = byId("smoke_candle_runtime");
  const combinedSmoke = byId("smoke_published_runtime");
  const promote = byId("promote");
  requireText(combined, "target: runpod");
  requireText(combined, "platforms: linux/amd64");
  requireText(combined, "push: true");
  requireText(combined, "tags: ${{ env.IMAGE_NAME }}:staging-${{ github.run_id }}-${{ github.run_attempt }}");
  requireText(combined, "cache-to: type=gha,mode=max,scope=runpod-amd64");
  assert.ok(!combined.includes("load: true"), "the staged registry digest, not a separately loaded image, must be tested");
  assert.ok(!combined.includes("steps.metadata.outputs.tags"), "release/manual tags must not be exposed before runtime gates pass");
  requireText(stagedCheck, "${{ steps.build_staging.outputs.digest }}");
  requireText(stagedCheck, 'docker pull --platform linux/amd64 "$staged_ref"');
  requireText(stagedCheck, "scripts/check-runpod-staged-image.mjs");
  requireText(stagedCheck, '"$GITHUB_SHA"');
  for (const [step, target] of [[candle, "rust-worker-candle"]]) {
    requireText(step, `target: ${target}`);
    requireText(step, "platforms: linux/amd64");
    requireText(step, "push: false");
    requireText(step, "load: true");
    assert.ok(!step.includes("no-cache: true"), "acceptance builds must share cached CUDA stages");
  }
  requireText(smoke, "bash scripts/check-docker-candle-runtime.sh");
  requireText(smoke, "SCENEWORKS_CANDLE_SMOKE_API_IMAGE: ${{ env.IMAGE_NAME }}@${{ steps.build_staging.outputs.digest }}");
  requireText(smoke, "SCENEWORKS_CANDLE_SMOKE_WORKER_IMAGE: sceneworks-candle-smoke:ci");
  requireText(smoke, "SCENEWORKS_CANDLE_SMOKE_DRIVER_DIR: /usr/local/cuda/compat");
  requireText(smoke, "timeout-minutes: 5");
  requireText(combinedSmoke, "SCENEWORKS_CANDLE_SMOKE_API_IMAGE: ${{ env.IMAGE_NAME }}@${{ steps.build_staging.outputs.digest }}");
  requireText(combinedSmoke, "SCENEWORKS_CANDLE_SMOKE_WORKER_IMAGE: ${{ env.IMAGE_NAME }}@${{ steps.build_staging.outputs.digest }}");
  requireText(combinedSmoke, "SCENEWORKS_CANDLE_SMOKE_WORKER_ENTRYPOINT: /usr/local/bin/sceneworks-rust-worker");
  requireText(combinedSmoke, "SCENEWORKS_CANDLE_SMOKE_WORKER_USER: 1000:1000");
  requireText(combinedSmoke, "SCENEWORKS_CANDLE_SMOKE_DRIVER_DIR: /usr/local/cuda/compat");
  requireText(candleSmoke, 'worker_entrypoint="${SCENEWORKS_CANDLE_SMOKE_WORKER_ENTRYPOINT:-}"');
  requireText(candleSmoke, 'worker_default_user="${SCENEWORKS_CANDLE_SMOKE_WORKER_USER:-}"');
  requireText(candleSmoke, 'worker_entrypoint_args=(--entrypoint "${worker_entrypoint}")');
  requireText(candleSmoke, 'user_args=(--user "${worker_default_user}")');
  requireText(promote, "if: success()");
  requireText(promote, "STANDALONE_SMOKE_RESULT: ${{ steps.smoke_candle_runtime.outcome }}");
  requireText(promote, "PUBLISHED_SMOKE_RESULT: ${{ steps.smoke_published_runtime.outcome }}");
  requireText(promote, "run: node scripts/promote-runpod-image.mjs");
  assert.ok(!promote.includes("docker/build-push-action"), "promotion must reuse the staged manifest instead of building");
  const runpodBuilds = steps.filter((step) => /^          target: runpod$/m.test(step));
  assert.equal(runpodBuilds.length, 1, "combined RunPod target must be built exactly once");
  assert.ok(steps.indexOf(combined) < steps.indexOf(stagedCheck));
  assert.ok(steps.indexOf(stagedCheck) < steps.indexOf(candle));
  assert.ok(steps.indexOf(candle) < steps.indexOf(smoke));
  assert.ok(steps.indexOf(smoke) < steps.indexOf(combinedSmoke));
  assert.ok(steps.indexOf(combinedSmoke) < steps.indexOf(promote), "all staged-image runtime checks must pass before final tags are promoted");
  assert.equal(candidate.match(/uses: docker\/setup-buildx-action@/g)?.length, 1,
    "the standalone CUDA image and single combined image must share one builder");
}
validateCandleRuntimeGate(workflow);
for (const mutated of [
  workflow.replace("target: rust-worker-candle", "target: rust-worker"),
  workflow.replace("id: smoke_candle_runtime", "id: skipped_smoke"),
  workflow.replace("load: true", "load: false"),
  workflow.replace("push: true\n          tags: ${{ env.IMAGE_NAME }}:staging-", "push: false\n          tags: ${{ env.IMAGE_NAME }}:staging-"),
  workflow.replace("SCENEWORKS_CANDLE_SMOKE_DRIVER_DIR: /usr/local/cuda/compat", "SCENEWORKS_CANDLE_SMOKE_DRIVER_DIR: /missing-driver"),
  workflow.replace("SCENEWORKS_CANDLE_SMOKE_API_IMAGE: ${{ env.IMAGE_NAME }}@${{ steps.build_staging.outputs.digest }}", "SCENEWORKS_CANDLE_SMOKE_API_IMAGE: ${{ env.IMAGE_NAME }}:staging-${{ github.run_id }}-${{ github.run_attempt }}"),
  workflow.replace("if: success()\n        env:\n          IMAGE_NAME:", "if: always()\n        env:\n          IMAGE_NAME:"),
  workflow.replace("run: node scripts/promote-runpod-image.mjs", "uses: docker/build-push-action@deadbeef"),
]) assert.throws(() => validateCandleRuntimeGate(mutated));

// Pin *shape* is enforced repo-wide by scripts/check-action-pins.mjs, which stays
// true across Dependabot bumps. Naming exact SHAs here only re-broke this check
// every time Dependabot rewrote the workflow it was guarding.
assert.deepEqual(
  findActionPinViolations(workflow, workflowPath),
  [],
  "publication workflow actions must be pinned to immutable commits",
);

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
// sc-17879: this lane used to inject an inference read token twice -- once as a `url.…insteadOf`
// rewrite for the host `cargo fetch`, once as a BuildKit secret the Dockerfile turned into the same
// rewrite. `SceneWorks/inference` is public and always was, so both bought nothing; the guard is now
// that NEITHER exists, in either place. Both fetches resolve the pinned revision anonymously.
assert.ok(
  !/SCENEWORKS_INFERENCE_READ_TOKEN/.test(workflow),
  "the inference repo is public; this workflow must carry no inference credential (sc-17879)",
);
assert.ok(
  !/inference_token/.test(workflow) && !/inference_token/.test(dockerfile),
  "the vestigial inference BuildKit secret must not come back (sc-17879)",
);
assert.ok(
  !/x-access-token:/.test(workflow) && !/x-access-token:/.test(dockerfile),
  "no credential rewrite for the public inference repo, in the workflow or the Dockerfile (sc-17879)",
);
assert.ok(
  !/(?:echo|printf)[^\n]*\$\{\{\s*(?:secrets\.|github\.token)/.test(workflow),
  "publication workflow must never print an inference credential",
);

const runtimeCuda = /FROM nvidia\/cuda:([0-9.]+)-runtime-ubuntu24\.04 AS rust-worker-candle-base/.exec(
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
