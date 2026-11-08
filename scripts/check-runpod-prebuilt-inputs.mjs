#!/usr/bin/env node
// Fail closed before publishing with a prebuilt CUDA worker from another source tree.
import assert from "node:assert/strict";
import { execFileSync, spawnSync } from "node:child_process";
import { appendFileSync, readFileSync } from "node:fs";

const dockerfile = readFileSync("docker/rust.Dockerfile", "utf8");
const image = /^ARG RUNPOD_PREBUILT_CANDLE_IMAGE=(ghcr\.io\/sceneworks\/sceneworks-runpod@sha256:[0-9a-f]{64})$/m.exec(dockerfile)?.[1];
assert.ok(image, "RunPod prebuilt worker must use an immutable GHCR digest");
assert.match(dockerfile, /^FROM \$\{RUNPOD_PREBUILT_CANDLE_IMAGE\} AS runpod$/m);

execFileSync("docker", ["pull", "--platform", "linux/amd64", image], { stdio: "pipe" });
const [inspected] = JSON.parse(execFileSync("docker", ["image", "inspect", image], { encoding: "utf8" }));
assert.equal(`${inspected.Os}/${inspected.Architecture}`, "linux/amd64");
assert.ok(inspected.Config.Env.includes("CUDA_VERSION=12.9.1"), "prebuilt worker must retain CUDA 12.9.1");
const revision = inspected.Config.Labels?.["org.opencontainers.image.revision"];
assert.match(revision ?? "", /^[0-9a-f]{40}$/, "prebuilt worker needs an exact source revision label");
const hasRevision = spawnSync("git", ["cat-file", "-e", `${revision}^{commit}`]);
if (hasRevision.status !== 0) {
  execFileSync("git", ["-c", "gc.auto=0", "-c", "maintenance.auto=false",
    "fetch", "--no-tags", "--depth=1", "origin", revision], { stdio: "pipe" });
}

// These are the files copied into the Candle build stage, including the
// worker's embedded source/configuration inputs. Keep the list conservative:
// an unrelated change may require a fresh prebuilt, but a changed worker must
// never be released with an old executable. Workflow-only changes are safe.
const workerInputs = [
  "Cargo.toml", "Cargo.lock", "rust-toolchain.toml", "rustfmt.toml",
  ".cargo", "vendor", "crates", "apps/rust-api", "apps/rust-worker",
  "apps/desktop/Cargo.toml", "apps/desktop/build.rs", "apps/web/src",
  "apps/web/content-security-policy.txt", "config", "docs/generated",
  "docs/calibration",
];
const diff = spawnSync("git", ["diff", "--quiet", revision, "HEAD", "--", ...workerInputs]);
assert.equal(diff.status, 0, `prebuilt worker inputs differ from ${revision} (git diff exit ${diff.status})`);

// The Docker builder, CUDA flags, ORT packaging and runtime base must match too.
const sourceDockerfile = execFileSync("git", ["show", `${revision}:docker/rust.Dockerfile`], { encoding: "utf8" });
const workerStages = (source) => {
  const start = source.indexOf("# --- Candle GPU worker build");
  const end = source.indexOf("# --- Combined RunPod GPU runtime", start);
  assert.ok(start >= 0 && end > start, "missing Candle Docker stages");
  return source.slice(start, end);
};
assert.equal(workerStages(dockerfile), workerStages(sourceDockerfile),
  "Candle builder/runtime stages differ from the prebuilt worker source");

const outputIndex = process.argv.indexOf("--github-output");
if (outputIndex >= 0) {
  const outputPath = process.argv[outputIndex + 1];
  assert.ok(outputPath, "--github-output requires a path");
  appendFileSync(outputPath, `image=${image}\nrevision=${revision}\n`, "utf8");
}
console.log(`RunPod prebuilt worker matches ${revision} (${image})`);
