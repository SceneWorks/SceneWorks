#!/usr/bin/env node
// Fail closed before publishing with a prebuilt CUDA worker from another source tree.
import assert from "node:assert/strict";
import { execFileSync, spawnSync } from "node:child_process";
import { appendFileSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

export function validatePrebuiltImageConfig(inspected) {
  const object = (value) => value !== null && typeof value === "object" && !Array.isArray(value);
  assert.ok(object(inspected), "prebuilt worker needs one OCI image config, not a platform map or array");
  // Buildx .Image returns lowercase OCI fields for a single image. Multi-platform
  // maps are deliberately refused instead of choosing a platform.
  assert.equal(inspected.os, "linux", "prebuilt worker OCI os must be linux");
  assert.equal(inspected.architecture, "amd64", "prebuilt worker OCI architecture must be amd64");
  assert.ok(object(inspected.config), "prebuilt worker needs an OCI config object");
  assert.ok(Array.isArray(inspected.config.Env) && inspected.config.Env.every((entry) => typeof entry === "string"),
    "prebuilt worker OCI config.Env must be an array of strings");
  assert.deepEqual(inspected.config.Env.filter((entry) => entry.startsWith("CUDA_VERSION=")), ["CUDA_VERSION=12.9.1"],
    "prebuilt worker must retain exactly one CUDA_VERSION=12.9.1");
  assert.ok(object(inspected.config.Labels), "prebuilt worker OCI config.Labels must be an object");
  const revision = inspected.config.Labels["org.opencontainers.image.revision"];
  assert.match(revision ?? "", /^[0-9a-f]{40}$/, "prebuilt worker needs an exact source revision label");
  return revision;
}

function main() {
  const dockerfile = readFileSync("docker/rust.Dockerfile", "utf8");
  const image = /^ARG RUNPOD_PREBUILT_CANDLE_IMAGE=(ghcr\.io\/sceneworks\/sceneworks-runpod@sha256:[0-9a-f]{64})$/m.exec(dockerfile)?.[1];
  assert.ok(image, "RunPod prebuilt worker must use an immutable GHCR digest");
  assert.match(dockerfile, /^FROM \$\{RUNPOD_PREBUILT_CANDLE_IMAGE\} AS runpod$/m);

  // Inspect the registry's OCI config without a daemon or downloading layers.
  // Publication still pulls and tests the exact worker/combined runtime images.
  const inspected = JSON.parse(execFileSync("docker", ["buildx", "imagetools", "inspect", image,
    "--format", "{{json .Image}}"], { encoding: "utf8" }));
  const revision = validatePrebuiltImageConfig(inspected);
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
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main();
