import assert from "node:assert/strict";
import test from "node:test";
import { validatePrebuiltImageConfig } from "./check-runpod-prebuilt-inputs.mjs";

const revision = "6435067bd5c57aab4e7f0e8722bd53ea6c39ba92";
const config = () => ({ os: "linux", architecture: "amd64", config: {
  Env: ["PATH=/usr/local/bin", "CUDA_VERSION=12.9.1"],
  Labels: { "org.opencontainers.image.revision": revision },
} });

test("accepts Buildx single-image OCI config and returns its exact worker revision", () => {
  assert.equal(validatePrebuiltImageConfig(config()), revision);
});

test("refuses missing, malformed, Docker-inspect, and ambiguous multi-platform config", () => {
  for (const value of [null, [], "config", {}, { "linux/amd64": config() },
    { "linux/amd64": config(), "linux/arm64": { ...config(), architecture: "arm64" } },
    [config()], { Os: "linux", Architecture: "amd64", Config: config().config }]) {
    assert.throws(() => validatePrebuiltImageConfig(value));
  }
});

test("refuses other platforms and missing OCI configuration fields", () => {
  for (const patch of [{ os: "windows" }, { architecture: "arm64" }, { os: "Linux" },
    { architecture: null }, { config: null }, { config: [] }, { config: {} }]) {
    assert.throws(() => validatePrebuiltImageConfig({ ...config(), ...patch }));
  }
});

test("refuses absent, malformed, changed, or ambiguous CUDA metadata", () => {
  for (const Env of [undefined, null, "CUDA_VERSION=12.9.1", [null], [],
    ["CUDA_VERSION=12.8.1"], ["CUDA_VERSION=12.9.1", "CUDA_VERSION=12.8.1"],
    ["CUDA_VERSION=12.9.1", "CUDA_VERSION=12.9.1"]]) {
    const value = config(); value.config.Env = Env;
    assert.throws(() => validatePrebuiltImageConfig(value));
  }
});

test("refuses absent or malformed provenance including nonexact source revisions", () => {
  for (const Labels of [undefined, null, [], "label", {},
    ...["main", "6435067", revision.toUpperCase(), `${revision}\n`, 123, null].map((value) => ({ "org.opencontainers.image.revision": value }))]) {
    const value = config(); value.config.Labels = Labels;
    assert.throws(() => validatePrebuiltImageConfig(value));
  }
});
