import assert from "node:assert/strict";
import test from "node:test";
import { execFileSync } from "node:child_process";
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { validatePrebuiltImageConfig, validatePrebuiltWorkerSources } from "./check-runpod-prebuilt-inputs.mjs";

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

const dockerfile = `# --- Candle GPU worker build\nCOPY crates ./crates\nCOPY apps/web/content-security-policy.txt ./apps/web/content-security-policy.txt\n# --- Combined RunPod GPU runtime\nFROM prebuilt AS runpod\n`;

function sourceRepository(changedPath, changedContents) {
  const cwd = mkdtempSync(path.join(tmpdir(), "runpod-prebuilt-inputs-"));
  const git = (...args) => execFileSync("git", args, { cwd, encoding: "utf8" }).trim();
  const write = (file, contents) => {
    const destination = path.join(cwd, file);
    mkdirSync(path.dirname(destination), { recursive: true });
    writeFileSync(destination, contents);
  };
  git("init", "-q");
  for (const file of ["crates/sceneworks-core/src/lib.rs", "apps/web/src/ImageStudio.jsx",
    "apps/web/content-security-policy.txt", "packages/schemas/model.json", ".dockerignore"]) write(file, "original");
  write("docker/rust.Dockerfile", dockerfile);
  git("add", ".");
  git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "baseline");
  const revision = git("rev-parse", "HEAD");
  write(changedPath, changedContents);
  git("add", ".");
  git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "changed input");
  return { cwd, revision, currentDockerfile: readFileSync(path.join(cwd, "docker/rust.Dockerfile"), "utf8") };
}

function withSourceRepository(changedPath, changedContents, check) {
  const source = sourceRepository(changedPath, changedContents);
  try {
    check(source);
  } finally {
    rmSync(source.cwd, { recursive: true, force: true });
  }
}

test("web source drift keeps the immutable Candle worker eligible", () => {
  withSourceRepository("apps/web/src/ImageStudio.jsx", "changed UI", ({ cwd, revision, currentDockerfile }) => {
    assert.doesNotThrow(() => validatePrebuiltWorkerSources(revision, currentDockerfile, cwd));
  });
});

test("copied worker source, CSP, schema, and context-filter drift refuse the prebuilt worker", () => {
  for (const file of ["crates/sceneworks-core/src/lib.rs", "apps/web/content-security-policy.txt",
    "packages/schemas/model.json", ".dockerignore"]) {
    withSourceRepository(file, "changed input", ({ cwd, revision, currentDockerfile }) => {
      assert.throws(() => validatePrebuiltWorkerSources(revision, currentDockerfile, cwd),
        /prebuilt worker inputs differ/);
    });
  }
});

test("Candle Docker stage drift refuses while RunPod wrapper drift remains eligible", () => {
  withSourceRepository("docker/rust.Dockerfile", dockerfile.replace("COPY crates ./crates", "COPY crates ./worker-crates"),
    ({ cwd, revision, currentDockerfile }) => {
      assert.throws(() => validatePrebuiltWorkerSources(revision, currentDockerfile, cwd),
        /Candle builder\/runtime stages differ/);
    });
  withSourceRepository("docker/rust.Dockerfile", dockerfile.replace("FROM prebuilt AS runpod", "FROM current AS runpod"),
    ({ cwd, revision, currentDockerfile }) => {
      assert.doesNotThrow(() => validatePrebuiltWorkerSources(revision, currentDockerfile, cwd));
    });
});
