import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import test from "node:test";

import { validateStagedImage } from "./check-runpod-staged-image.mjs";
import { promoteRunpodImage } from "./promote-runpod-image.mjs";

const repository = "ghcr.io/sceneworks/sceneworks-runpod";
const revision = "e42cd40d5312374214b85b616eb078c455a40560";
const sourceHashes = {
  "docker/runpod-entrypoint.sh": "1".repeat(64),
  "docker/runpod-privileges.sh": "2".repeat(64),
};
const installedHashes = {
  "/usr/local/bin/sceneworks-runpod-entrypoint": sourceHashes["docker/runpod-entrypoint.sh"],
  "/usr/local/bin/runpod-privileges.sh": sourceHashes["docker/runpod-privileges.sh"],
};
const inspectJson = {
  Architecture: "amd64",
  Os: "linux",
  Config: {
    Entrypoint: ["/usr/local/bin/sceneworks-runpod-entrypoint"],
    Labels: { "org.opencontainers.image.revision": revision },
  },
};

function sha256(value) {
  return `sha256:${createHash("sha256").update(value).digest("hex")}`;
}

function fixtureManifests() {
  const imageManifest = JSON.stringify({
    schemaVersion: 2,
    mediaType: "application/vnd.oci.image.manifest.v1+json",
    config: { mediaType: "application/vnd.oci.image.config.v1+json", digest: `sha256:${"c".repeat(64)}`, size: 2 },
    layers: [{ mediaType: "application/vnd.oci.image.layer.v1.tar+gzip", digest: `sha256:${"d".repeat(64)}`, size: 3 }],
  });
  const imageDigest = sha256(imageManifest);
  const index = JSON.stringify({
    schemaVersion: 2,
    mediaType: "application/vnd.oci.image.index.v1+json",
    manifests: [{
      mediaType: "application/vnd.oci.image.manifest.v1+json",
      digest: imageDigest,
      size: Buffer.byteLength(imageManifest),
      platform: { architecture: "amd64", os: "linux" },
    }],
  });
  return { imageManifest, imageDigest, index, indexDigest: sha256(index) };
}

test("staged artifact is immutable, source-bound, and contains the source helper bytes", () => {
  assert.equal(validateStagedImage({
    image: `${repository}@sha256:${"a".repeat(64)}`,
    revision,
    inspectJson,
    installedHashes,
    sourceHashes,
  }), true);
  assert.throws(() => validateStagedImage({
    image: `${repository}@sha256:${"a".repeat(64)}`,
    revision: "f".repeat(40),
    inspectJson,
    installedHashes,
    sourceHashes,
  }), /revision label/);
  assert.throws(() => validateStagedImage({
    image: `${repository}@sha256:${"a".repeat(64)}`,
    revision,
    inspectJson: { ...inspectJson, Config: { ...inspectJson.Config, User: "1000:1000" } },
    installedHashes,
    sourceHashes,
  }), /root-only initialization/);
  assert.throws(() => validateStagedImage({
    image: `${repository}@sha256:${"a".repeat(64)}`,
    revision,
    inspectJson,
    installedHashes: { ...installedHashes, "/usr/local/bin/runpod-privileges.sh": "3".repeat(64) },
    sourceHashes,
  }), /must match docker\/runpod-privileges.sh/);
  assert.throws(() => validateStagedImage({
    image: `${repository}:mutable-staging-tag`,
    revision,
    inspectJson,
    installedHashes,
    sourceHashes,
  }), /immutable digest/);
});

test("promotion uses one staged digest and verifies every promoted AMD64 manifest", async () => {
  const fixture = fixtureManifests();
  const calls = [];
  const outputs = new Map([
    [`${repository}@${fixture.indexDigest}`, fixture.index],
    [`${repository}@${fixture.imageDigest}`, fixture.imageManifest],
  ]);
  const result = await promoteRunpodImage({
    repository,
    expectedDigest: fixture.indexDigest,
    tags: `${repository}:1.2.3\n${repository}:latest`,
    smokeResults: ["success", "success"],
    run: (args) => {
      calls.push(args);
      if (args[2] === "create") {
        for (let index = 3; index < args.length - 1; index += 2) outputs.set(args[index + 1], fixture.index);
        return "";
      }
      const ref = args.at(-1);
      assert.ok(outputs.has(ref), `unexpected registry readback ${ref}`);
      return outputs.get(ref);
    },
  });
  assert.equal(result.stagingDigest, fixture.indexDigest);
  assert.equal(result.amd64ManifestDigest, fixture.imageDigest);
  assert.equal(result.promoted.length, 2);
  const promotionCall = calls.find((args) => args[2] === "create");
  assert.deepEqual(promotionCall.slice(3, -1), ["--tag", `${repository}:1.2.3`, "--tag", `${repository}:latest`]);
  assert.equal(promotionCall.at(-1), `${repository}@${fixture.indexDigest}`);
});

test("failed smoke blocks promotion before any registry operation", async () => {
  let called = false;
  await assert.rejects(promoteRunpodImage({
    repository,
    expectedDigest: `sha256:${"a".repeat(64)}`,
    tags: `${repository}:1.2.3`,
    smokeResults: ["success", "failure"],
    run: () => { called = true; return ""; },
  }), /refusing publication unless every smoke passed/);
  assert.equal(called, false);
});

test("promotion rejects a changed AMD64 manifest after tag creation", async () => {
  const fixture = fixtureManifests();
  const different = JSON.stringify({ ...JSON.parse(fixture.imageManifest), config: { digest: `sha256:${"e".repeat(64)}` } });
  const differentIndex = JSON.stringify({
    schemaVersion: 2,
    mediaType: "application/vnd.oci.image.index.v1+json",
    manifests: [{
      mediaType: "application/vnd.oci.image.manifest.v1+json",
      digest: sha256(different),
      size: Buffer.byteLength(different),
      platform: { architecture: "amd64", os: "linux" },
    }],
  });
  const refs = new Map([
    [`${repository}@${fixture.indexDigest}`, fixture.index],
    [`${repository}@${fixture.imageDigest}`, fixture.imageManifest],
  ]);
  await assert.rejects(promoteRunpodImage({
    repository,
    expectedDigest: fixture.indexDigest,
    tags: `${repository}:1.2.3`,
    smokeResults: ["success", "success"],
    run: (args) => {
      if (args[2] === "create") {
        refs.set(args[4], differentIndex);
        return "";
      }
      const ref = args.at(-1);
      return refs.get(ref) ?? (ref === `${repository}@${sha256(different)}` ? different : "");
    },
  }), /does not retain the staged AMD64 manifest/);
});
