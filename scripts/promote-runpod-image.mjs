import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { appendFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const DIGEST = /^sha256:[a-f0-9]{64}$/;

function digest(bytes) {
  return `sha256:${createHash("sha256").update(bytes).digest("hex")}`;
}

function parseManifest(raw, description) {
  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch (error) {
    throw new Error(`${description} is not a JSON OCI manifest: ${error.message}`);
  }
  assert.equal(parsed.schemaVersion, 2, `${description} must use OCI schema version 2`);
  return parsed;
}

function amd64Descriptor(index, description) {
  const matches = (index.manifests ?? []).filter(
    (item) => item.platform?.os === "linux" && item.platform?.architecture === "amd64",
  );
  assert.equal(matches.length, 1, `${description} must contain exactly one linux/amd64 image manifest`);
  assert.match(matches[0].digest, DIGEST, `${description} has an invalid amd64 manifest digest`);
  return matches[0];
}

function parseTags(value, repository) {
  const tags = value.split(/\r?\n/).map((tag) => tag.trim()).filter(Boolean);
  assert.ok(tags.length > 0, "metadata must provide at least one final image tag");
  for (const tag of tags) {
    assert.ok(tag.startsWith(`${repository}:`), `refusing to promote outside ${repository}: ${tag}`);
    assert.ok(!tag.includes("@"), `promotion destinations must be tags: ${tag}`);
    assert.ok(!tag.includes(":staging-"), `refusing to promote onto a staging tag: ${tag}`);
  }
  assert.equal(new Set(tags).size, tags.length, "metadata contains duplicate promotion tags");
  return tags;
}

export function promotionPlan({ repository, digest: expectedDigest, tags, smokeResults }) {
  assert.ok(smokeResults.length > 0, "at least one required staged-image smoke result is required");
  assert.ok(
    smokeResults.every((result) => result === "success"),
    `refusing publication unless every smoke passed (got: ${smokeResults.join(", ")})`,
  );
  assert.match(expectedDigest, DIGEST, "staged Buildx output must be a SHA-256 digest");
  const destinations = parseTags(tags, repository);
  return { sourceRef: `${repository}@${expectedDigest}`, destinations, expectedDigest };
}

export async function promoteRunpodImage({ repository, expectedDigest, tags, smokeResults, run }) {
  const plan = promotionPlan({ repository, digest: expectedDigest, tags, smokeResults });
  const sourceRaw = run(["buildx", "imagetools", "inspect", "--raw", plan.sourceRef]);
  assert.equal(digest(sourceRaw), expectedDigest, "immutable staged reference does not match Buildx output digest");
  const sourceIndex = parseManifest(sourceRaw, "staged image index");
  const sourceAmd64 = amd64Descriptor(sourceIndex, "staged image index");
  const sourceManifestRaw = run([
    "buildx",
    "imagetools",
    "inspect",
    "--raw",
    `${repository}@${sourceAmd64.digest}`,
  ]);
  assert.equal(digest(sourceManifestRaw), sourceAmd64.digest, "staged AMD64 manifest digest does not match its descriptor");
  const sourceManifest = parseManifest(sourceManifestRaw, "staged AMD64 manifest");
  assert.match(sourceManifest.config?.digest ?? "", DIGEST, "staged AMD64 config digest is invalid");

  run([
    "buildx",
    "imagetools",
    "create",
    ...plan.destinations.flatMap((destination) => ["--tag", destination]),
    plan.sourceRef,
  ]);
  const promoted = [];
  for (const destination of plan.destinations) {
    const publishedRaw = run(["buildx", "imagetools", "inspect", "--raw", destination]);
    const publishedIndex = parseManifest(publishedRaw, `promoted tag ${destination}`);
    const publishedAmd64 = amd64Descriptor(publishedIndex, `promoted tag ${destination}`);
    assert.equal(
      publishedAmd64.digest,
      sourceAmd64.digest,
      `promoted tag ${destination} does not retain the staged AMD64 manifest`,
    );
    const publishedManifestRaw = run([
      "buildx",
      "imagetools",
      "inspect",
      "--raw",
      `${repository}@${publishedAmd64.digest}`,
    ]);
    assert.equal(digest(publishedManifestRaw), sourceAmd64.digest, `promoted AMD64 bytes changed for ${destination}`);
    const publishedManifest = parseManifest(publishedManifestRaw, `promoted AMD64 manifest ${destination}`);
    assert.equal(
      publishedManifest.config?.digest,
      sourceManifest.config.digest,
      `promoted tag ${destination} changed the staged image config`,
    );
    promoted.push({ tag: destination, indexDigest: digest(publishedRaw), amd64ManifestDigest: publishedAmd64.digest, configDigest: publishedManifest.config.digest });
  }
  return {
    stagingDigest: expectedDigest,
    amd64ManifestDigest: sourceAmd64.digest,
    configDigest: sourceManifest.config.digest,
    promoted,
  };
}

function main() {
  const repository = process.env.IMAGE_NAME;
  const expectedDigest = process.env.STAGED_DIGEST;
  const tags = process.env.PUBLISH_TAGS ?? "";
  const smokeResults = [process.env.STANDALONE_SMOKE_RESULT, process.env.PUBLISHED_SMOKE_RESULT];
  assert.ok(repository, "IMAGE_NAME is required");
  const resultPromise = promoteRunpodImage({
    repository,
    expectedDigest,
    tags,
    smokeResults,
    run: (args) => execFileSync("docker", args, { encoding: "utf8" }),
  });
  resultPromise.then((result) => {
    const summary = [
      "### RunPod image promoted from the tested staged artifact",
      "",
      `- Staging index: \`${repository}@${result.stagingDigest}\``,
      `- AMD64 manifest: \`${result.amd64ManifestDigest}\``,
      `- Config: \`${result.configDigest}\``,
      ...result.promoted.map((item) => `- Promoted \`${item.tag}\` (index \`${item.indexDigest}\`, AMD64 \`${item.amd64ManifestDigest}\`, config \`${item.configDigest}\`)`),
      "",
      "Every promoted tag was read back and its AMD64 manifest and config matched the staged artifact.",
      "",
    ].join("\n");
    if (process.env.GITHUB_STEP_SUMMARY) appendFileSync(process.env.GITHUB_STEP_SUMMARY, summary);
    if (process.env.GITHUB_OUTPUT) {
      appendFileSync(
        process.env.GITHUB_OUTPUT,
        `staging_digest=${result.stagingDigest}\namd64_manifest_digest=${result.amd64ManifestDigest}\nconfig_digest=${result.configDigest}\n`,
      );
    }
    process.stdout.write(summary);
  }).catch((error) => {
    process.stderr.write(`${error.stack ?? error.message}\n`);
    process.exitCode = 1;
  });
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main();
