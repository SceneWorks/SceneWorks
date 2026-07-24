import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

const templatePath = "config/runpod-template.json";
const docPath = "docs/deploy-runpod.md";
const [templateText, doc, workflow, readme] = await Promise.all([
  readFile(templatePath, "utf8"),
  readFile(docPath, "utf8"),
  readFile(".github/workflows/publish-runpod.yml", "utf8"),
  readFile("README.md", "utf8"),
]);

const template = JSON.parse(templateText);
const validationDigest =
  "sha256:fdd60e35655708915ea046a9db86093360a81c2946f08fe63cef58f59f9ab065";
const releaseRepository = "ghcr.io/sceneworks/sceneworks-runpod";
const validationTag = "manual-sc10367-2c4dd777560a";
const accessSecret = "{{ RUNPOD_SECRET_sceneworks_access_token }}";

function validateTemplate(candidate) {
  assert.equal(candidate.category, "NVIDIA", "template must request an NVIDIA Pod");
  assert.equal(candidate.isServerless, false, "template must describe a Pod, not Serverless");
  assert.equal(candidate.isPublic, false, "checked-in template must default to private");
  assert.match(
    candidate.imageName,
    /^ghcr\.io\/sceneworks\/sceneworks-runpod:[a-z0-9][a-z0-9._-]{0,127}$/,
    "template image must use a RunPod-compatible GHCR image tag",
  );
  assert.ok(
    candidate.imageName === `${releaseRepository}:${validationTag}` ||
      /^ghcr\.io\/sceneworks\/sceneworks-runpod:[0-9]+\.[0-9]+\.[0-9]+$/.test(candidate.imageName),
    "template image must use the reviewed validation tag or an official semver tag",
  );
  assert.equal(candidate.volumeMountPath, "/workspace");
  assert.ok(
    Number.isInteger(candidate.containerDiskInGb) && candidate.containerDiskInGb >= 20,
    "template needs enough container disk for the CUDA image",
  );
  assert.ok(
    Number.isInteger(candidate.volumeInGb) && candidate.volumeInGb >= 20,
    "template needs a useful fallback volume size",
  );
  assert.deepEqual(candidate.ports, ["8010/http"], "template must expose only the HTTPS-proxied API");
  assert.deepEqual(candidate.dockerEntrypoint, [], "template must retain the image entrypoint");
  assert.deepEqual(candidate.dockerStartCmd, [], "template must retain the image start command");
  assert.deepEqual(candidate.env, {
    SCENEWORKS_ACCESS_TOKEN: accessSecret,
    SCENEWORKS_API_PORT: "8010",
    SCENEWORKS_VOLUME: "/workspace",
  });
  assert.ok(!templateText.includes("HF_TOKEN"), "optional HF token must not be forced into the template");
  assert.ok(
    !candidate.ports.some((port) => port.endsWith("/tcp")),
    "template must not expose a raw TCP port",
  );
}

validateTemplate(template);

// Mutation checks prove the validator rejects the security-sensitive non-defaults
// that this contract is intended to prevent.
for (const mutation of [
  { ...template, ports: ["8010/tcp"] },
  { ...template, env: { ...template.env, SCENEWORKS_ACCESS_TOKEN: "" } },
  { ...template, imageName: `${releaseRepository}:latest` },
]) {
  assert.throws(() => validateTemplate(mutation));
}

for (const contract of [
  releaseRepository,
  `${releaseRepository}:${validationTag}`,
  `${releaseRepository}@${validationDigest}`,
  "SCENEWORKS_ACCESS_TOKEN",
  "HF_TOKEN",
  "SCENEWORKS_VOLUME",
  "SCENEWORKS_API_PORT",
  "SCENEWORKS_CORS_ORIGINS",
  "https://<podid>-<port>.proxy.runpod.net",
  "https://<podid>-8010.proxy.runpod.net",
  "8010/http",
  "/workspace",
  "Model Manager",
  "RunPod's HTTP proxy provides HTTPS",
  "raw TCP",
  "manual-*",
  ":latest",
  "vX.Y.Z",
  "RUNPOD_SECRET_sceneworks_access_token",
]) {
  assert.ok(doc.includes(contract), `${docPath} is missing ${contract}`);
}

assert.ok(
  doc.includes("not needed for the default embedded UI"),
  "CORS guidance must preserve the same-origin default",
);
assert.ok(
  doc.includes("Do not assume") && doc.includes("before the first official release"),
  "documentation must not imply that :latest already exists",
);
assert.ok(
  workflow.includes('type=raw,value=latest,enable=${{ steps.publication.outputs.release == \'true\' }}'),
  "documentation relies on latest remaining release-only",
);
assert.ok(
  workflow.includes("^manual-[a-z0-9][a-z0-9._-]{0,119}$"),
  "documentation relies on manual validation tags remaining separate",
);
assert.ok(
  readme.includes("docs/deploy-runpod.md"),
  "README must link to the turnkey RunPod guide",
);

console.log("SceneWorks RunPod deployment template and documentation contract check passed.");
