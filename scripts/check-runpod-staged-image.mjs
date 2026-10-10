import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const INSTALLED_SCRIPTS = [
  ["/usr/local/bin/sceneworks-runpod-entrypoint", "docker/runpod-entrypoint.sh"],
  ["/usr/local/bin/runpod-privileges.sh", "docker/runpod-privileges.sh"],
];

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

export function validateStagedImage({ image, revision, inspectJson, installedHashes, sourceHashes }) {
  assert.match(image, /@sha256:[a-f0-9]{64}$/, "staged image must be addressed by immutable digest");
  assert.match(revision, /^[a-f0-9]{40}$/, "expected source revision must be a full Git SHA");

  const imageConfig = typeof inspectJson === "string" ? JSON.parse(inspectJson) : inspectJson;
  assert.equal(imageConfig.Os, "linux", "staged image must be Linux");
  assert.equal(imageConfig.Architecture, "amd64", "staged image must be amd64");
  assert.deepEqual(
    imageConfig.Config?.Entrypoint,
    ["/usr/local/bin/sceneworks-runpod-entrypoint"],
    "staged image must retain the production RunPod entrypoint",
  );
  assert.equal(imageConfig.Config?.User ?? "", "", "staged image must retain root-only initialization entrypoint access");
  assert.equal(
    imageConfig.Config?.Labels?.["org.opencontainers.image.revision"],
    revision,
    "staged image OCI revision label must match the checked-out source",
  );

  for (const [installedPath, sourcePath] of INSTALLED_SCRIPTS) {
    const actual = installedHashes[installedPath];
    const expected = sourceHashes[sourcePath];
    assert.match(actual ?? "", /^[a-f0-9]{64}$/, `missing SHA-256 for installed ${installedPath}`);
    assert.equal(actual, expected, `installed ${installedPath} must match ${sourcePath} from source`);
  }
  return true;
}

function parseArgs(argv) {
  const values = {};
  for (let index = 2; index < argv.length; index += 1) {
    const key = argv[index];
    assert.ok(["--image", "--revision"].includes(key), `unknown argument ${key}`);
    assert.ok(argv[index + 1], `missing value for ${key}`);
    values[key] = argv[index + 1];
    index += 1;
  }
  assert.ok(values["--image"], "--image is required");
  assert.ok(values["--revision"], "--revision is required");
  return values;
}

function main() {
  const { ["--image"]: image, ["--revision"]: revision } = parseArgs(process.argv);
  const inspected = execFileSync("docker", ["image", "inspect", "--format", "{{json .}}", image], {
    encoding: "utf8",
  });
  const imageConfig = JSON.parse(inspected);
  const installedOutput = execFileSync(
    "docker",
    [
      "run",
      "--rm",
      "--pull=never",
      "--entrypoint",
      "sha256sum",
      image,
      ...INSTALLED_SCRIPTS.map(([installedPath]) => installedPath),
    ],
    { encoding: "utf8" },
  );
  const installedHashes = Object.fromEntries(
    installedOutput
      .trim()
      .split(/\r?\n/)
      .map((line) => {
        const match = /^([a-f0-9]{64})\s+(.+)$/.exec(line);
        assert.ok(match, `unexpected sha256sum output: ${line}`);
        return [match[2], match[1]];
      }),
  );
  const sourceHashes = Object.fromEntries(
    INSTALLED_SCRIPTS.map(([, sourcePath]) => [sourcePath, sha256(readFileSync(path.join(ROOT, sourcePath)))]),
  );
  validateStagedImage({ image, revision, inspectJson: imageConfig, installedHashes, sourceHashes });
  console.log(`Verified staged RunPod image ${image} revision ${revision}; linux/amd64; root entrypoint identity retained.`);
  for (const [installedPath, sourcePath] of INSTALLED_SCRIPTS) {
    console.log(`SHA-256 ${installedPath}=${installedHashes[installedPath]} matches ${sourcePath}`);
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main();
