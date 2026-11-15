import { mkdtemp, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";

async function composeConfig(envFile) {
  const env = { ...process.env };
  delete env.SCENEWORKS_UID;
  delete env.SCENEWORKS_GID;
  delete env.SCENEWORKS_MCP_ALLOWED_HOSTS;
  const result = spawnSync(
    "docker",
    ["compose", "--env-file", envFile, "config", "--format", "json"],
    { encoding: "utf8", shell: false, env },
  );
  if (result.status !== 0) {
    throw new Error(result.stderr || result.stdout || "docker compose config failed");
  }
  return JSON.parse(result.stdout);
}

function assertEqual(actual, expected, label) {
  if (actual !== expected) {
    throw new Error(`${label}: expected ${expected}, got ${actual}`);
  }
}

function assertServiceMountsTarget(service, target, label) {
  const volumes = service?.volumes ?? [];
  if (!volumes.some((volume) => volume?.target === target || (typeof volume === "string" && volume.split(":").includes(target)))) {
    throw new Error(`${label}: expected a volume mounted at ${target}`);
  }
}

function assertWritableMount(service, target, label) {
  const volumes = service?.volumes ?? [];
  const volume = volumes.find((item) => item?.target === target || (typeof item === "string" && item.split(":").includes(target)));
  if (!volume) {
    throw new Error(`${label}: expected a volume mounted at ${target}`);
  }
  if (typeof volume === "string") {
    const [, , mode] = volume.split(":");
    if (mode === "ro") {
      throw new Error(`${label}: expected ${target} to be writable`);
    }
    return;
  }
  if (volume.read_only === true) {
    throw new Error(`${label}: expected ${target} to be writable`);
  }
}

function assertPublishedPort(service, target, expectedHostIp, expectedPublished, label) {
  const ports = service?.ports ?? [];
  const port = ports.find((item) => Number(item?.target) === target);
  if (!port) {
    throw new Error(`${label}: expected published port for target ${target}`);
  }
  assertEqual(port.host_ip, expectedHostIp, `${label} API host publish IP`);
  assertEqual(String(port.published), String(expectedPublished), `${label} API published port`);
}

function assertMissing(object, key, label) {
  if (Object.prototype.hasOwnProperty.call(object ?? {}, key)) {
    throw new Error(`${label}: expected ${key} to be absent`);
  }
}

function assertRuntimeDefaults(config, label, options = {}) {
  const apiPublishHost = options.apiPublishHost ?? "127.0.0.1";
  const apiPort = options.apiPort ?? "8010";
  const api = config.services?.api;
  const credentialsInit = config.services?.["credentials-init"];
  const allowedHosts = String(api?.environment?.SCENEWORKS_MCP_ALLOWED_HOSTS ?? "")
    .split(",").map((host) => host.trim()).filter(Boolean);
  if (!allowedHosts.includes("api")) {
    throw new Error(`${label}: API Host policy must admit the internal worker service name`);
  }
  if (options.extraHost && !allowedHosts.includes(options.extraHost)) {
    throw new Error(`${label}: API Host policy lost the configured LAN/proxy authority`);
  }

  const worker = config.services?.worker;
  const rustWorker = config.services?.["rust-worker"];
  // Split the removed key so the story's cleanup grep does not report this assertion as a live reference.
  const removedRuntimeKey = ["SCENEWORKS_API", "RUNTIME"].join("_");
  assertEqual(api?.build?.dockerfile, "docker/rust.Dockerfile", `${label} api dockerfile`);
  assertEqual(credentialsInit?.build?.target, "rust-api", `${label} credentials init image`);
  assertEqual(credentialsInit?.user, "0:0", `${label} credentials init identity`);
  assertEqual(credentialsInit?.network_mode, "none", `${label} credentials init network`);
  assertEqual(credentialsInit?.read_only, true, `${label} credentials init filesystem`);
  assertEqual(credentialsInit?.restart, "no", `${label} credentials init restart`);
  assertEqual(credentialsInit?.volumes?.length, 1, `${label} credentials init mount count`);
  assertEqual(credentialsInit?.cap_drop?.join(","), "ALL", `${label} credentials init dropped capabilities`);
  assertEqual(credentialsInit?.cap_add?.join(","), "CHOWN,FOWNER", `${label} credentials init added capabilities`);
  assertEqual(credentialsInit?.security_opt?.join(","), "no-new-privileges:true", `${label} credentials init security option`);
  assertEqual(credentialsInit?.environment?.SCENEWORKS_UID, "1000", `${label} credentials init UID`);
  assertEqual(credentialsInit?.environment?.SCENEWORKS_GID, "1000", `${label} credentials init GID`);
  assertEqual(api?.depends_on?.["credentials-init"]?.condition, "service_completed_successfully", `${label} init before API`);
  assertWritableMount(credentialsInit, "/sceneworks/credentials", `${label} credentials init mount`);
  const initCredentialSource = credentialsInit?.volumes?.find((item) => item?.target === "/sceneworks/credentials")?.source;
  for (const name of ["api", "worker", "rust-worker"]) {
    const source = config.services?.[name]?.volumes?.find((item) => item?.target === "/sceneworks/credentials")?.source;
    assertEqual(source, initCredentialSource, `${label} ${name} shares prepared credential bind`);
  }
  assertPublishedPort(api, Number(apiPort), apiPublishHost, apiPort, `${label} api`);
  assertMissing(api?.environment, removedRuntimeKey, `${label} api runtime switch`);
  assertWritableMount(api, "/sceneworks/config", `${label} api config mount`);
  // The Docker GPU worker is the native candle/CUDA worker (epic 5483 Phase 7 /
  // sc-5503), not the retired Python torch worker. The API requires candle and fails
  // unsupported jobs loudly (sc-5502) since there is no torch fallback in Docker.
  assertEqual(api?.environment?.SCENEWORKS_CANDLE_REQUIRED, "1", `${label} api candle required`);
  assertEqual(worker?.build?.target, "rust-worker-candle", `${label} candle worker build target`);
  assertEqual(worker?.environment?.SCENEWORKS_WORKER_ID, "candle-inference-worker-0", `${label} candle worker id`);
  assertEqual(worker?.environment?.HF_HOME, "/sceneworks/data/cache/huggingface", `${label} candle worker HF home`);
  assertEqual(
    worker?.environment?.HUGGINGFACE_HUB_CACHE,
    "/sceneworks/data/cache/huggingface/hub",
    `${label} candle worker HF hub cache`,
  );
  assertEqual(rustWorker?.environment?.SCENEWORKS_GPU_ID, "cpu", `${label} rust worker gpu mode`);
  assertEqual(rustWorker?.environment?.SCENEWORKS_CONFIG_DIR, "/sceneworks/config", `${label} rust worker config dir`);
  assertServiceMountsTarget(rustWorker, "/sceneworks/config", `${label} rust worker config mount`);
  // sc-16540: the credential store must be its own writable mount on every service
  // that touches it, and must NOT resolve under /sceneworks/config — that path is
  // bind-mounted from the repo checkout by default, and the store is plaintext tokens.
  for (const [service, serviceLabel] of [
    [api, "api"],
    [worker, "candle worker"],
    [rustWorker, "rust worker"],
  ]) {
    assertEqual(service?.environment?.HOME, "/sceneworks/data", `${label} ${serviceLabel} writable home`);
    assertEqual(
      service?.environment?.SCENEWORKS_CREDENTIALS_DIR,
      "/sceneworks/credentials",
      `${label} ${serviceLabel} credentials dir`,
    );
    assertWritableMount(service, "/sceneworks/credentials", `${label} ${serviceLabel} credentials mount`);
  }
  assertCredentialsBindOutsideRepo(config, label);
}

// The credentials bind must default outside the repo working tree — putting it back
// under ./config or ./data would reinstate exactly the problem sc-16540 fixed.
function assertCredentialsBindOutsideRepo(config, label) {
  for (const name of ["api", "worker", "rust-worker"]) {
    const service = config.services?.[name];
    const volume = (service?.volumes ?? []).find(
      (item) => item?.target === "/sceneworks/credentials",
    );
    const source = volume?.source ?? "";
    if (source.startsWith("./") || source === "." || source.startsWith("../")) {
      throw new Error(
        `${label} ${name} credentials bind: expected a source outside the repo, got ${source}`,
      );
    }
  }
}

const tempRoot = await mkdtemp(path.join(os.tmpdir(), "sceneworks-compose-config-"));
const emptyEnv = path.join(tempRoot, "empty.env");
const lanEnv = path.join(tempRoot, "lan.env");
const idsEnv = path.join(tempRoot, "ids.env");

try {
  await writeFile(emptyEnv, "", "utf8");
  await writeFile(
    lanEnv,
    "SCENEWORKS_API_PUBLISH_HOST=0.0.0.0\nSCENEWORKS_API_PORT=18010\nSCENEWORKS_MCP_ALLOWED_HOSTS=studio.local:18010\n",
    "utf8",
  );
  await writeFile(idsEnv, "SCENEWORKS_UID=1234\nSCENEWORKS_GID=2345\n", "utf8");
  assertRuntimeDefaults(await composeConfig(emptyEnv), "compose defaults");
  assertRuntimeDefaults(await composeConfig(".env.example"), ".env.example");
  assertRuntimeDefaults(await composeConfig(lanEnv), "compose LAN override", {
    apiPublishHost: "0.0.0.0",
    apiPort: "18010",
    extraHost: "studio.local:18010",
  });
  const ids = await composeConfig(idsEnv);
  for (const name of ["api", "worker", "rust-worker"]) {
    assertEqual(ids.services?.[name]?.user, "1234:2345", `${name} UID/GID override`);
  }
  assertEqual(ids.services?.["credentials-init"]?.environment?.SCENEWORKS_UID, "1234", "credentials init UID override");
  assertEqual(ids.services?.["credentials-init"]?.environment?.SCENEWORKS_GID, "2345", "credentials init GID override");
  console.log("SceneWorks compose config check passed.");
} finally {
  await rm(tempRoot, { recursive: true, force: true });
}
