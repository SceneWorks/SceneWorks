import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

const [dockerfile, entrypoint, supervisor, gpuSource, engineSource, readme] = await Promise.all([
  readFile("docker/rust.Dockerfile", "utf8"),
  readFile("docker/runpod-entrypoint.sh", "utf8"),
  readFile("crates/sceneworks-worker/src/supervisor.rs", "utf8"),
  readFile("crates/sceneworks-worker/src/gpu.rs", "utf8"),
  readFile("crates/sceneworks-worker/src/engines.rs", "utf8"),
  readFile("README.md", "utf8"),
]);

function stage(name) {
  const header = new RegExp(`^FROM ([^\\r\\n]+) AS ${name}\\r?$`, "m").exec(dockerfile);
  assert.ok(header, `missing Docker stage ${name}`);
  const start = header.index + header[0].length;
  const next = dockerfile.indexOf("\nFROM ", start);
  return {
    base: header[1],
    body: dockerfile.slice(start, next === -1 ? undefined : next),
  };
}

const candleRuntime = stage("rust-worker-candle");
const runpod = stage("runpod");

assert.equal(
  candleRuntime.base,
  "nvidia/cuda:12.9.1-runtime-ubuntu24.04",
  "combined image must retain the validated CUDA 12.9.1 / Ubuntu 24.04 runtime",
);
assert.equal(runpod.base, "rust-worker-candle", "combined image must inherit candle runtime");

for (const contract of [
  "ffmpeg",
  "COPY --from=candle-builder /out/sceneworks-rust-worker",
  "onnxruntime-gpu==${ONNXRUNTIME_GPU_VERSION}",
]) {
  assert.ok(candleRuntime.body.includes(contract), `candle runtime is missing ${contract}`);
}
for (const contract of [
  "COPY --from=embed-builder /out/sceneworks-rust-api",
  "COPY docker/runpod-entrypoint.sh",
  "SCENEWORKS_API_URL=http://127.0.0.1:8010",
  "SCENEWORKS_CANDLE_REQUIRED=1",
  'VOLUME ["/sceneworks"]',
  "/api/v1/health",
  'ENTRYPOINT ["/usr/local/bin/sceneworks-runpod-entrypoint"]',
]) {
  assert.ok(runpod.body.includes(contract), `runpod stage is missing ${contract}`);
}

assert.ok(
  entrypoint.includes("SCENEWORKS_GPU_ID=auto"),
  "entrypoint must use worker auto-supervision to create GPU and utility children",
);
assert.ok(
  entrypoint.includes('SCENEWORKS_API_URL="${api_url}"'),
  "worker must connect to the API over loopback",
);
assert.ok(
  entrypoint.includes('SCENEWORKS_ACCESS_TOKEN="${SCENEWORKS_ACCESS_TOKEN:-}"'),
  "entrypoint must explicitly propagate the access token to the worker supervisor",
);
assert.ok(
  entrypoint.includes("unset SCENEWORKS_ALLOW_OPEN_BIND"),
  "entrypoint must remove the legacy unauthenticated-bind override",
);
assert.ok(
  entrypoint.includes("is_loopback_host") && entrypoint.includes("access_token_trimmed"),
  "entrypoint must reject a network bind before startup when the token is blank",
);
assert.ok(
  entrypoint.includes("shutdown_children"),
  "entrypoint must own coordinated child shutdown",
);
assert.ok(
  supervisor.includes('command.env_remove("SCENEWORKS_ACCESS_TOKEN")') &&
    supervisor.includes('"SCENEWORKS_ACCESS_TOKEN".to_owned()'),
  "worker supervisor must explicitly propagate the parsed token to GPU and utility children",
);
assert.ok(
  !dockerfile.includes("SCENEWORKS_ALLOW_OPEN_BIND"),
  "combined image Dockerfile must never configure the unauthenticated-bind override",
);

for (const capability of [
  "Cap::ImageGenerate",
  "Cap::VideoGenerate",
]) {
  assert.ok(engineSource.includes(capability), `worker engine registry is missing ${capability}`);
}
for (const capability of [
  "WorkerCapability::ModelDownload",
  "WorkerCapability::ModelImport",
  "WorkerCapability::LoraImport",
  "WorkerCapability::TimelineExport",
]) {
  assert.ok(gpuSource.includes(capability), `worker source is missing ${capability}`);
}

assert.ok(
  dockerfile.includes("native in-process") && dockerfile.includes("do not install the retired Hugging Face CLI"),
  "combined image must document the native Model Manager download path",
);
assert.ok(
  !/pip install[^\n]*huggingface[_-]hub/i.test(dockerfile),
  "combined image must not reintroduce the retired Hugging Face CLI",
);
for (const contract of [
  "--target runpod",
  "--gpus all",
  "--env SCENEWORKS_ACCESS_TOKEN=",
  "--volume sceneworks-data:/sceneworks",
]) {
  assert.ok(readme.includes(contract), `RunPod deployment docs are missing ${contract}`);
}

console.log("SceneWorks combined RunPod image contract check passed.");
