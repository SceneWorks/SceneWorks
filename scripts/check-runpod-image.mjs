import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

const [dockerfile, entrypoint, supervisor, gpuSource, engineSource, readme, privileges] = await Promise.all([
  readFile("docker/rust.Dockerfile", "utf8"),
  readFile("docker/runpod-entrypoint.sh", "utf8"),
  readFile("crates/sceneworks-worker/src/supervisor.rs", "utf8"),
  readFile("crates/sceneworks-worker/src/gpu.rs", "utf8"),
  readFile("crates/sceneworks-worker/src/engines.rs", "utf8"),
  readFile("README.md", "utf8"),
  readFile("docker/runpod-privileges.sh", "utf8"),
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

const webBuilder = stage("web-builder");
assert.ok(
  webBuilder.body.includes("COPY docs/film-script-writing.md docs/film-editor.md ./docs/"),
  "web builder must include both documents bundled by the editor-guides plugin",
);

const ortBuilder = stage("ort-builder");
const candleBase = stage("rust-worker-candle-base");
const candleRuntime = stage("rust-worker-candle");
const runpod = stage("runpod");

assert.equal(
  candleBase.base,
  "nvidia/cuda:12.9.1-runtime-ubuntu24.04",
  "combined image must retain the validated CUDA 12.9.1 / Ubuntu 24.04 runtime",
);
assert.equal(candleRuntime.base, "rust-worker-candle-base", "standalone candle image must inherit the shared CUDA runtime");
assert.equal(runpod.base, "rust-worker-candle-base", "combined image must inherit the shared CUDA runtime");
assert.ok(candleBase.body.includes("install -d -m 0755 -o 1000 -g 1000 /home/sceneworks"));
assert.match(candleRuntime.body, /^USER 1000:1000$/m, "explicit primary GID must exclude the base user's supplementary groups");
for (const body of [candleBase.body, candleRuntime.body, runpod.body]) {
  assert.doesNotMatch(body, /\b(?:useradd|groupadd|userdel|groupdel|usermod|groupmod)\b/, "CUDA leaves must preserve existing base accounts");
}

for (const contract of [
  "ffmpeg",
  "COPY --from=candle-builder /out/sceneworks-rust-worker",
  "COPY --from=ort-builder ${ORT_PY_SITE} ${ORT_PY_SITE}",
]) {
  assert.ok(candleBase.body.includes(contract), `candle runtime base is missing ${contract}`);
}
assert.ok(
  ortBuilder.body.includes("onnxruntime-gpu==${ONNXRUNTIME_GPU_VERSION}"),
  "the ort staging stage must pin onnxruntime-gpu",
);

// The shipped worker image has no Python interpreter (epic 3482): pip is only the
// delivery mechanism for onnxruntime's shared libraries, so the venv is built in
// ort-builder and only its .so tree is copied forward. Re-adding python3/pip to a
// runtime stage would put a dead interpreter back in every published RunPod image —
// nothing in the container would ever invoke it. Matched against instructions only
// (comments discuss the packages by name) and as whole apt tokens, so the venv's
// hard-coded `python3.12` site-packages path is not a hit.
const runtimePython = /(?:^|\s)(?:python3(?:-venv)?|pip)(?=\s|$)/m;
function instructionsOnly(body) {
  return body
    .split(/\r?\n/)
    .filter((line) => !/^\s*#/.test(line))
    .join("\n");
}
assert.ok(
  !runtimePython.test(instructionsOnly(candleBase.body)),
  "candle runtime must stay Python-free — stage onnxruntime in ort-builder instead",
);
assert.ok(
  !runtimePython.test(instructionsOnly(runpod.body)),
  "combined RunPod runtime must stay Python-free",
);
for (const contract of [
  "COPY --from=embed-builder /out/sceneworks-rust-api",
  "COPY docker/runpod-entrypoint.sh",
  "SCENEWORKS_API_URL=http://127.0.0.1:8010",
  "SCENEWORKS_VOLUME=/workspace",
  "SCENEWORKS_JOBS_DB_PATH=/tmp/sceneworks/cache/jobs.db",
  "SCENEWORKS_CANDLE_REQUIRED=1",
  'VOLUME ["/workspace"]',
  "/api/v1/health",
  'ENTRYPOINT ["/usr/local/bin/sceneworks-runpod-entrypoint"]',
]) {
  assert.ok(runpod.body.includes(contract), `runpod stage is missing ${contract}`);
}

assert.ok(!/^USER\s/m.test(runpod.body), "provider initialization must retain root entrypoint access");
for (const contract of ["COPY docker/runpod-privileges.sh", "acl util-linux", "SCENEWORKS_SERVICE_UID=1000"]) {
  assert.ok(runpod.body.includes(contract), `RunPod privilege setup is missing ${contract}`);
}
for (const contract of ["exec setpriv", "--bounding-set=-all", "--inh-caps=-all", "--ambient-caps=-all", "--no-new-privs"]) {
  assert.ok(entrypoint.includes(contract), `RunPod privilege drop is missing ${contract}`);
}
assert.ok(privileges.includes('find -P') && privileges.includes('setfacl --no-mask --set-file=-'), "legacy managed files need ownership-preserving, physical ACL migration");
for (const contract of ['SCENEWORKS_PERMISSION_STRATEGY:-acl', 'acl|private-owned', 'initialize_private_owned_paths', 'preflight_private_owned_permissions', 'umask 077']) {
  assert.ok(privileges.includes(contract), `optional private-owned strategy is missing ${contract}`);
}
assert.ok(!/\bchown\s+-R|\bchmod\s+(?:777|a\+w)/.test(privileges), "RunPod must not recursively change ownership or grant world writes");

assert.ok(
  entrypoint.includes("SCENEWORKS_GPU_ID=auto"),
  "entrypoint must use worker auto-supervision to create GPU and utility children",
);
assert.ok(
  entrypoint.includes("NVIDIA_VISIBLE_DEVICES=all"),
  "entrypoint must normalize init-time RunPod visibility before auto GPU discovery",
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
  "--volume sceneworks-data:/workspace",
  "SCENEWORKS_VOLUME=/runpod-volume",
  "/tmp/sceneworks/cache/jobs.db",
]) {
  assert.ok(readme.includes(contract), `RunPod deployment docs are missing ${contract}`);
}

console.log("SceneWorks combined RunPod image contract check passed.");
