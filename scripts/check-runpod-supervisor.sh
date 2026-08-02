#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
entrypoint="${repo_root}/docker/runpod-entrypoint.sh"
temp_root="$(mktemp -d)"
trap 'rm -rf "${temp_root}"' EXIT

fail() {
  printf 'runpod supervisor self-test failed: %s\n' "$*" >&2
  exit 1
}

wait_for_event() {
  local event="$1"
  local file="$2"
  local attempt
  for ((attempt = 0; attempt < 100; attempt += 1)); do
    if [[ -f "${file}" ]] && grep -Fqx "${event}" "${file}"; then
      return 0
    fi
    sleep 0.05
  done
  fail "timed out waiting for '${event}' in ${file}"
}

cat >"${temp_root}/fake-api" <<'EOF'
#!/usr/bin/env bash
set -u
if [[ -v SCENEWORKS_ALLOW_OPEN_BIND ]]; then
  printf 'api:override:present\n' >>"${SCENEWORKS_TEST_EVENTS}"
  exit 91
fi
if [[ -n "${SCENEWORKS_TEST_EXPECTED_TOKEN:-}" ]]; then
  if [[ "${SCENEWORKS_ACCESS_TOKEN:-}" != "${SCENEWORKS_TEST_EXPECTED_TOKEN}" ]]; then
    printf 'api:token:mismatch\n' >>"${SCENEWORKS_TEST_EVENTS}"
    exit 92
  fi
  printf 'api:token:present\n' >>"${SCENEWORKS_TEST_EVENTS}"
elif [[ -n "${SCENEWORKS_ACCESS_TOKEN:-}" ]]; then
  printf 'api:token:unexpected\n' >>"${SCENEWORKS_TEST_EVENTS}"
  exit 93
else
  printf 'api:token:absent\n' >>"${SCENEWORKS_TEST_EVENTS}"
fi
if [[ "${SCENEWORKS_TEST_RECORD_PATHS:-0}" == "1" ]]; then
  printf 'api:paths:%s|%s|%s|%s|%s|%s|%s\n' \
    "${SCENEWORKS_VOLUME}" \
    "${SCENEWORKS_DATA_DIR}" \
    "${SCENEWORKS_CONFIG_DIR}" \
    "${SCENEWORKS_CREDENTIALS_DIR}" \
    "${SCENEWORKS_JOBS_DB_PATH}" \
    "${HF_HOME}" \
    "${HF_HUB_CACHE:-${HUGGINGFACE_HUB_CACHE:-}}" \
    >>"${SCENEWORKS_TEST_EVENTS}"
  printf 'api:hf-env:%s|%s\n' \
    "${HF_HUB_CACHE-<unset>}" \
    "${HUGGINGFACE_HUB_CACHE-<unset>}" \
    >>"${SCENEWORKS_TEST_EVENTS}"
fi
printf 'api:start\n' >>"${SCENEWORKS_TEST_EVENTS}"
trap 'printf "api:term\n" >>"${SCENEWORKS_TEST_EVENTS}"; exit 0' TERM INT HUP
while true; do sleep 0.1; done
EOF

cat >"${temp_root}/fake-worker" <<'EOF'
#!/usr/bin/env bash
set -u
if [[ -v SCENEWORKS_ALLOW_OPEN_BIND ]]; then
  printf 'worker:override:present\n' >>"${SCENEWORKS_TEST_EVENTS}"
  exit 91
fi
if [[ -n "${SCENEWORKS_TEST_EXPECTED_TOKEN:-}" ]]; then
  if [[ "${SCENEWORKS_ACCESS_TOKEN:-}" != "${SCENEWORKS_TEST_EXPECTED_TOKEN}" ]]; then
    printf 'worker:token:mismatch\n' >>"${SCENEWORKS_TEST_EVENTS}"
    exit 92
  fi
  printf 'worker:token:present\n' >>"${SCENEWORKS_TEST_EVENTS}"
elif [[ -n "${SCENEWORKS_ACCESS_TOKEN:-}" ]]; then
  printf 'worker:token:unexpected\n' >>"${SCENEWORKS_TEST_EVENTS}"
  exit 93
else
  printf 'worker:token:absent\n' >>"${SCENEWORKS_TEST_EVENTS}"
fi
printf 'worker:start:%s:%s\n' "${SCENEWORKS_API_URL}" "${SCENEWORKS_GPU_ID}" >>"${SCENEWORKS_TEST_EVENTS}"
printf 'worker:nvidia-visible:%s\n' "${NVIDIA_VISIBLE_DEVICES:-<unset>}" >>"${SCENEWORKS_TEST_EVENTS}"
trap 'printf "worker:term\n" >>"${SCENEWORKS_TEST_EVENTS}"; exit 0' TERM INT HUP
if [[ "${SCENEWORKS_TEST_WORKER_EXIT_CODE:-}" =~ ^[0-9]+$ ]]; then
  sleep 0.1
  exit "${SCENEWORKS_TEST_WORKER_EXIT_CODE}"
fi
while true; do sleep 0.1; done
EOF

cat >"${temp_root}/fake-curl" <<'EOF'
#!/usr/bin/env bash
set -u
count=0
if [[ -f "${SCENEWORKS_TEST_CURL_COUNT}" ]]; then
  count="$(cat "${SCENEWORKS_TEST_CURL_COUNT}")"
fi
count=$((count + 1))
printf '%s' "${count}" >"${SCENEWORKS_TEST_CURL_COUNT}"
[[ "${SCENEWORKS_TEST_CURL_ALWAYS_FAIL:-0}" != "1" && "${count}" -ge 2 ]]
EOF
chmod +x "${temp_root}/fake-api" "${temp_root}/fake-worker" "${temp_root}/fake-curl"

assert_public_bind_rejected() {
  local label="$1"
  local host="$2"
  local token_mode="$3"
  local override="${4:-}"
  local run_dir="${temp_root}/${label}"
  local status
  mkdir -p "${run_dir}"

  set +e
  if [[ "${token_mode}" == "unset" ]]; then
    env -u SCENEWORKS_ACCESS_TOKEN \
      SCENEWORKS_API_HOST="${host}" \
      SCENEWORKS_ALLOW_OPEN_BIND="${override}" \
      SCENEWORKS_TEST_EVENTS="${run_dir}/events" \
      SCENEWORKS_API_BIN="${temp_root}/fake-api" \
      SCENEWORKS_WORKER_BIN="${temp_root}/fake-worker" \
      SCENEWORKS_DATA_DIR="${run_dir}/data" \
      SCENEWORKS_CONFIG_DIR="${run_dir}/config" \
      SCENEWORKS_CREDENTIALS_DIR="${run_dir}/credentials" \
      HF_HOME="${run_dir}/hf" \
      bash "${entrypoint}" >"${run_dir}/stdout" 2>"${run_dir}/stderr"
  else
    SCENEWORKS_API_HOST="${host}" \
    SCENEWORKS_ACCESS_TOKEN="${token_mode}" \
    SCENEWORKS_ALLOW_OPEN_BIND="${override}" \
    SCENEWORKS_TEST_EVENTS="${run_dir}/events" \
    SCENEWORKS_API_BIN="${temp_root}/fake-api" \
    SCENEWORKS_WORKER_BIN="${temp_root}/fake-worker" \
    SCENEWORKS_DATA_DIR="${run_dir}/data" \
    SCENEWORKS_CONFIG_DIR="${run_dir}/config" \
    SCENEWORKS_CREDENTIALS_DIR="${run_dir}/credentials" \
    HF_HOME="${run_dir}/hf" \
    bash "${entrypoint}" >"${run_dir}/stdout" 2>"${run_dir}/stderr"
  fi
  status=$?
  set -e

  [[ "${status}" -eq 1 ]] || fail "${label} status was ${status}, expected 1"
  [[ ! -e "${run_dir}/events" ]] || fail "${label} started a child before auth preflight"
  [[ ! -d "${run_dir}/data" ]] || fail "${label} created data before auth preflight"
  grep -Fq "SCENEWORKS_ACCESS_TOKEN is required" "${run_dir}/stderr" ||
    fail "${label} did not return an actionable token error"
}

# Fail closed before any child/process/filesystem side effect for every public
# wildcard spelling supported by the API. A whitespace-only value is missing,
# and the legacy override cannot bypass the combined-image contract.
assert_public_bind_rejected "public-v4-missing" "0.0.0.0" "unset"
assert_public_bind_rejected "public-v4-blank" "0.0.0.0" $' \t '
assert_public_bind_rejected "public-v4-override" "0.0.0.0" "unset" "1"
assert_public_bind_rejected "public-v6-bare" "::" "unset"
assert_public_bind_rejected "public-v6-bracketed" "[::]" "unset"

assert_storage_rejected() {
  local label="$1"
  shift
  local run_dir="${temp_root}/${label}"
  local status
  mkdir -p "${run_dir}"

  set +e
  env -u SCENEWORKS_DATA_DIR \
    -u SCENEWORKS_CONFIG_DIR \
    -u SCENEWORKS_JOBS_DB_PATH \
    -u HF_HOME \
    -u HF_HUB_CACHE \
    -u HUGGINGFACE_HUB_CACHE \
    SCENEWORKS_API_HOST=127.0.0.1 \
    SCENEWORKS_ACCESS_TOKEN= \
    SCENEWORKS_TEST_EVENTS="${run_dir}/events" \
    SCENEWORKS_API_BIN="${temp_root}/fake-api" \
    SCENEWORKS_WORKER_BIN="${temp_root}/fake-worker" \
    "$@" \
    bash "${entrypoint}" >"${run_dir}/stdout" 2>"${run_dir}/stderr"
  status=$?
  set -e

  [[ "${status}" -eq 1 ]] || fail "${label} status was ${status}, expected 1"
  [[ ! -e "${run_dir}/events" ]] || fail "${label} started a child with invalid storage paths"
  grep -Fq "must " "${run_dir}/stderr" || fail "${label} lacked an actionable path error"
}

# Empty settings use their safe derived defaults, but relative container paths
# and a jobs database whose parent is filesystem root are rejected before any
# process starts.
assert_storage_rejected "relative-volume" SCENEWORKS_VOLUME=relative
assert_storage_rejected "slash-only-root-volume" SCENEWORKS_VOLUME=//
assert_storage_rejected "dot-segment-volume" SCENEWORKS_VOLUME=/workspace/..
assert_storage_rejected \
  "relative-data-override" \
  SCENEWORKS_VOLUME="${temp_root}/valid-base" \
  SCENEWORKS_DATA_DIR=relative-data
assert_storage_rejected \
  "root-alias-data-override" \
  SCENEWORKS_VOLUME="${temp_root}/valid-base" \
  SCENEWORKS_DATA_DIR=/tmp/..
assert_storage_rejected \
  "root-jobs-parent" \
  SCENEWORKS_VOLUME="${temp_root}/valid-base" \
  SCENEWORKS_JOBS_DB_PATH=/jobs.db
assert_storage_rejected \
  "dot-segment-jobs-parent" \
  SCENEWORKS_VOLUME="${temp_root}/valid-base" \
  SCENEWORKS_JOBS_DB_PATH=/tmp/../jobs.db

# A missing token must also fail before the new single-volume defaults create
# anything. Do not let the security preflight regress behind storage setup.
run_dir="${temp_root}/public-volume-no-side-effects"
mkdir -p "${run_dir}"
set +e
env -u SCENEWORKS_ACCESS_TOKEN \
  -u SCENEWORKS_DATA_DIR \
  -u SCENEWORKS_CONFIG_DIR \
  -u SCENEWORKS_JOBS_DB_PATH \
  -u HF_HOME \
  -u HF_HUB_CACHE \
  -u HUGGINGFACE_HUB_CACHE \
  SCENEWORKS_API_HOST=0.0.0.0 \
  SCENEWORKS_VOLUME="${run_dir}/volume" \
  SCENEWORKS_TEST_EVENTS="${run_dir}/events" \
  SCENEWORKS_API_BIN="${temp_root}/fake-api" \
  SCENEWORKS_WORKER_BIN="${temp_root}/fake-worker" \
  bash "${entrypoint}" >"${run_dir}/stdout" 2>"${run_dir}/stderr"
status=$?
set -e
[[ "${status}" -eq 1 ]] || fail "public volume preflight status was ${status}, expected 1"
[[ ! -e "${run_dir}/events" ]] || fail "public volume preflight started a child"
[[ ! -e "${run_dir}/volume" ]] || fail "public volume preflight touched the volume"

# One base must derive every durable default, while the jobs database stays on
# the explicitly selected ephemeral path. The persisted marker proves startup
# never replaces or clears a reused mount.
run_dir="${temp_root}/volume-layout"
mkdir -p "${run_dir}/volume"
printf 'keep-me\n' >"${run_dir}/volume/existing-model-marker"
events="${run_dir}/events"
env -u SCENEWORKS_DATA_DIR \
  -u SCENEWORKS_CONFIG_DIR \
  -u HF_HOME \
  -u HF_HUB_CACHE \
  -u HUGGINGFACE_HUB_CACHE \
  SCENEWORKS_DATA_DIR= \
  SCENEWORKS_CONFIG_DIR= \
  HF_HOME= \
  HF_HUB_CACHE= \
  HUGGINGFACE_HUB_CACHE= \
  SCENEWORKS_TEST_EVENTS="${events}" \
  SCENEWORKS_TEST_RECORD_PATHS=1 \
  SCENEWORKS_TEST_CURL_COUNT="${run_dir}/curl-count" \
  SCENEWORKS_API_HOST=127.0.0.1 \
  SCENEWORKS_ACCESS_TOKEN= \
  SCENEWORKS_VOLUME="${run_dir}/volume" \
  SCENEWORKS_JOBS_DB_PATH="${run_dir}/ephemeral/jobs.db" \
  SCENEWORKS_API_BIN="${temp_root}/fake-api" \
  SCENEWORKS_WORKER_BIN="${temp_root}/fake-worker" \
  SCENEWORKS_CURL_BIN="${temp_root}/fake-curl" \
  SCENEWORKS_READINESS_MAX_ATTEMPTS=5 \
  SCENEWORKS_READINESS_INTERVAL_SECONDS=0.05 \
  SCENEWORKS_SHUTDOWN_GRACE_ATTEMPTS=6 \
  SCENEWORKS_SHUTDOWN_GRACE_INTERVAL_SECONDS=0.05 \
  bash "${entrypoint}" >"${run_dir}/stdout" 2>"${run_dir}/stderr" &
supervisor_pid=$!
wait_for_event \
  "api:paths:${run_dir}/volume|${run_dir}/volume/data|${run_dir}/volume/config|${run_dir}/volume/credentials|${run_dir}/ephemeral/jobs.db|${run_dir}/volume/cache/huggingface|${run_dir}/volume/cache/huggingface/hub" \
  "${events}"
wait_for_event \
  "api:hf-env:${run_dir}/volume/cache/huggingface/hub|${run_dir}/volume/cache/huggingface/hub" \
  "${events}"
wait_for_event "worker:start:http://127.0.0.1:8010:auto" "${events}"
kill -TERM "${supervisor_pid}"
wait "${supervisor_pid}"
[[ -f "${run_dir}/volume/existing-model-marker" ]] ||
  fail "startup removed existing persistent-volume data"
[[ -d "${run_dir}/volume/data" ]] || fail "derived data directory was not created"
[[ -d "${run_dir}/volume/config" ]] || fail "derived config directory was not created"
[[ -d "${run_dir}/volume/cache/huggingface/hub" ]] ||
  fail "derived Hugging Face cache was not created"
[[ ! -e "${run_dir}/volume/tmp" ]] || fail "ephemeral jobs data landed on persistent volume"
if find "${run_dir}" -name '.sceneworks-write-test.*' -print -quit | grep -q .; then
  fail "startup left a write-probe artifact behind"
fi

# Explicit paths beat the base independently. Also exercise the legacy HF cache
# override without allowing a generated HF_HUB_CACHE to shadow it.
run_dir="${temp_root}/volume-overrides"
mkdir -p "${run_dir}"
events="${run_dir}/events"
env -u HF_HUB_CACHE \
  SCENEWORKS_TEST_EVENTS="${events}" \
  SCENEWORKS_TEST_RECORD_PATHS=1 \
  SCENEWORKS_TEST_CURL_COUNT="${run_dir}/curl-count" \
  SCENEWORKS_API_HOST=127.0.0.1 \
  SCENEWORKS_ACCESS_TOKEN= \
  SCENEWORKS_VOLUME= \
  SCENEWORKS_DATA_DIR="${run_dir}/custom-data" \
  SCENEWORKS_CONFIG_DIR="${run_dir}/custom-config" \
  SCENEWORKS_CREDENTIALS_DIR="${run_dir}/custom-credentials" \
  SCENEWORKS_JOBS_DB_PATH="${run_dir}/custom-runtime/jobs.db" \
  HF_HOME="${run_dir}/custom-hf-home" \
  HUGGINGFACE_HUB_CACHE="${run_dir}/legacy-hub-cache" \
  SCENEWORKS_API_BIN="${temp_root}/fake-api" \
  SCENEWORKS_WORKER_BIN="${temp_root}/fake-worker" \
  SCENEWORKS_CURL_BIN="${temp_root}/fake-curl" \
  SCENEWORKS_READINESS_MAX_ATTEMPTS=5 \
  SCENEWORKS_READINESS_INTERVAL_SECONDS=0.05 \
  SCENEWORKS_SHUTDOWN_GRACE_ATTEMPTS=6 \
  SCENEWORKS_SHUTDOWN_GRACE_INTERVAL_SECONDS=0.05 \
  bash "${entrypoint}" >"${run_dir}/stdout" 2>"${run_dir}/stderr" &
supervisor_pid=$!
wait_for_event \
  "api:paths:/workspace|${run_dir}/custom-data|${run_dir}/custom-config|${run_dir}/custom-credentials|${run_dir}/custom-runtime/jobs.db|${run_dir}/custom-hf-home|${run_dir}/legacy-hub-cache" \
  "${events}"
wait_for_event "api:hf-env:<unset>|${run_dir}/legacy-hub-cache" "${events}"
wait_for_event "worker:start:http://127.0.0.1:8010:auto" "${events}"
kill -TERM "${supervisor_pid}"
wait "${supervisor_pid}"

# When both hub-cache spellings are explicit, HF_HUB_CACHE remains the effective
# high-priority value and neither operator setting is rewritten.
run_dir="${temp_root}/hf-both-overrides"
mkdir -p "${run_dir}"
events="${run_dir}/events"
SCENEWORKS_TEST_EVENTS="${events}" \
SCENEWORKS_TEST_RECORD_PATHS=1 \
SCENEWORKS_TEST_CURL_COUNT="${run_dir}/curl-count" \
SCENEWORKS_API_HOST=127.0.0.1 \
SCENEWORKS_ACCESS_TOKEN= \
SCENEWORKS_VOLUME="${run_dir}/volume" \
SCENEWORKS_DATA_DIR="${run_dir}/data" \
SCENEWORKS_CONFIG_DIR="${run_dir}/config" \
SCENEWORKS_JOBS_DB_PATH="${run_dir}/runtime/jobs.db" \
HF_HOME="${run_dir}/hf-home" \
HF_HUB_CACHE="${run_dir}/priority-hub-cache" \
HUGGINGFACE_HUB_CACHE="${run_dir}/legacy-hub-cache" \
SCENEWORKS_API_BIN="${temp_root}/fake-api" \
SCENEWORKS_WORKER_BIN="${temp_root}/fake-worker" \
SCENEWORKS_CURL_BIN="${temp_root}/fake-curl" \
SCENEWORKS_READINESS_MAX_ATTEMPTS=5 \
SCENEWORKS_READINESS_INTERVAL_SECONDS=0.05 \
SCENEWORKS_SHUTDOWN_GRACE_ATTEMPTS=6 \
SCENEWORKS_SHUTDOWN_GRACE_INTERVAL_SECONDS=0.05 \
bash "${entrypoint}" >"${run_dir}/stdout" 2>"${run_dir}/stderr" &
supervisor_pid=$!
wait_for_event \
  "api:paths:${run_dir}/volume|${run_dir}/data|${run_dir}/config|${run_dir}/volume/credentials|${run_dir}/runtime/jobs.db|${run_dir}/hf-home|${run_dir}/priority-hub-cache" \
  "${events}"
wait_for_event \
  "api:hf-env:${run_dir}/priority-hub-cache|${run_dir}/legacy-hub-cache" \
  "${events}"
wait_for_event "worker:start:http://127.0.0.1:8010:auto" "${events}"
kill -TERM "${supervisor_pid}"
wait "${supervisor_pid}"

run_dir="${temp_root}/signal"
mkdir -p "${run_dir}"
events="${run_dir}/events"
test_access_token="runpod-self-test-$PPID-$RANDOM"
SCENEWORKS_TEST_EVENTS="${events}" \
SCENEWORKS_TEST_CURL_COUNT="${run_dir}/curl-count" \
SCENEWORKS_TEST_EXPECTED_TOKEN="${test_access_token}" \
NVIDIA_VISIBLE_DEVICES=void \
SCENEWORKS_API_HOST=0.0.0.0 \
SCENEWORKS_ACCESS_TOKEN="${test_access_token}" \
SCENEWORKS_ALLOW_OPEN_BIND=1 \
SCENEWORKS_API_BIN="${temp_root}/fake-api" \
SCENEWORKS_WORKER_BIN="${temp_root}/fake-worker" \
SCENEWORKS_CURL_BIN="${temp_root}/fake-curl" \
SCENEWORKS_DATA_DIR="${run_dir}/data" \
SCENEWORKS_CONFIG_DIR="${run_dir}/config" \
SCENEWORKS_CREDENTIALS_DIR="${run_dir}/credentials" \
HF_HOME="${run_dir}/hf" \
SCENEWORKS_READINESS_MAX_ATTEMPTS=5 \
SCENEWORKS_READINESS_INTERVAL_SECONDS=0.05 \
SCENEWORKS_SHUTDOWN_GRACE_ATTEMPTS=6 \
SCENEWORKS_SHUTDOWN_GRACE_INTERVAL_SECONDS=0.05 \
bash "${entrypoint}" >"${run_dir}/stdout" 2>"${run_dir}/stderr" &
supervisor_pid=$!
wait_for_event "worker:start:http://127.0.0.1:8010:auto" "${events}"
wait_for_event "worker:nvidia-visible:all" "${events}"
wait_for_event "api:token:present" "${events}"
wait_for_event "worker:token:present" "${events}"
kill -TERM "${supervisor_pid}"
wait "${supervisor_pid}"
wait_for_event "api:term" "${events}"
wait_for_event "worker:term" "${events}"
[[ "$(cat "${run_dir}/curl-count")" -eq 2 ]] || fail "readiness did not retry exactly once"
if grep -Fq -- "${test_access_token}" "${run_dir}/stdout" "${run_dir}/stderr"; then
  fail "access token was written to supervisor output"
fi
unset test_access_token

run_dir="${temp_root}/child-failure"
mkdir -p "${run_dir}"
events="${run_dir}/events"
set +e
SCENEWORKS_TEST_EVENTS="${events}" \
SCENEWORKS_TEST_CURL_COUNT="${run_dir}/curl-count" \
SCENEWORKS_TEST_WORKER_EXIT_CODE=23 \
SCENEWORKS_API_HOST=127.0.0.2 \
SCENEWORKS_ACCESS_TOKEN= \
SCENEWORKS_API_BIN="${temp_root}/fake-api" \
SCENEWORKS_WORKER_BIN="${temp_root}/fake-worker" \
SCENEWORKS_CURL_BIN="${temp_root}/fake-curl" \
SCENEWORKS_DATA_DIR="${run_dir}/data" \
SCENEWORKS_CONFIG_DIR="${run_dir}/config" \
SCENEWORKS_CREDENTIALS_DIR="${run_dir}/credentials" \
HF_HOME="${run_dir}/hf" \
SCENEWORKS_READINESS_MAX_ATTEMPTS=5 \
SCENEWORKS_READINESS_INTERVAL_SECONDS=0.05 \
SCENEWORKS_SHUTDOWN_GRACE_ATTEMPTS=6 \
SCENEWORKS_SHUTDOWN_GRACE_INTERVAL_SECONDS=0.05 \
bash "${entrypoint}" >"${run_dir}/stdout" 2>"${run_dir}/stderr"
status=$?
set -e
[[ "${status}" -eq 23 ]] || fail "worker failure status was ${status}, expected 23"
wait_for_event "api:term" "${events}"
wait_for_event "api:token:absent" "${events}"
wait_for_event "worker:token:absent" "${events}"

run_dir="${temp_root}/readiness-failure"
mkdir -p "${run_dir}"
events="${run_dir}/events"
set +e
SCENEWORKS_TEST_EVENTS="${events}" \
SCENEWORKS_TEST_CURL_COUNT="${run_dir}/curl-count" \
SCENEWORKS_TEST_CURL_ALWAYS_FAIL=1 \
SCENEWORKS_API_HOST="[::1]" \
SCENEWORKS_ACCESS_TOKEN= \
SCENEWORKS_API_BIN="${temp_root}/fake-api" \
SCENEWORKS_WORKER_BIN="${temp_root}/fake-worker" \
SCENEWORKS_CURL_BIN="${temp_root}/fake-curl" \
SCENEWORKS_DATA_DIR="${run_dir}/data" \
SCENEWORKS_CONFIG_DIR="${run_dir}/config" \
SCENEWORKS_CREDENTIALS_DIR="${run_dir}/credentials" \
HF_HOME="${run_dir}/hf" \
SCENEWORKS_READINESS_MAX_ATTEMPTS=3 \
SCENEWORKS_READINESS_INTERVAL_SECONDS=0.05 \
SCENEWORKS_SHUTDOWN_GRACE_ATTEMPTS=6 \
SCENEWORKS_SHUTDOWN_GRACE_INTERVAL_SECONDS=0.05 \
bash "${entrypoint}" >"${run_dir}/stdout" 2>"${run_dir}/stderr"
status=$?
set -e
[[ "${status}" -eq 1 ]] || fail "readiness failure status was ${status}, expected 1"
wait_for_event "api:term" "${events}"
if grep -q '^worker:start:' "${events}"; then
  fail "worker started before API readiness"
fi

printf 'SceneWorks RunPod supervisor lifecycle self-test passed.\n'
