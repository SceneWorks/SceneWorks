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

run_dir="${temp_root}/signal"
mkdir -p "${run_dir}"
events="${run_dir}/events"
test_access_token="runpod-self-test-$PPID-$RANDOM"
SCENEWORKS_TEST_EVENTS="${events}" \
SCENEWORKS_TEST_CURL_COUNT="${run_dir}/curl-count" \
SCENEWORKS_TEST_EXPECTED_TOKEN="${test_access_token}" \
SCENEWORKS_API_HOST=0.0.0.0 \
SCENEWORKS_ACCESS_TOKEN="${test_access_token}" \
SCENEWORKS_ALLOW_OPEN_BIND=1 \
SCENEWORKS_API_BIN="${temp_root}/fake-api" \
SCENEWORKS_WORKER_BIN="${temp_root}/fake-worker" \
SCENEWORKS_CURL_BIN="${temp_root}/fake-curl" \
SCENEWORKS_DATA_DIR="${run_dir}/data" \
SCENEWORKS_CONFIG_DIR="${run_dir}/config" \
HF_HOME="${run_dir}/hf" \
SCENEWORKS_READINESS_MAX_ATTEMPTS=5 \
SCENEWORKS_READINESS_INTERVAL_SECONDS=0.05 \
SCENEWORKS_SHUTDOWN_GRACE_ATTEMPTS=6 \
SCENEWORKS_SHUTDOWN_GRACE_INTERVAL_SECONDS=0.05 \
bash "${entrypoint}" >"${run_dir}/stdout" 2>"${run_dir}/stderr" &
supervisor_pid=$!
wait_for_event "worker:start:http://127.0.0.1:8010:auto" "${events}"
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
