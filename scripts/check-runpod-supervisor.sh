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
printf 'api:start\n' >>"${SCENEWORKS_TEST_EVENTS}"
trap 'printf "api:term\n" >>"${SCENEWORKS_TEST_EVENTS}"; exit 0' TERM INT HUP
while true; do sleep 0.1; done
EOF

cat >"${temp_root}/fake-worker" <<'EOF'
#!/usr/bin/env bash
set -u
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

run_dir="${temp_root}/signal"
mkdir -p "${run_dir}"
events="${run_dir}/events"
SCENEWORKS_TEST_EVENTS="${events}" \
SCENEWORKS_TEST_CURL_COUNT="${run_dir}/curl-count" \
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
kill -TERM "${supervisor_pid}"
wait "${supervisor_pid}"
wait_for_event "api:term" "${events}"
wait_for_event "worker:term" "${events}"
[[ "$(cat "${run_dir}/curl-count")" -eq 2 ]] || fail "readiness did not retry exactly once"

run_dir="${temp_root}/child-failure"
mkdir -p "${run_dir}"
events="${run_dir}/events"
set +e
SCENEWORKS_TEST_EVENTS="${events}" \
SCENEWORKS_TEST_CURL_COUNT="${run_dir}/curl-count" \
SCENEWORKS_TEST_WORKER_EXIT_CODE=23 \
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

run_dir="${temp_root}/readiness-failure"
mkdir -p "${run_dir}"
events="${run_dir}/events"
set +e
SCENEWORKS_TEST_EVENTS="${events}" \
SCENEWORKS_TEST_CURL_COUNT="${run_dir}/curl-count" \
SCENEWORKS_TEST_CURL_ALWAYS_FAIL=1 \
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
