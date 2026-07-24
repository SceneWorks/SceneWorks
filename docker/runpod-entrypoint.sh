#!/usr/bin/env bash
set -u

api_bin="${SCENEWORKS_API_BIN:-/usr/local/bin/sceneworks-rust-api}"
worker_bin="${SCENEWORKS_WORKER_BIN:-/usr/local/bin/sceneworks-rust-worker}"
curl_bin="${SCENEWORKS_CURL_BIN:-curl}"
api_port="${SCENEWORKS_API_PORT:-8010}"
api_url="http://127.0.0.1:${api_port}"
readiness_attempts="${SCENEWORKS_READINESS_MAX_ATTEMPTS:-120}"
readiness_interval="${SCENEWORKS_READINESS_INTERVAL_SECONDS:-1}"
shutdown_attempts="${SCENEWORKS_SHUTDOWN_GRACE_ATTEMPTS:-100}"
shutdown_interval="${SCENEWORKS_SHUTDOWN_GRACE_INTERVAL_SECONDS:-0.1}"

api_pid=""
worker_pid=""
shutting_down=0

log() {
  printf '[runpod] %s\n' "$*" >&2
}

is_running() {
  local pid="$1"
  local state
  [[ -n "${pid}" ]] || return 1
  if [[ "${OSTYPE:-}" == linux* && -r "/proc/${pid}/stat" ]]; then
    # `kill -0` still succeeds for an exited-but-not-yet-waited zombie. Reading
    # proc state keeps readiness/monitoring responsive and avoids burning the
    # full shutdown grace period before reaping a child that already stopped.
    read -r _ _ state _ <"/proc/${pid}/stat" || return 1
    [[ "${state}" != "Z" ]]
    return
  fi
  # Test/development shells without Linux procfs (not the production image).
  kill -0 "${pid}" 2>/dev/null
}

reap_status() {
  local pid="$1"
  local status
  set +e
  wait "${pid}"
  status=$?
  set -e
  printf '%s' "${status}"
}

stop_child() {
  local pid="$1"
  local label="$2"
  local attempt

  if [[ -z "${pid}" ]]; then
    return
  fi

  if is_running "${pid}"; then
    kill -TERM "${pid}" 2>/dev/null || true
  fi

  for ((attempt = 0; attempt < shutdown_attempts; attempt += 1)); do
    if ! is_running "${pid}"; then
      break
    fi
    sleep "${shutdown_interval}"
  done

  if is_running "${pid}"; then
    log "${label} child ${pid} exceeded the shutdown grace period; sending SIGKILL"
    kill -KILL "${pid}" 2>/dev/null || true
  fi
  wait "${pid}" 2>/dev/null || true
}

shutdown_children() {
  if (( shutting_down )); then
    return
  fi
  shutting_down=1
  trap - TERM INT HUP

  # Keep the API alive while the worker supervisor drains its children so their
  # final Offline heartbeat succeeds instead of producing shutdown-only transport
  # errors. Once every GPU/utility child is gone, stop and reap the API.
  stop_child "${worker_pid}" "worker supervisor"
  stop_child "${api_pid}" "API"
}

handle_signal() {
  log "shutdown signal received"
  shutdown_children
  exit 0
}

trap handle_signal TERM INT HUP

mkdir -p \
  "${SCENEWORKS_DATA_DIR:-/sceneworks/data}/cache" \
  "${SCENEWORKS_CONFIG_DIR:-/sceneworks/config}" \
  "${HF_HOME:-/sceneworks/data/cache/huggingface}"

log "starting embedded-web API on 0.0.0.0:${api_port}"
"${api_bin}" &
api_pid=$!

ready=0
for ((attempt = 1; attempt <= readiness_attempts; attempt += 1)); do
  if ! is_running "${api_pid}"; then
    status="$(reap_status "${api_pid}")"
    api_pid=""
    log "API exited before becoming healthy (status ${status})"
    exit "${status}"
  fi
  if "${curl_bin}" -fsS "${api_url}/api/v1/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep "${readiness_interval}"
done

if (( ! ready )); then
  log "API did not become healthy after ${readiness_attempts} attempts"
  shutdown_children
  exit 1
fi

log "API healthy; starting candle GPU and utility workers"
# `auto` is important here: the worker supervisor creates one child for every
# visible GPU plus a CPU-only child. child_environment() sets
# SCENEWORKS_UTILITY_JOBS=0 for GPU children and =1 for the CPU child, which is
# the capability-routing boundary this combined image relies on.
SCENEWORKS_API_URL="${api_url}" \
SCENEWORKS_GPU_ID=auto \
"${worker_bin}" &
worker_pid=$!

while true; do
  if ! is_running "${api_pid}"; then
    status="$(reap_status "${api_pid}")"
    api_pid=""
    log "API exited (status ${status}); stopping worker"
    shutdown_children
    exit "${status}"
  fi
  if ! is_running "${worker_pid}"; then
    status="$(reap_status "${worker_pid}")"
    worker_pid=""
    log "worker supervisor exited (status ${status}); stopping API"
    shutdown_children
    exit "${status}"
  fi
  sleep 0.2
done
