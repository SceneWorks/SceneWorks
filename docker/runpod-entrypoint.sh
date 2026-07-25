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

trim_whitespace() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "${value}"
}

require_absolute_path() {
  local label="$1"
  local path="$2"
  local non_slashes
  case "${path}" in
    /*) ;;
    *)
      log "${label} must be an absolute container path (got '${path}')"
      return 1
      ;;
  esac
  non_slashes="${path//\//}"
  if [[ -z "${non_slashes}" ]]; then
    log "${label} must name a managed subdirectory, not the filesystem root"
    return 1
  fi
  case "${path}" in
    */./* | */. | */../* | */..)
      log "${label} must not contain '.' or '..' path segments (got '${path}')"
      return 1
      ;;
  esac
}

ensure_writable_dir() {
  local label="$1"
  local dir="$2"
  local probe

  require_absolute_path "${label}" "${dir}" || return 1
  if ! mkdir -p -- "${dir}"; then
    log "cannot create ${label} at '${dir}'. The RunPod volume must be writable by the container root user."
    return 1
  fi
  if ! probe="$(mktemp "${dir}/.sceneworks-write-test.XXXXXX")"; then
    log "${label} at '${dir}' is not writable. This image runs as root and does not recursively chown network volumes; fix the mount's export/ACL permissions or choose a writable per-path override."
    return 1
  fi
  rm -f -- "${probe}"
}

effective_api_host() {
  local host
  if [[ -v SCENEWORKS_API_HOST ]]; then
    host="$(trim_whitespace "${SCENEWORKS_API_HOST}")"
    # Match the API's env_string fallback for an explicitly blank value.
    [[ -n "${host}" ]] || host="127.0.0.1"
  else
    # The combined image's Dockerfile sets this value. Keeping the same fallback
    # here makes direct entrypoint tests exercise the production posture.
    host="0.0.0.0"
  fi
  printf '%s' "${host}"
}

is_loopback_host() {
  case "$1" in
    127.* | "::1" | "[::1]") return 0 ;;
    *) return 1 ;;
  esac
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

# This is a combined-image contract, not merely a late API-server safeguard:
# fail before creating directories or starting any process when the configured
# bind is network-reachable and the token is missing/blank. The legacy API
# override is deliberately removed from every child so an injected value cannot
# bypass this image's secure-by-default posture.
api_host="$(effective_api_host)"
access_token_trimmed="$(trim_whitespace "${SCENEWORKS_ACCESS_TOKEN:-}")"
unset SCENEWORKS_ALLOW_OPEN_BIND
if ! is_loopback_host "${api_host}" && [[ -z "${access_token_trimmed}" ]]; then
  log "refusing network bind ${api_host}:${api_port}: SCENEWORKS_ACCESS_TOKEN is required by the combined RunPod image. Set a non-blank SCENEWORKS_ACCESS_TOKEN or bind to a loopback address; SCENEWORKS_ALLOW_OPEN_BIND cannot override this requirement."
  exit 1
fi
unset access_token_trimmed

# RunPod mounts a pod network volume at /workspace by default. Resolve every
# durable default from that one base at runtime so changing SCENEWORKS_VOLUME is
# enough; explicit per-path settings always win. Keep SQLite off NFS-style
# storage unless the operator deliberately overrides SCENEWORKS_JOBS_DB_PATH.
SCENEWORKS_VOLUME="${SCENEWORKS_VOLUME:-/workspace}"
SCENEWORKS_DATA_DIR="${SCENEWORKS_DATA_DIR:-${SCENEWORKS_VOLUME}/data}"
SCENEWORKS_CONFIG_DIR="${SCENEWORKS_CONFIG_DIR:-${SCENEWORKS_VOLUME}/config}"
SCENEWORKS_JOBS_DB_PATH="${SCENEWORKS_JOBS_DB_PATH:-/tmp/sceneworks/cache/jobs.db}"
HF_HOME="${HF_HOME:-${SCENEWORKS_VOLUME}/cache/huggingface}"

# Preserve the resolver's documented cache-variable precedence. In particular,
# an explicit legacy HUGGINGFACE_HUB_CACHE must not be shadowed by a generated
# HF_HUB_CACHE value.
if [[ -n "${HF_HUB_CACHE:-}" ]]; then
  effective_hf_hub_cache="${HF_HUB_CACHE}"
elif [[ -n "${HUGGINGFACE_HUB_CACHE:-}" ]]; then
  effective_hf_hub_cache="${HUGGINGFACE_HUB_CACHE}"
else
  HF_HUB_CACHE="${HF_HOME}/hub"
  HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
  effective_hf_hub_cache="${HF_HUB_CACHE}"
fi

export SCENEWORKS_VOLUME SCENEWORKS_DATA_DIR SCENEWORKS_CONFIG_DIR
export SCENEWORKS_JOBS_DB_PATH HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE

jobs_db_parent="${SCENEWORKS_JOBS_DB_PATH%/*}"
if [[ -z "${jobs_db_parent}" && "${SCENEWORKS_JOBS_DB_PATH}" == /* ]]; then
  jobs_db_parent="/"
fi
[[ "${jobs_db_parent}" != "${SCENEWORKS_JOBS_DB_PATH}" ]] || jobs_db_parent="."

# The runtime is intentionally root so a newly attached root-owned volume and a
# directory owned by a different numeric UID both work without a recursive
# ownership walk over multi-gigabyte model caches. Restrict startup work to the
# exact managed directories and verify writes before launching either child.
require_absolute_path "SCENEWORKS_VOLUME" "${SCENEWORKS_VOLUME}" || exit 1
require_absolute_path "SceneWorks data directory" "${SCENEWORKS_DATA_DIR}" || exit 1
require_absolute_path "SceneWorks config directory" "${SCENEWORKS_CONFIG_DIR}" || exit 1
require_absolute_path "Hugging Face home" "${HF_HOME}" || exit 1
require_absolute_path "Hugging Face hub cache" "${effective_hf_hub_cache}" || exit 1
require_absolute_path "jobs.db parent directory" "${jobs_db_parent}" || exit 1
ensure_writable_dir "SceneWorks data directory" "${SCENEWORKS_DATA_DIR}" || exit 1
ensure_writable_dir "SceneWorks config directory" "${SCENEWORKS_CONFIG_DIR}" || exit 1
ensure_writable_dir "Hugging Face home" "${HF_HOME}" || exit 1
ensure_writable_dir "Hugging Face hub cache" "${effective_hf_hub_cache}" || exit 1
ensure_writable_dir "jobs.db parent directory" "${jobs_db_parent}" || exit 1
unset effective_hf_hub_cache jobs_db_parent

log "starting embedded-web API on ${api_host}:${api_port}"
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
#
# RunPod can start PID 1 with NVIDIA_VISIBLE_DEVICES=void while the NVIDIA
# container runtime is still attaching the pod's configured GPU, even though
# later sessions report `all` and nvidia-smi succeeds. `void` is normally an
# intentional hard disable, so the Rust supervisor correctly refuses to probe
# it. This combined image is explicitly the automatic GPU deployment path:
# normalize visibility only for its auto-supervisor child so the supervisor's
# bounded nvidia-smi retry can observe the device once initialization completes.
# Explicit `SCENEWORKS_GPU_ID=cpu` workers launched outside this entrypoint keep
# their original visibility and CPU-only behavior.
SCENEWORKS_API_URL="${api_url}" \
SCENEWORKS_GPU_ID=auto \
NVIDIA_VISIBLE_DEVICES=all \
SCENEWORKS_ACCESS_TOKEN="${SCENEWORKS_ACCESS_TOKEN:-}" \
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
