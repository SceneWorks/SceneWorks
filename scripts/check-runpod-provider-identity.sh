#!/usr/bin/env bash
# Execute inside an authorized candidate Pod as root. This is a bounded identity,
# storage and real device check; a successful generation is a separate acceptance.
set -euo pipefail
uid="${SCENEWORKS_SERVICE_UID:-1000}"
gid="${SCENEWORKS_SERVICE_GID:-1000}"
[[ "$(id -u)" == 0 && "${uid}" != 0 && "${gid}" != 0 ]]
api_count=0
worker_count=0
for status in /proc/[0-9]*/status; do
  pid="${status#/proc/}"
  pid="${pid%/status}"
  binary="$(readlink "/proc/${pid}/exe" 2>/dev/null || true)"
  if [[ "${pid}" != 1 && "${binary}" != */sceneworks-rust-api && "${binary}" != */sceneworks-rust-worker ]]; then continue; fi
  [[ "${binary}" != */sceneworks-rust-api ]] || api_count=$((api_count + 1))
  [[ "${binary}" != */sceneworks-rust-worker ]] || worker_count=$((worker_count + 1))
  awk -v uid="${uid}" -v gid="${gid}" '
    /^Uid:/ { for (i=2; i<=5; i++) if ($i != uid) exit 1 }
    /^Gid:/ { for (i=2; i<=5; i++) if ($i != gid) exit 1 }
    /^Cap(Inh|Prm|Eff|Bnd|Amb):/ { if ($2 !~ /^0+$/) exit 1 }
    /^NoNewPrivs:/ { if ($2 != 1) exit 1 }
  ' "${status}"
  printf 'Verified service process pid=%s uid=%s gid=%s\n' "${pid}" "${uid}" "${gid}"
done
[[ "${api_count}" == 1 && "${worker_count}" -ge 3 ]]
groups="$(awk '/^Groups:/ { for(i=2;i<=NF;i++) printf "%s%s",(i>2?",":""),$i }' /proc/1/status)"
group_args=(--clear-groups)
[[ -z "${groups}" ]] || group_args=("--groups=${groups}")
setpriv --reuid="${uid}" --regid="${gid}" "${group_args[@]}" \
  --bounding-set=-all --inh-caps=-all --ambient-caps=-all --no-new-privs bash -euc '
  volume="${SCENEWORKS_VOLUME:-/workspace}"
  hf="${HF_HOME:-${volume}/cache/huggingface}"
  db="${SCENEWORKS_JOBS_DB_PATH:-/tmp/sceneworks/cache/jobs.db}"
  for dir in "${SCENEWORKS_DATA_DIR:-${volume}/data}" "${SCENEWORKS_CONFIG_DIR:-${volume}/config}" \
    "${SCENEWORKS_CREDENTIALS_DIR:-${volume}/credentials}" "${HF_HUB_CACHE:-${HUGGINGFACE_HUB_CACHE:-${hf}/hub}}" "${db%/*}" /home/sceneworks; do
    probe="$(mktemp "${dir}/.sc24258-acceptance.XXXXXX")"
    rm -- "${probe}"
  done
  nvidia-smi --query-gpu=name,uuid,memory.total --format=csv
  curl -fsS "http://127.0.0.1:${SCENEWORKS_API_PORT:-8010}/api/v1/health"
'
printf '\nProvider process/storage/device checks passed; generation and restart evidence still required.\n'
