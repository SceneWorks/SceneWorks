#!/usr/bin/env bash
# Exact standalone candle image, actual API/worker services, CPU-only hosted
# startup/identity/bind-write evidence. Real CUDA remains a provider gate.
set -euo pipefail
api_image="${SCENEWORKS_CANDLE_SMOKE_API_IMAGE:-sceneworks-runpod-smoke:ci}"
worker_image="${SCENEWORKS_CANDLE_SMOKE_WORKER_IMAGE:-sceneworks-candle-smoke:ci}"
worker_entrypoint="${SCENEWORKS_CANDLE_SMOKE_WORKER_ENTRYPOINT:-}"
worker_default_user="${SCENEWORKS_CANDLE_SMOKE_WORKER_USER:-}"
# GPU-less CI must resolve the CUDA-linked executable before it can select CPU
# mode. Use NVIDIA's real compatibility library only for this smoke; deployed
# GPU containers still receive their driver through the NVIDIA runtime.
driver_args=()
if [[ -n "${SCENEWORKS_CANDLE_SMOKE_DRIVER_DIR:-}" ]]; then
  docker run --rm --entrypoint sh "${worker_image}" -ec '
    library="$1/libcuda.so.1"
    test -r "$library"
    test "$(od -An -tx1 -N4 "$library" | tr -d " \n")" = 7f454c46
  ' sh "${SCENEWORKS_CANDLE_SMOKE_DRIVER_DIR}"
  image_library_path="$(docker image inspect --format '{{json .Config.Env}}' "${worker_image}" |
    node -e 'const fs=require("node:fs");const env=JSON.parse(fs.readFileSync(0,"utf8"));process.stdout.write((env.find(v=>v.startsWith("LD_LIBRARY_PATH="))??"LD_LIBRARY_PATH=").slice(16))')"
  driver_args=(-e "LD_LIBRARY_PATH=${SCENEWORKS_CANDLE_SMOKE_DRIVER_DIR}${image_library_path:+:${image_library_path}}")
fi
# Report every unresolved dependency before starting services, instead of hiding
# a loader failure behind the registration deadline.
docker run --rm ${driver_args[@]+"${driver_args[@]}"} --entrypoint sh "${worker_image}" -ec '
  dependencies="$(ldd "$(command -v sceneworks-rust-worker)")"
  if printf "%s\n" "$dependencies" | grep -q "not found"; then
    printf "%s\n" "$dependencies" >&2; exit 1
  fi
'
smoke_root="$(mktemp -d)"
network="sceneworks-candle-smoke-$$"
api="${network}-api"
worker="${network}-worker"
assert_service_identity_and_writes() {
  local container="$1" uid="$2" gid="$3" expected_home="$4"
  docker exec "${container}" sh -ec '
    test "$(id -u)" = "$1"; test "$(id -g)" = "$2"; test "$HOME" = "$3"
    awk -v uid="$1" -v gid="$2" "/^Uid:/ {if (\$2 != uid || \$3 != uid || \$4 != uid || \$5 != uid) exit 1} /^Gid:/ {if (\$2 != gid || \$3 != gid || \$4 != gid || \$5 != gid) exit 1}" /proc/1/status
    for dir in "$HOME" /smoke/data /smoke/data/cache /smoke/config /smoke/credentials /smoke/hf; do
      probe="$(mktemp "$dir/.candle-smoke.XXXXXX")"
      printf smoke > "$probe"
      renamed="${probe}.renamed"
      mv "$probe" "$renamed"
      test "$(cat "$renamed")" = smoke
      rm "$renamed"
    done
    ! touch /etc/sceneworks-candle-smoke 2>/dev/null
  ' sh "${uid}" "${gid}" "${expected_home}"
}
# Pass the temporary token by environment name, never in command arguments/logs.
SCENEWORKS_ACCESS_TOKEN="$(node -e 'process.stdout.write(require("node:crypto").randomBytes(24).toString("hex"))')"
export SCENEWORKS_ACCESS_TOKEN
cleanup() {
  docker rm -f "${worker}" "${api}" >/dev/null 2>&1 || true
  docker network rm "${network}" >/dev/null 2>&1 || true
  # Only this run's mktemp bind tree: API fixtures may have container-only UIDs.
  docker run --rm --user 0 --entrypoint sh -v "${smoke_root}:/smoke" "${api_image}" \
    -ec 'find /smoke -mindepth 1 -delete' >/dev/null 2>&1 || true
  rm -rf "${smoke_root}"
}
trap cleanup EXIT
docker network create "${network}" >/dev/null
for identity in default override; do
  uid=1000 gid=1000
  user_args=()
  if [[ -n "${worker_default_user}" ]]; then
    user_args=(--user "${worker_default_user}")
  fi
  worker_entrypoint_args=()
  if [[ -n "${worker_entrypoint}" ]]; then
    worker_entrypoint_args=(--entrypoint "${worker_entrypoint}")
  fi
  home_args=()
  expected_home=/home/sceneworks
  if [[ "${identity}" == override ]]; then
    uid=2345 gid=2346
    user_args=(--user "${uid}:${gid}")
    home_args=(-e HOME=/smoke/data)
    expected_home=/smoke/data
  fi
  case_dir="${smoke_root}/${identity}"
  mkdir "${case_dir}"
  docker run --rm --user 0 --entrypoint sh -v "${case_dir}:/smoke" "${api_image}" -ec '
    for dir in /smoke /smoke/data /smoke/data/cache /smoke/config /smoke/credentials /smoke/hf; do
      mkdir -p "$dir"; chown "$1:$2" "$dir"; chmod 0750 "$dir"
    done
  ' sh "${uid}" "${gid}"
  common_args=(-v "${case_dir}:/smoke" -e SCENEWORKS_ACCESS_TOKEN
    -e SCENEWORKS_DATA_DIR=/smoke/data -e SCENEWORKS_CONFIG_DIR=/smoke/config
    -e SCENEWORKS_CREDENTIALS_DIR=/smoke/credentials -e HF_HOME=/smoke/hf
    -e SCENEWORKS_JOBS_DB_PATH=/smoke/data/cache/jobs.db)
  docker run -d --name "${api}" --network "${network}" --network-alias api \
    --user "${uid}:${gid}" --entrypoint sceneworks-rust-api "${common_args[@]}" \
    -e HOME=/smoke/data -e SCENEWORKS_API_HOST=0.0.0.0 -e SCENEWORKS_API_PORT=8010 \
    -e SCENEWORKS_CANDLE_REQUIRED=0 "${api_image}" >/dev/null
  ready=0
  for ((attempt=0; attempt<40; attempt++)); do
    if docker exec "${api}" curl -fsS --max-time 2 http://127.0.0.1:8010/api/v1/health >/dev/null 2>&1; then ready=1; break; fi
    sleep 0.5
  done
  [[ "${ready}" == 1 ]] || { docker logs "${api}"; exit 1; }
  assert_service_identity_and_writes "${api}" "${uid}" "${gid}" /smoke/data
  # Unless combined-image overrides are supplied, the default case exercises the
  # worker image's actual USER and CMD. Explicit CPU selection needs no GPU.
  docker run -d --name "${worker}" --network "${network}" \
    ${user_args[@]+"${user_args[@]}"} ${home_args[@]+"${home_args[@]}"} ${driver_args[@]+"${driver_args[@]}"} "${common_args[@]}" \
    ${worker_entrypoint_args[@]+"${worker_entrypoint_args[@]}"} \
    -e SCENEWORKS_API_URL=http://api:8010 -e SCENEWORKS_GPU_ID=cpu \
    -e SCENEWORKS_UTILITY_WORKERS=1 -e "SCENEWORKS_WORKER_ID=smoke-${identity}" \
    "${worker_image}" >/dev/null
  registered=0
  for ((attempt=0; attempt<40; attempt++)); do
    if docker exec "${api}" sh -ec 'curl -fsS --max-time 2 -H "Authorization: Bearer $SCENEWORKS_ACCESS_TOKEN" http://127.0.0.1:8010/api/v1/workers' |
      node -e 'let s="";process.stdin.on("data",v=>s+=v);process.stdin.on("end",()=>{try{const w=JSON.parse(s);process.exit(w.some(x=>x.id.startsWith("smoke-")&&x.gpuId==="cpu"&&!["offline","unhealthy"].includes(x.status))?0:1)}catch{process.exit(1)}})'; then
      registered=1; break
    fi
    sleep 0.5
  done
  [[ "${registered}" == 1 ]] || { docker logs "${worker}"; exit 1; }
  assert_service_identity_and_writes "${worker}" "${uid}" "${gid}" "${expected_home}"
  docker stop -t 15 "${worker}" >/dev/null
  [[ "$(docker inspect -f '{{.State.ExitCode}}' "${worker}")" == 0 ]]
  docker rm "${worker}" >/dev/null
  docker stop -t 15 "${api}" >/dev/null
  [[ "$(docker inspect -f '{{.State.ExitCode}}' "${api}")" == 0 ]]
  docker rm "${api}" >/dev/null
  printf 'Candle worker smoke %s identity %s:%s: real worker registered, bind writes passed, shutdown clean (CPU-only).\n' "${identity}" "${uid}" "${gid}"
done
