#!/usr/bin/env bash
# Linux/Docker permission regression. Device is /dev/null under an NVIDIA name;
# this proves DAC access only, never CUDA or RunPod network-volume acceptance.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_root="$(mktemp -d)"
image="sceneworks-runpod-privilege-test-$$"
container="sceneworks-runpod-privilege-test-$$"
volume="sceneworks-runpod-privilege-test-$$"
cleanup() {
  docker rm -f "${container}" >/dev/null 2>&1 || true
  docker volume rm "${volume}" >/dev/null 2>&1 || true
  docker image rm "${image}" >/dev/null 2>&1 || true
  rm -rf "${test_root}"
}
trap cleanup EXIT
cp "${repo_root}/docker/runpod-entrypoint.sh" "${repo_root}/docker/runpod-privileges.sh" "${test_root}/"
cat > "${test_root}/Dockerfile" <<'DOCKER'
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends acl util-linux bash ca-certificates && rm -rf /var/lib/apt/lists/*
COPY . /test/
DOCKER
cat > "${test_root}/service" <<'SERVICE'
#!/usr/bin/env bash
set -euo pipefail
[[ "$(id -u)" == "${SCENEWORKS_SERVICE_UID}" ]]
[[ "$(id -g)" == "${SCENEWORKS_SERVICE_GID}" ]]
[[ "$(id -G)" == "${SCENEWORKS_SERVICE_GID} 0${SCENEWORKS_TEST_DELAYED:+ 2600}" ]]
for field in CapEff CapPrm CapBnd CapAmb; do
  grep -Eq "^${field}:[[:space:]]+0+$" /proc/self/status
done
grep -Eq '^NoNewPrivs:[[:space:]]+1$' /proc/self/status
[[ "$(id -u)" != 0 ]]
! touch /etc/service-must-not-write 2>/dev/null
! cat /workspace/unrelated/secret >/dev/null 2>&1
[[ "$(cat "${HF_HOME}/hub/models--fixture/snapshots/current/model")" == model ]]
printf updated > "${SCENEWORKS_CONFIG_DIR}/legacy/config"
if [[ -n "${SCENEWORKS_TEST_NESTED:-}" ]]; then
  [[ "$(cat "/workspace/data/nested mount/config")" == updated ]]
fi
printf output > "${SCENEWORKS_DATA_DIR}/legacy/output"
printf credential > "${SCENEWORKS_CREDENTIALS_DIR}/legacy-token"
printf cache > "${HF_HOME}/hub/models--fixture/new-download"
printf sqlite > "${SCENEWORKS_JOBS_DB_PATH}"
mkdir -p "${HOME}/.nv/ComputeCache"
printf kernel > "${HOME}/.nv/ComputeCache/fixture"
printf device > /dev/nvidia0
if [[ -n "${SCENEWORKS_TEST_DELAYED:-}" ]]; then printf device > /dev/nvidia1; fi
for parent in /workspace/data/readonly /workspace/data/restricted-default; do
  mkdir -p "$parent/child-${SCENEWORKS_GPU_ID:-api}/grandchild"
  printf nested > "$parent/child-${SCENEWORKS_GPU_ID:-api}/grandchild/file"
  printf reopened >> "$parent/child-${SCENEWORKS_GPU_ID:-api}/grandchild/file"
done
mkdir -p /workspace/config/masked/inherited
printf inherited > /workspace/config/masked/inherited/file
chmod 660 /workspace/config/masked/inherited/file
if [[ "${SCENEWORKS_GPU_ID:-}" == auto ]]; then
  [[ "${NVIDIA_VISIBLE_DEVICES}" == all ]]
  printf 'worker:ready\n'
else
  printf 'api:ready\n'
fi
trap 'printf "service:term\n"; exit 0' TERM
while true; do sleep 0.1; done
SERVICE
cat > "${test_root}/start-marker" <<'MARKER'
#!/usr/bin/env bash
printf started > /workspace/data/services-started
printf 'api:ready\n'
MARKER
mkdir "${test_root}/faults"
cat > "${test_root}/faults/chmod" <<'CHMOD'
#!/usr/bin/env bash
if [[ "${SCENEWORKS_TEST_PERMISSION_FAULT:-}" == permissive* ]]; then
  for path in "$@"; do
    if [[ "$path" == *'.sceneworks-permission-test.'* &&
          "$path" == "${SCENEWORKS_TEST_FAULT_ROOT}"/* ]]; then
      # Actual inode permissions become permissive; command status lies about
      # applying the requested restriction, just like the observed provider.
      mode=666
      [[ ! -d "$path" ]] || mode=777
      /usr/bin/chmod "$mode" -- "$path"
      exit 0
    fi
  done
fi
exec /usr/bin/chmod "$@"
CHMOD
cat > "${test_root}/faults/setfacl" <<'ACL'
#!/usr/bin/env bash
path="${!#}"
if [[ "$path" != *'.sceneworks-permission-test.'* ]]; then
  printf migrated >> /workspace/legacy-migration-started
elif [[ "$path" == "${SCENEWORKS_TEST_FAULT_ROOT}"/* ]]; then
  case "${SCENEWORKS_TEST_PERMISSION_FAULT:-}" in
    permissive*) cat >/dev/null; exit 0 ;;
    ignored-grant)
      if [[ "$1" == --no-mask && ! -d "$path" ]]; then
        cat >/dev/null
        exit 0
      fi ;;
    leaked-grant)
      if [[ "$1" == --no-mask ]]; then
        /usr/bin/setfacl "$@" || exit
        /usr/bin/chmod a+rwx -- "$path"
        exit 0
      fi ;;
    leaked-default)
      if [[ "$1" == --no-mask && "$path" == */private ]]; then
        /usr/bin/setfacl "$@" || exit
        /usr/bin/setfacl -m d:o::rwx -- "$path"
        exit $?
      fi ;;
  esac
fi
exec /usr/bin/setfacl "$@"
ACL
cat > "${test_root}/faults/getfacl" <<'GETACL'
#!/usr/bin/env bash
path="${!#}"
if [[ "${SCENEWORKS_TEST_PERMISSION_FAULT:-}" == permissive* &&
      "$path" == "${SCENEWORKS_TEST_FAULT_ROOT}"/* &&
      "$path" == *'.sceneworks-permission-test.'* ]]; then
  rights=rw-
  [[ ! -d "$path" ]] || rights=rwx
  printf 'user::%s\ngroup::---\nother::---\n' "$rights"
  exit 0
fi
exec /usr/bin/getfacl "$@"
GETACL
cat > "${test_root}/faults/setpriv" <<'PRIV'
#!/usr/bin/env bash
[[ "${SCENEWORKS_TEST_PERMISSION_FAULT:-}" != launch-failed ]] || exit 99
exec /usr/bin/setpriv "$@"
PRIV
cat > "${test_root}/faults/timeout" <<'TIMEOUT'
#!/usr/bin/env bash
[[ "${SCENEWORKS_TEST_PERMISSION_FAULT:-}" != timed-out ]] || exit 124
exec /usr/bin/timeout "$@"
TIMEOUT
cat > "${test_root}/start" <<'START'
#!/usr/bin/env bash
set -euo pipefail
mkdir -p /workspace/{data/legacy,config/legacy,credentials,cache/huggingface/hub/models--fixture/{blobs,snapshots/current},unrelated}
printf model > /workspace/cache/huggingface/hub/models--fixture/blobs/model
ln -sf ../../blobs/model /workspace/cache/huggingface/hub/models--fixture/snapshots/current/model
printf config > /workspace/config/legacy/config
printf token > /workspace/credentials/legacy-token
printf private > /workspace/unrelated/secret
chmod 700 /workspace /workspace/cache /workspace/unrelated /workspace/config/legacy
chmod 600 /workspace/config/legacy/config /workspace/credentials/legacy-token
chmod 660 /dev/nvidia0
mknod -m 660 /dev/nvidiactl c 1 3
mknod -m 660 /dev/nvidia-uvm c 1 3
mkdir -p /workspace/data/readonly /workspace/data/restricted-default
chmod 0555 /workspace/data/readonly /workspace/data/restricted-default
setfacl -m d:u::r-x,d:u:2501:rwx,d:g::rwx,d:g:2502:rwx,d:m::r-x,d:o::--- /workspace/data/restricted-default
# Real masked owning-group and named-user/group entries, both access/default.
mkdir -p /workspace/config/masked
setfacl -m u:2501:rwx,g::rwx,g:2502:rwx,m::r-x /workspace/config/masked
setfacl -m d:u::rwx,d:u:2501:rwx,d:g::rwx,d:g:2502:rwx,d:m::r-x,d:o::--- /workspace/config/masked
setfacl -m u:2501:rwx,g::rwx,g:2502:rwx,m::--- /workspace/cache
setfacl -m u:2501:--x,u:2504:--x /workspace
if [[ -n "${SCENEWORKS_TEST_NESTED:-}" ]]; then
  mkdir -p "/workspace/data/nested mount"
  mount --bind /workspace/config/legacy "/workspace/data/nested mount"
fi
if [[ -n "${SCENEWORKS_TEST_DELAYED:-}" ]]; then
  (
    while [[ ! -f /tmp/device-probe-started ]]; do sleep 0.05; done
    sleep 0.4
    mknod -m 660 /dev/nvidia1 c 1 3
    chgrp 2600 /dev/nvidia1
  ) &
fi
exec bash /test/runpod-entrypoint.sh
START
cat > "${test_root}/nvidia-smi" <<'SMI'
#!/usr/bin/env bash
[[ "$*" == "-q -x" ]] || exit 91
printf started >/tmp/device-probe-started
[[ "${SCENEWORKS_TEST_DEVICE_MISSING:-0}" != 1 ]] || exit 1
if [[ "${SCENEWORKS_TEST_DEVICE_HUNG:-0}" == 1 ]]; then trap "" TERM; sleep 30; fi
printf '    <minor_number>0</minor_number>\n'
[[ -z "${SCENEWORKS_TEST_DELAYED:-}" ]] || printf '    <minor_number>1</minor_number>\n'
SMI
chmod +x "${test_root}/service" "${test_root}/nvidia-smi" "${test_root}/start-marker" "${test_root}"/faults/*
docker build -q -t "${image}" "${test_root}" >/dev/null
for scenario in default override collision nested delayed; do
  identity=1000:1000
  delayed=""
  nested=""
  docker_options=(--name "${container}" --device=/dev/null:/dev/nvidia0)
  [[ "${scenario}" != override ]] || identity=2345:2346
  [[ "${scenario}" != collision ]] || identity=65534:65534
  [[ "${scenario}" != nested ]] || { nested=1; docker_options+=(--cap-add=SYS_ADMIN); }
  [[ "${scenario}" != delayed ]] || delayed=1
  docker volume create "${volume}" >/dev/null
  docker run -d "${docker_options[@]}" \
    -v "${volume}:/workspace" \
    -e SCENEWORKS_SERVICE_UID="${identity%:*}" -e SCENEWORKS_SERVICE_GID="${identity#*:}" \
    -e SCENEWORKS_API_HOST=127.0.0.1 -e SCENEWORKS_API_BIN=/test/service \
    -e PATH=/test:/usr/bin:/bin -e SCENEWORKS_TEST_DELAYED="${delayed}" \
    -e SCENEWORKS_TEST_NESTED="${nested}" \
    -e SCENEWORKS_DEVICE_READINESS_TIMEOUT_SECONDS=5 -e SCENEWORKS_DEVICE_READINESS_INTERVAL_SECONDS=0.1 \
    -e SCENEWORKS_WORKER_BIN=/test/service -e SCENEWORKS_CURL_BIN=/bin/true \
    "${image}" bash /test/start >/dev/null
  ready=0
  for (( attempt=0; attempt<100; attempt++ )); do
    if docker logs "${container}" 2>&1 | grep -q 'worker:ready'; then ready=1; break; fi
    sleep 0.1
  done
  docker logs "${container}"
  [[ "${ready}" == 1 ]] || exit 1
  docker exec "${container}" bash -ec '
    test "$(stat -c %u:%g /workspace/config/legacy/config)" = 0:0
    test "$(stat -c %a /workspace/unrelated)" = 700
    test "$(stat -c %a /workspace/unrelated/secret)" = 644
    ! getfacl -cp /workspace/unrelated | grep -q "user:[0-9]"
    test -L /workspace/cache/huggingface/hub/models--fixture/snapshots/current/model
    test -z "$(find /workspace ! -type l -perm -0002 -print)"
    grep -Eq "^Uid:[[:space:]]+${SCENEWORKS_SERVICE_UID}[[:space:]]" /proc/1/status
    for path in /workspace/config/masked /workspace/config/masked/inherited; do
      acl="$(getfacl -cpEn "$path")"
      for entry in user:2501:r-x group::r-x group:2502:r-x default:user:2501:r-x default:group::r-x default:group:2502:r-x; do
        grep -Fxq "$entry" <<<"$acl"
      done
    done
    test "$(stat -c %u /workspace/data/readonly)" = 0
    test "$(getfacl -cpEn /workspace/data/readonly | sed -n /^user::/p)" = user::r-x
    test "$(getfacl -cpEn /workspace/data/restricted-default | sed -n /^user::/p)" = user::r-x
    for parent in /workspace/data/readonly /workspace/data/restricted-default; do
      for child in child-api child-auto; do
        test "$(cat "$parent/$child/grandchild/file")" = nestedreopened
        test "$(stat -c %u "$parent/$child/grandchild/file")" = "$SCENEWORKS_SERVICE_UID"
      done
    done
    acl="$(getfacl -cpEn /workspace/data/restricted-default/child-api)"
    for entry in user:2501:r-x group::r-x group:2502:r-x default:user:2501:r-x default:group::r-x default:group:2502:r-x; do grep -Fxq "$entry" <<<"$acl"; done
    acl="$(getfacl -cpEn /workspace/cache)"
    for entry in user:2501:--- group::--- group:2502:---; do grep -Fxq "$entry" <<<"$acl"; done
    # Unrelated users/groups can still traverse/read their granted tree, but
    # cannot create files there or traverse a previously mask-denied ancestor.
    for principal in user group owning-group; do
      principal_gid=2503
      principal_uid=2501
      [[ "$principal" != group ]] || { principal_uid=2504; principal_gid=2502; }
      [[ "$principal" != owning-group ]] || { principal_uid=2504; principal_gid=0; }
      setpriv --reuid="$principal_uid" --regid="$principal_gid" --clear-groups bash -ec "
        test -x /workspace/config/masked
        ! touch /workspace/config/masked/unrelated-write 2>/dev/null
        ! touch /workspace/config/masked/inherited/unrelated-write 2>/dev/null
        ! test -x /workspace/cache
      "
    done
  ' || exit 1
  docker stop -t 10 "${container}" >/dev/null
  [[ "$(docker inspect -f '{{.State.ExitCode}}' "${container}")" == 0 ]] || exit 1
  [[ "$(docker logs "${container}" 2>&1 | grep -c service:term)" == 2 ]] || exit 1
  docker rm "${container}" >/dev/null
  docker volume rm "${volume}" >/dev/null
 done
# Invalid identity and symlinked managed roots fail before child startup.
for scenario in root-id symlink device-timeout device-hung; do
  docker volume create "${volume}" >/dev/null
  started_at=${SECONDS}
  set +e
  docker run --rm -v "${volume}:/workspace" \
    -e SCENEWORKS_API_HOST=127.0.0.1 -e SCENEWORKS_API_BIN=/test/service \
    -e SCENEWORKS_WORKER_BIN=/test/service "${image}" bash -ec '
      if [[ "$1" == root-id ]]; then export SCENEWORKS_SERVICE_UID=0;
      elif [[ "$1" == symlink ]]; then ln -s /etc /workspace/data;
      else
        export PATH=/test:/usr/bin:/bin
        if [[ "$1" == device-timeout ]]; then export SCENEWORKS_TEST_DEVICE_MISSING=1;
        else export SCENEWORKS_TEST_DEVICE_HUNG=1; fi
        export SCENEWORKS_DEVICE_READINESS_TIMEOUT_SECONDS=1 SCENEWORKS_DEVICE_READINESS_INTERVAL_SECONDS=10
      fi
      exec bash /test/runpod-entrypoint.sh
    ' _ "${scenario}" >"${test_root}/failure" 2>&1
  status=$?
  set -e
  [[ "${status}" == 1 ]] || exit 1
  if [[ "${scenario}" == device-* ]]; then
    (( SECONDS - started_at < 5 )) || exit 1
    grep -q "initialization exceeded 1s" "${test_root}/failure" || exit 1
  fi
  ! grep -q ':ready' "${test_root}/failure" || exit 1
  docker volume rm "${volume}" >/dev/null
done
# Real opens expose permissive inodes even when permission tools claim success.
# Failure on a later root or nested mount must precede all legacy ACL migration.
for scenario in permissive permissive-override permissive-nested ignored-grant leaked-grant leaked-default launch-failed timed-out; do
  docker volume create "${volume}" >/dev/null
  set +e
  docker run --rm --cap-add=SYS_ADMIN -v "${volume}:/workspace" \
    -e SCENEWORKS_TEST_PERMISSION_FAULT="${scenario}" \
    -e SCENEWORKS_API_HOST=127.0.0.1 -e SCENEWORKS_CANDLE_REQUIRED=0 \
    -e SCENEWORKS_API_BIN=/test/start-marker -e SCENEWORKS_WORKER_BIN=/test/start-marker \
    "${image}" bash -euc '
      mkdir -p /workspace/data /workspace/config /workspace/credentials
      printf preserved > /workspace/config/legacy
      chmod 600 /workspace/config/legacy
      export SCENEWORKS_TEST_FAULT_ROOT=/workspace/data
      case "$1" in
        permissive-override)
          mkdir -p /overrides/config
          mount --bind /workspace/config /overrides/config
          export SCENEWORKS_CONFIG_DIR=/overrides/config SCENEWORKS_TEST_FAULT_ROOT=/overrides/config ;;
        permissive-nested)
          mkdir -p /nested-source "/workspace/data/nested mount"
          mount --bind /nested-source "/workspace/data/nested mount"
          export SCENEWORKS_TEST_FAULT_ROOT="/workspace/data/nested mount" ;;
      esac
      export PATH=/test/faults:/usr/bin:/bin
      if [[ "$1" == permissive* ]]; then
        # Independently prove that the fake filesystem allows a real reduced-UID
        # read/append although chmod and restrictive ACL tools return success.
        control="$(mktemp -d "$SCENEWORKS_TEST_FAULT_ROOT/.sceneworks-permission-test.XXXXXX")"
        chmod 700 "$control"
        printf control > "$control/protected"
        chmod 600 "$control/protected"
        setfacl -b "$control/protected" </dev/null
        getfacl -cpEn "$control/protected" | grep -Fxq other::---
        setpriv --reuid=1000 --regid=1000 --clear-groups --bounding-set=-all \
          --inh-caps=-all --ambient-caps=-all --no-new-privs \
          bash -euc '\''[[ "$(< "$1")" == control ]]; printf changed >> "$1"'\'' _ "$control/protected"
        rm -rf "$control"
      fi
      set +e
      bash /test/runpod-entrypoint.sh > /workspace/startup-log 2>&1
      status=$?
      set -e
      cat /workspace/startup-log
      [[ "$status" == 1 ]]
      grep -q "does not enforce private permissions" /workspace/startup-log
      [[ ! -e /workspace/data/services-started && ! -e /workspace/legacy-migration-started ]]
      [[ "$(stat -c %u:%g:%a /workspace/config/legacy)" == 0:0:600 ]]
      [[ "$(cat /workspace/config/legacy)" == preserved ]]
      [[ -z "$(find /workspace /overrides /home/sceneworks -name .sceneworks-permission-test.\* -print 2>/dev/null)" ]]
    ' _ "${scenario}" >"${test_root}/permission-failure" 2>&1
  status=$?
  set -e
  cat "${test_root}/permission-failure"
  [[ "${status}" == 0 ]] || exit 1
  docker volume rm "${volume}" >/dev/null
done
printf 'RunPod Linux nonroot mount/device permission tests passed (simulated device; no CUDA/provider claim).\n'
