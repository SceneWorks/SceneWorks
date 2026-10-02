#!/usr/bin/env bash
# Invoked by check-runpod-privileges.sh with its CPU fixture image. No ACL
# utilities are present at runtime; all devices are simulated, never GPU/CUDA.
set -euo pipefail
image="$1"
test_root="$(mktemp -d)"
container="sceneworks-private-owned-test-$$"
volume="sceneworks-private-owned-test-$$"
cleanup() {
  docker rm -f "${container}" >/dev/null 2>&1 || true
  docker volume rm "${volume}" >/dev/null 2>&1 || true
  rm -rf "${test_root}"
}
trap cleanup EXIT
cat > "${test_root}/service" <<'SERVICE'
#!/usr/bin/env bash
set -euo pipefail
[[ "$(id -u):$(id -g)" == "${SCENEWORKS_SERVICE_UID}:${SCENEWORKS_SERVICE_GID}" ]]
for field in CapInh CapPrm CapEff CapBnd CapAmb; do grep -Eq "^${field}:[[:space:]]+0+$" /proc/self/status; done
grep -Eq '^NoNewPrivs:[[:space:]]+1$' /proc/self/status
expected_groups="${SCENEWORKS_SERVICE_GID}"
[[ "${SCENEWORKS_TEST_OWNED_DEVICE:-0}" != 1 ]] || expected_groups+=" 0"
[[ "$(id -G)" == "$expected_groups" ]]
[[ "${SCENEWORKS_TEST_OWNED_DEVICE:-0}" != 1 ]] || printf simulated > /dev/nvidia0
[[ "$(umask)" == 0077 ]]
for dir in "$SCENEWORKS_DATA_DIR" "$SCENEWORKS_CONFIG_DIR" "$SCENEWORKS_CREDENTIALS_DIR" "$HF_HOME" "${HF_HUB_CACHE:-$HUGGINGFACE_HUB_CACHE}" "${SCENEWORKS_JOBS_DB_PATH%/*}" "$HOME"; do
  mkdir -p "$dir/child-${SCENEWORKS_GPU_ID:-api}/grandchild"
  printf persisted > "$dir/child-${SCENEWORKS_GPU_ID:-api}/grandchild/file"
  mv "$dir/child-${SCENEWORKS_GPU_ID:-api}/grandchild/file" "$dir/child-${SCENEWORKS_GPU_ID:-api}/grandchild/renamed"
  printf reopened >> "$dir/child-${SCENEWORKS_GPU_ID:-api}/grandchild/renamed"
  [[ "$(< "$dir/child-${SCENEWORKS_GPU_ID:-api}/grandchild/renamed")" == persistedreopened ]]
done
if [[ -e "$HF_HOME/hub/snapshot/model" ]]; then [[ "$(< "$HF_HOME/hub/snapshot/model")" == model ]]; fi
if [[ "${SCENEWORKS_GPU_ID:-}" == auto ]]; then printf 'worker:ready\n'; else printf 'api:ready\n'; fi
trap 'printf "service:term\n"; exit 0' TERM
while true; do sleep 0.1; done
SERVICE
cat > "${test_root}/nvidia-smi" <<'SMI'
#!/bin/bash
[[ "$*" == '-q -x' ]] || exit 91
printf '    <minor_number>0</minor_number>\n'
SMI
cat > "${test_root}/fault" <<'FAULT'
#!/bin/bash
command="${0##*/}"
case "${SCENEWORKS_TEST_OWNED_FAULT:-}:$command" in
  chown-success:chown) exit 0 ;;
  launch:setpriv) exit 99 ;;
  timeout:timeout) exit 124 ;;
  probe-launch:setpriv)
    [[ "$*" != *'1000 1000 denied'* ]] || exit 99 ;;
  probe-timeout:timeout)
    [[ "$*" != *'1000 1000 denied'* ]] || exit 124 ;;
  permissive:chmod)
    for path in "$@"; do
      if [[ "$path" == *'.sceneworks-owned-test.'* && "$path" == "${SCENEWORKS_TEST_OWNED_FAULT_ROOT:-/}"* ]]; then
        mode=666; [[ ! -d "$path" ]] || mode=777
        /usr/bin/chmod "$mode" -- "$path"
        exit 0
      fi
    done ;;
  leaked-chown:chown)
    /usr/bin/chown "$@" || exit
    if [[ "$*" == *protected* ]]; then /usr/bin/chmod 666 protected; fi
    exit 0 ;;
  leaked-child:mv)
    /usr/bin/mv "$@" || exit
    /usr/bin/chmod 666 "${!#}"
    exit 0 ;;
  denied-open:bash)
    # Preserve ancestor setup, but independently fail the actual allow stage.
    [[ "$*" != *'protected private'* && "$*" != *'1000 1000 allowed'* ]] || exit 88 ;;
esac
exec "/usr/bin/$command" "$@"
FAULT
cat > "${test_root}/case" <<'CASE'
#!/usr/bin/env bash
set -euo pipefail
scenario="$1"
export SCENEWORKS_SERVICE_UID=1000 SCENEWORKS_SERVICE_GID=1000
export SCENEWORKS_PERMISSION_STRATEGY=private-owned SCENEWORKS_CANDLE_REQUIRED=0
export SCENEWORKS_API_HOST=127.0.0.1 SCENEWORKS_API_BIN=/owned-test/service SCENEWORKS_WORKER_BIN=/owned-test/service SCENEWORKS_CURL_BIN=/bin/true
if [[ "$scenario" == reused && -f /workspace/test-fixture-seeded ]]; then
  export PATH=/owned-tools:/owned-test:/usr/bin:/bin
  exec bash /test/runpod-entrypoint.sh
fi
mkdir -p /home/sceneworks
chown 1000:1000 /home/sceneworks
chmod 755 /home/sceneworks
printf original > /workspace/unrelated
chmod 600 /workspace/unrelated
before="$(stat -c %u:%g:%a /workspace)"
case "$scenario" in
  custom) export SCENEWORKS_SERVICE_UID=2345 SCENEWORKS_SERVICE_GID=2346 ;;
  collision) export SCENEWORKS_SERVICE_UID=65534 SCENEWORKS_SERVICE_GID=65534 ;;
  simulated-device)
    export SCENEWORKS_CANDLE_REQUIRED=1 SCENEWORKS_TEST_OWNED_DEVICE=1
    for node in nvidia0 nvidiactl nvidia-uvm; do mknod -m 660 "/dev/$node" c 1 3; done ;;

  overrides)
    export SCENEWORKS_DATA_DIR=/workspace/isolated/data SCENEWORKS_CONFIG_DIR=/workspace/isolated/config SCENEWORKS_CREDENTIALS_DIR=/workspace/isolated/tokens
    export HF_HOME=/workspace/isolated/hf HUGGINGFACE_HUB_CACHE=/workspace/isolated/legacy-hub SCENEWORKS_JOBS_DB_PATH=/tmp/private-jobs/queue/jobs.db ;;
  reused|nested|nested-fault|external-link|world-content|mixed-owner)
    setpriv --reuid=1000 --regid=1000 --clear-groups bash -euc '
      umask 077; mkdir -p /workspace/cache/huggingface/hub/{blobs,snapshot} /workspace/data
      printf model > /workspace/cache/huggingface/hub/blobs/model
      ln -s ../blobs/model /workspace/cache/huggingface/hub/snapshot/model
    '
    if [[ "$scenario" == nested* ]]; then
      mkdir -p /source '/workspace/data/nested mount'
      chown 1000:1000 /source '/workspace/data/nested mount'
      chmod 700 /source '/workspace/data/nested mount'
      mount --bind /source '/workspace/data/nested mount'
    elif [[ "$scenario" == external-link ]]; then ln -s /etc/passwd /workspace/data/external;
    elif [[ "$scenario" == world-content ]]; then chmod 666 /workspace/cache/huggingface/hub/blobs/model;
    elif [[ "$scenario" == mixed-owner ]]; then chown 0:0 /workspace/cache/huggingface/hub/blobs/model; fi ;;
  root-legacy)
    mkdir -p /workspace/config/legacy
    printf preserved > /workspace/config/legacy/file
    chmod 700 /workspace/config /workspace/config/legacy
    chmod 600 /workspace/config/legacy/file ;;
  unknown) export SCENEWORKS_PERMISSION_STRATEGY=unknown ;;
  unsafe) chmod 777 /workspace ;;
  inaccessible) chmod 700 /workspace ;;
  symlink) ln -s /tmp /workspace/config ;;
esac
# New paths need a safe existing creation ancestor. The provider root remains
# untouched by the initializer; this fixture deliberately grants pre-existing
# service creation access for reuse setup, then restores its safe ownership.
chown 0:0 /workspace
chmod 755 /workspace
[[ "$scenario" != unsafe ]] || chmod 777 /workspace
[[ "$scenario" != inaccessible ]] || chmod 700 /workspace
before="$(stat -c %u:%g:%a /workspace)"
if [[ "$scenario" == nested-fault ]]; then export SCENEWORKS_TEST_OWNED_FAULT=permissive SCENEWORKS_TEST_OWNED_FAULT_ROOT="/workspace/data/nested mount";
elif [[ "$scenario" == fault-* ]]; then export SCENEWORKS_TEST_OWNED_FAULT="${scenario#fault-}"; fi
mkdir -p /owned-tools
for tool in chown chmod setpriv timeout mv bash; do ln -s /owned-test/fault "/owned-tools/$tool"; done
export PATH=/owned-tools:/owned-test:/usr/bin:/bin
# Absence is real, rather than a fake successful ACL tool.
rm -f /usr/bin/setfacl /usr/bin/getfacl
case "$scenario" in
  fresh|custom|collision|overrides|reused|nested|simulated-device)
    [[ "$scenario" != reused ]] || printf seeded > /workspace/test-fixture-seeded
    exec bash /test/runpod-entrypoint.sh ;;
esac
set +e
bash /test/runpod-entrypoint.sh > /tmp/owned-startup.log 2>&1
status=$?
set -e
cat /tmp/owned-startup.log
[[ "$status" == 1 ]]
! grep -q ':ready' /tmp/owned-startup.log
[[ "$(stat -c %u:%g:%a /workspace)" == "$before" ]]
[[ "$(stat -c %u:%g:%a /workspace/unrelated)" == 0:0:600 && "$(< /workspace/unrelated)" == original ]]
[[ -z "$(find /workspace /home/sceneworks /tmp/sceneworks -name '.sceneworks-owned-test.*' -print 2>/dev/null)" ]]
if [[ "$scenario" == root-legacy ]]; then
  [[ "$(stat -c %u:%g:%a /workspace/config/legacy/file)" == 0:0:600 && "$(< /workspace/config/legacy/file)" == preserved ]]
fi
case "$scenario" in unknown|unsafe|inaccessible|symlink|root-legacy|mixed-owner|world-content|external-link)
  [[ ! -e /workspace/credentials && ! -e /tmp/sceneworks ]]
  [[ "$(stat -c %u:%g:%a /home/sceneworks)" == 1000:1000:755 ]] ;;
esac
printf 'OWNED_REFUSAL=PASS %s\n' "$scenario"
CASE
chmod +x "${test_root}"/*
for scenario in fresh custom collision overrides reused nested simulated-device unknown unsafe inaccessible symlink root-legacy mixed-owner world-content external-link nested-fault fault-chown-success fault-permissive fault-leaked-chown fault-leaked-child fault-launch fault-timeout fault-probe-launch fault-probe-timeout fault-denied-open; do
  docker volume create "${volume}" >/dev/null
  # Reuse fixtures are seeded with the service identity under this temporary
  # writable root; case restores 0755/root ownership before initialization.
  docker run --rm -v "${volume}:/workspace" "${image}" chmod 777 /workspace
  case "$scenario" in
    fresh|custom|collision|overrides|reused|nested|simulated-device)
      docker run -d --name "${container}" --cap-add=SYS_ADMIN -v "${volume}:/workspace" -v "${test_root}:/owned-test:ro" "${image}" bash /owned-test/case "$scenario" >/dev/null
      ready=0
      for ((attempt=0;attempt<150;attempt++)); do
        if docker logs "${container}" 2>&1 | grep -q worker:ready && docker logs "${container}" 2>&1 | grep -q api:ready; then ready=1; break; fi
        sleep 0.1
      done
      docker logs "${container}"
      [[ "$ready" == 1 ]]
      docker exec "${container}" bash -euc '
        [[ "$(stat -c %u:%g:%a /workspace)" == 0:0:755 ]]
        [[ "$(stat -c %u:%g:%a /workspace/unrelated)" == 0:0:600 && "$(< /workspace/unrelated)" == original ]]
        [[ -z "$(find /workspace ! -type l -perm -0002 -print)" ]]
        setpriv --reuid=65533 --regid=65533 --clear-groups --bounding-set=-all --inh-caps=-all --ambient-caps=-all --no-new-privs \
          bash -euc '\''if ( : < /workspace/unrelated ) 2>/dev/null; then exit 1; fi'\''
      '
      docker stop -t 5 "${container}" >/dev/null
      [[ "$(docker inspect -f '{{.State.ExitCode}}' "${container}")" == 0 ]]
      if [[ "$scenario" == reused ]]; then
        docker start "${container}" >/dev/null
        sleep 1
        docker exec "${container}" bash -euc '[[ "$(< /workspace/cache/huggingface/hub/snapshot/model)" == model ]]'
        docker stop -t 5 "${container}" >/dev/null
      fi
      docker rm "${container}" >/dev/null
      printf 'OWNED_RUNTIME=PASS %s\n' "$scenario" ;;
    *) docker run --rm --cap-add=SYS_ADMIN -v "${volume}:/workspace" -v "${test_root}:/owned-test:ro" "${image}" bash /owned-test/case "$scenario" ;;
  esac
  docker volume rm "${volume}" >/dev/null
done
printf 'RunPod private-owned CPU cases passed; no provider/CUDA claim.\n'
