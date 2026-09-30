#!/usr/bin/env bash
# Sourced only during root initialization, before starting any service.
# Rebuild each ACL atomically after clipping mask-controlled entries to their
# existing effective permissions. Widening a mask must never reactivate dormant
# permissions for a different user/group, including inherited default ACLs.
grant_service_acl() (
  set -o pipefail
  local path="$1" grant="$2" defaults=0 owner
  [[ "${grant}" != rwX ]] || {
    grant=rw-
    [[ ! -d "${path}" && ! -x "${path}" ]] || grant=rwx
    [[ ! -d "${path}" ]] || defaults=1
  }
  owner="$(stat -c %u -- "${path}")" || return 1
  getfacl -cpEn -- "${path}" | awk -F: -v uid="${service_uid}" \
    -v owner="${owner}" -v grant="${grant}" -v defaults="${defaults}" '
    function intersect(a,b, r,i) {
      r=""; for(i=1;i<=3;i++) r=r (substr(a,i,1)==substr(b,i,1)?substr(a,i,1):"-")
      return r
    }
    function unite(a,b, r,i) {
      r=""; for(i=1;i<=3;i++) r=r (substr(a,i,1)!="-"?substr(a,i,1):substr(b,i,1))
      return r
    }
    NF >= 3 {
      scope=($1=="default"?"d":"a"); offset=(scope=="d"?1:0)
      kind=$(1+offset); id=$(2+offset); perms=$(3+offset)
      key=kind ":" id
      acl[scope,key]=perms; scopes[scope]=1
      if(kind=="mask") masks[scope]=perms
    }
    END {
      for(scope in scopes) {
        for(entry in acl) {
          split(entry, pair, SUBSEP); if(pair[1]!=scope) continue
          split(pair[2], who, ":")
          if((who[1]=="group" || (who[1]=="user" && who[2]!="")) && (scope in masks))
            acl[entry]=intersect(acl[entry], masks[scope])
        }
      }
      if(defaults && !("d" in scopes)) {
        scopes["d"]=1
        acl["d","user:"]=acl["a","user:"]
        acl["d","group:"]=acl["a","group:"]
        acl["d","other:"]=acl["a","other:"]
      }
      for(scope in scopes) {
        required=(scope=="a"?grant:"rwx")
        # Ancestor traversal changes only access ACLs, never default ACLs.
        if(scope=="a" || defaults) {
          key="user:" uid
          existing=((scope SUBSEP key) in acl?acl[scope,key]:"---")
          acl[scope,key]=unite(existing,required)
          # A newly created inode belongs to its creator. Its owner entry wins
          # over the named service entry, so descendants need usable owner
          # rights; their creation mode still filters these defaults. Existing
          # inode owners and unrelated named/group effective rights stay intact.
          if(scope=="d") acl[scope,"user:"]="rwx"
          if(scope=="a" && owner==uid) acl[scope,"user:"]=unite(acl[scope,"user:"],required)
          mask="---"
          for(entry in acl) {
            split(entry,pair,SUBSEP); if(pair[1]!=scope) continue
            split(pair[2],who,":")
            if(who[1]=="group" || (who[1]=="user" && who[2]!="")) mask=unite(mask,acl[entry])
          }
          acl[scope,"mask:"]=mask
        }
      }
      for(entry in acl) {
        split(entry,pair,SUBSEP)
        print (pair[1]=="d"?"default:":"") pair[2] ":" acl[entry]
      }
    }' | setfacl --no-mask --set-file=- -- "${path}"
)

grant_service_tree() (
  set -o pipefail
  local path
  find -P "$1" ! -type l -print0 | while IFS= read -r -d "" path; do
    grant_service_acl "${path}" rwX || exit 1
  done
)

# Successful chmod/setfacl calls do not prove that a provider enforces them.
# Work only on disposable inodes, before changing any legacy permissions.
preflight_runpod_permissions() (
  local dir="$1" scratch="" unrelated_uid=65534 unrelated_gid=65534
  [[ "${service_uid}" != "${unrelated_uid}" ]] || unrelated_uid=65533
  [[ "${service_gid}" != "${unrelated_gid}" ]] || unrelated_gid=65533
  trap '[[ -z "${scratch}" ]] || rm -rf -- "${scratch}"' EXIT
  scratch="$(mktemp -d "${dir}/.sceneworks-permission-test.XXXXXX")" || return 1
  # Remove inherited grants on our own fixture only; keep the parent untouched.
  setfacl -b -k -- "${scratch}" && chmod 0711 -- "${scratch}" || return 1
  ( umask 077
    mkdir -- "${scratch}/private" &&
    printf 'fixture\n' > "${scratch}/protected" &&
    printf 'fixture\n' > "${scratch}/private/protected"
  ) || return 1
  chmod 0600 -- "${scratch}/protected" "${scratch}/private/protected" &&
    chmod 0700 -- "${scratch}/private" || return 1
  # Relative lookups from an inherited cwd exercise this filesystem even when
  # an ancestor denies traversal. Such a denial must not mask a permissive mount.
  cd -- "${scratch}" || return 1
  probe_permission_identity() {
    timeout --signal=KILL 5 setpriv --reuid="$1" --regid="$2" --clear-groups \
      --bounding-set=-all --inh-caps=-all --ambient-caps=-all --no-new-privs \
      bash -euc '
        [[ "$(id -u)" == "$1" && "$(id -g)" == "$2" ]]
        case "$3" in
          denied)
            if ( : < protected ) 2>/dev/null; then exit 1; fi
            if ( : >> protected ) 2>/dev/null; then exit 1; fi
            if ( cd private ) 2>/dev/null; then exit 1; fi
            if ( : > private/unauthorized ) 2>/dev/null; then exit 1; fi
            ;;
          allowed)
            IFS= read -r content < protected
            [[ "$content" == fixture ]]
            printf appended >> protected
            mkdir -p private/child/grandchild
            printf inherited > private/child/grandchild/file
            printf reopened >> private/child/grandchild/file
            [[ "$(< private/child/grandchild/file)" == inheritedreopened ]]
            ;;
          unrelated)
            if ( : < protected ) 2>/dev/null; then exit 1; fi
            if ( : >> protected ) 2>/dev/null; then exit 1; fi
            if ( cd private ) 2>/dev/null; then exit 1; fi
            if ( : < private/child/grandchild/file ) 2>/dev/null; then exit 1; fi
            if ( : >> private/child/grandchild/file ) 2>/dev/null; then exit 1; fi
            if ( : > private/unauthorized ) 2>/dev/null; then exit 1; fi
            ;;
          inherited)
            if ( : < file ) 2>/dev/null; then exit 1; fi
            if ( : >> file ) 2>/dev/null; then exit 1; fi
            if ( : > unauthorized ) 2>/dev/null; then exit 1; fi
            ;;
          *) exit 1 ;;
        esac
      ' _ "$1" "$2" "$3"
  }
  # The child returns success only after proving every expected denial. A
  # setpriv/exec/timeout failure is therefore never mistaken for enforcement.
  probe_permission_identity "${service_uid}" "${service_gid}" denied &&
    grant_service_tree "${scratch}" &&
    probe_permission_identity "${service_uid}" "${service_gid}" allowed &&
    probe_permission_identity "${unrelated_uid}" "${unrelated_gid}" unrelated || return 1
  # Do not let the root-owned private parent conceal broken default inheritance.
  cd -- private/child/grandchild &&
    probe_permission_identity "${unrelated_uid}" "${unrelated_gid}" inherited || return 1
  cd -- "${dir}" && rm -rf -- "${scratch}" || return 1
  scratch=""
)

wait_for_runpod_devices() {
  # CUDA images require all driver-enumerated GPU nodes plus control/UVM before
  # irrevocably dropping privileges. Use device minor numbers: NVML indices can
  # be renumbered when the provider exposes a subset of host GPUs. A CPU-only
  # diagnostic may opt out explicitly.
  [[ "${SCENEWORKS_CANDLE_REQUIRED:-1}" != 0 ]] || return 0
  local timeout_seconds="${SCENEWORKS_DEVICE_READINESS_TIMEOUT_SECONDS:-63}"
  local interval="${SCENEWORKS_DEVICE_READINESS_INTERVAL_SECONDS:-1}"
  local deadline remaining probe_timeout report indices index devices_ready device
  if [[ ! "${timeout_seconds}" =~ ^[1-9][0-9]{0,4}$ ||
        ! "${interval}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    log "device readiness timeout must be positive integer seconds and interval nonnegative seconds"
    return 1
  fi
  deadline=$((SECONDS + timeout_seconds))
  while (( SECONDS < deadline )); do
    remaining=$((deadline - SECONDS))
    probe_timeout=3
    (( remaining >= probe_timeout )) || probe_timeout="${remaining}"
    devices_ready=0
    if report="$(NVIDIA_VISIBLE_DEVICES=all timeout --signal=KILL "${probe_timeout}" nvidia-smi -q -x 2>/dev/null)"; then
      # minor_number is emitted by the XML report, not the selective CSV query.
      indices="$(printf '%s\n' "${report}" | sed -n 's/^[[:space:]]*<minor_number>\([^<]*\)<\/minor_number>[[:space:]]*$/\1/p')"
      devices_ready=1
      [[ -n "${indices}" ]] || devices_ready=0
      while IFS= read -r index; do
        if [[ ! "${index}" =~ ^[0-9]+$ || ! -c "/dev/nvidia${index}" || -L "/dev/nvidia${index}" ]]; then devices_ready=0; fi
      done <<<"${indices}"
      for device in /dev/nvidiactl /dev/nvidia-uvm; do
        [[ -c "${device}" && ! -L "${device}" ]] || devices_ready=0
      done
    fi
    (( ! devices_ready )) || return 0
    remaining=$((deadline - SECONDS))
    (( remaining > 0 )) || break
    # Cap even an operator-supplied long interval at the remaining deadline.
    timeout --signal=KILL "${remaining}" sleep "${interval}" || true
  done
  log "NVIDIA device initialization exceeded ${timeout_seconds}s; required GPU/control/UVM nodes are not ready. Services were not started."
  return 1
}

initialize_runpod_service() {
  service_uid="${SCENEWORKS_SERVICE_UID:-1000}"
  service_gid="${SCENEWORKS_SERVICE_GID:-1000}"
  local dir parent device device_gid id_value physical_dir target
  local -a mount_fields
  local device_groups=""
  privilege_group_args=(--clear-groups)
  for id_value in "${service_uid}" "${service_gid}"; do
    if [[ ! "${id_value}" =~ ^[1-9][0-9]{0,9}$ ]] || (( id_value >= 4294967295 )); then
      log "SCENEWORKS_SERVICE_UID and SCENEWORKS_SERVICE_GID must be nonzero numeric Linux IDs"
      return 1
    fi
  done
  command -v setpriv >/dev/null && command -v setfacl >/dev/null || {
    log "RunPod initialization requires setpriv and setfacl from the combined image"
    return 1
  }
  # Keep runtime HOME writable too (CUDA's default compilation cache lives here).
  export HOME="/home/sceneworks"
  for dir in "$@" "${HOME}"; do
    # Refuse symlink roots/ancestors before mkdir or recursive ACL application.
    # Cache-internal HF snapshot symlinks are preserved and never followed (-P).
    if [[ "$(realpath -m -s -- "${dir}")" != "$(realpath -m -- "${dir}")" ]]; then
      log "managed directory '${dir}' must not contain symlink ancestors; choose its explicit physical path"
      return 1
    fi
    if ! mkdir -p -- "${dir}" || ! preflight_runpod_permissions "${dir}"; then
      log "managed directory '${dir}' does not enforce private permissions and named service ACL access; services were not started. Configure enforcing storage or writable per-path overrides."
      return 1
    fi
    physical_dir="$(realpath -m -- "${dir}")" || return 1
    # find -P also crosses nested mounts. Check those boundaries, including bind
    # mounts on the same device, rather than weakening migration with -xdev.
    while IFS=' ' read -r -a mount_fields; do
      target="${mount_fields[4]}"
      printf -v target '%b' "${target//\\/\\0}" || return 1
      case "${target}" in
        "${physical_dir}"/*)
          if ! preflight_runpod_permissions "${target}"; then
            log "nested managed mount '${target}' does not enforce private permissions and named service ACL access; services were not started."
            return 1
          fi
          ;;
      esac
    done < /proc/self/mountinfo
  done
  for dir in "$@" "${HOME}"; do
    if ! grant_service_tree "${dir}"; then
      log "cannot grant service UID ${service_uid} access to '${dir}'; configure the volume export/ACL or select writable per-path overrides. Root services are never a fallback."
      return 1
    fi
    # Provider mount roots and intermediate cache directories may be root-only.
    # Add traversal, not write/list access, only where the service needs it.
    parent="$(dirname "${dir}")"
    while [[ "${parent}" != / ]]; do
      if ! setpriv --reuid="${service_uid}" --regid="${service_gid}" --clear-groups \
        test -x "${parent}"; then
        if ! grant_service_acl "${parent}" --x; then
          log "cannot grant service traversal through '${parent}'; fix the volume export/ACL"
          return 1
        fi
      fi
      parent="$(dirname "${parent}")"
    done
  done
  wait_for_runpod_devices || return 1
  # Device mounts commonly reject ACLs. Join the groups of attached NVIDIA
  # character devices only when primary-identity access is insufficient. Keep
  # provider device owners/modes untouched (including host bind-mounted nodes).
  # Administrative nvidia-caps and debugging/modeset nodes are not required for
  # CUDA inference and may deliberately remain root-only.
  for device in /dev/nvidia[0-9]* /dev/nvidiactl /dev/nvidia-uvm; do
    [[ -c "${device}" && ! -L "${device}" ]] || continue
    if ! setpriv --reuid="${service_uid}" --regid="${service_gid}" --clear-groups \
      bash -c '[[ -r "$1" && -w "$1" ]]' _ "${device}"; then
      device_gid="$(stat -c %g -- "${device}")" || return 1
      case ",${device_groups}," in
        *,"${device_gid}",*) ;;
        *) device_groups="${device_groups:+${device_groups},}${device_gid}" ;;
      esac
    fi
  done
  if [[ -n "${device_groups}" ]]; then
    privilege_group_args=("--groups=${device_groups}")
    log "retaining supplementary NVIDIA device groups: ${device_groups}"
  fi
  for device in /dev/nvidia[0-9]* /dev/nvidiactl /dev/nvidia-uvm; do
    [[ -c "${device}" && ! -L "${device}" ]] || continue
    if ! setpriv --reuid="${service_uid}" --regid="${service_gid}" "${privilege_group_args[@]}" \
      bash -c '[[ -r "$1" && -w "$1" ]]' _ "${device}"; then
      log "NVIDIA device '${device}' must permit service UID ${service_uid} or its device group read/write access"
      return 1
    fi
  done
  log "volume initialization complete; dropping supervisor and services to ${service_uid}:${service_gid}"
}
