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

# Opt-in ownership setup never repairs existing network permissions. Only absent
# path components are provisioned; legacy/mixed-owner trees retain the ACL path.
private_owned_image_home() {
  [[ "$1" == /home/sceneworks && -d "$1" && ! -L "$1" &&
     "$(stat -c %u:%g:%a -- "$1")" == 1000:1000:755 &&
     "$(findmnt -n -o TARGET -T "$1")" == / &&
     -z "$(find -P "$1" -mindepth 1 -print -quit)" ]]
}

validate_private_owned_path() (
  set -o pipefail
  local dir="$1" parent="$1" owner mode path resolved root inside image_home=0
  [[ "$(realpath -m -s -- "${dir}")" == "$(realpath -m -- "${dir}")" ]] || return 1
  private_owned_image_home "${dir}" && image_home=1
  while [[ "${parent}" != / ]]; do
    if [[ -e "${parent}" && ! ( "${parent}" == "${dir}" && "${image_home}" == 1 ) ]]; then
      [[ -d "${parent}" && ! -L "${parent}" ]] || return 1
      read -r owner mode < <(stat -c '%u %a' -- "${parent}") || return 1
      [[ "${owner}" == 0 || "${owner}" == "${service_uid}" ]] || return 1
      # Root-owned sticky /tmp protects owned children. Other ancestors must
      # not permit an untrusted identity to rename/replace a managed path.
      if (( (8#${mode} & 0022) != 0 )); then
        [[ "${parent}" == /tmp && "${owner}" == 0 ]] &&
          (( (8#${mode} & 01000) != 0 )) || return 1
      fi
      timeout --signal=KILL 5 setpriv --reuid="${service_uid}" --regid="${service_gid}" \
        --clear-groups --bounding-set=-all --inh-caps=-all --ambient-caps=-all \
        --no-new-privs test -x "${parent}" || return 1
    fi
    parent="$(dirname -- "${parent}")"
  done
  [[ -e "${dir}" && "${image_home}" != 1 ]] || return 0
  find -P "${dir}" -print0 | while IFS= read -r -d '' path; do
    if [[ -L "${path}" ]]; then
      resolved="$(realpath -e -- "${path}")" || return 1
      inside=0
      for root in "${private_owned_roots[@]}"; do
        case "${resolved}" in "${root}"/*) inside=1 ;; esac
      done
      [[ "${inside}" == 1 && -f "${resolved}" ]] || return 1
      path="${resolved}"
    fi
    [[ -d "${path}" || -f "${path}" ]] || return 1
    read -r owner mode < <(stat -c '%u %a' -- "${path}") || return 1
    [[ "${owner}" == "${service_uid}" ]] &&
      (( (8#${mode} & 0077) == 0 && (8#${mode} & 0600) == 0600 )) || return 1
    [[ ! -d "${path}" ]] || (( (8#${mode} & 0100) != 0 )) || return 1
  done
)

preflight_private_owned_permissions() (
  local dir="$1" scratch="" unrelated_uid=65534 unrelated_gid=65534
  [[ "${service_uid}" != "${unrelated_uid}" ]] || unrelated_uid=65533
  [[ "${service_gid}" != "${unrelated_gid}" ]] || unrelated_gid=65533
  cleanup_owned_scratch() {
    [[ -n "${scratch}" ]] || return 0
    # Let the creator remove its children first: an NFS server need not grant
    # namespace root permission to traverse a service-owned private directory.
    timeout --signal=KILL 5 setpriv --reuid="${service_uid}" --regid="${service_gid}" \
      --clear-groups --bounding-set=-all --inh-caps=-all --ambient-caps=-all --no-new-privs \
      bash -c 'rm -rf -- "$1/private/child"' _ "${scratch}" 2>/dev/null || true
    rm -rf -- "${scratch}"
  }
  trap cleanup_owned_scratch EXIT
  scratch="$(mktemp -d "${dir}/.sceneworks-owned-test.XXXXXX")" || return 1
  chmod 0711 -- "${scratch}" || return 1
  ( umask 077; mkdir "${scratch}/private" && printf fixture > "${scratch}/protected" ) || return 1
  chmod 0700 -- "${scratch}/private" && chmod 0600 -- "${scratch}/protected" || return 1
  cd -- "${scratch}" || return 1
  probe_owned_identity() {
    timeout --signal=KILL 5 setpriv --reuid="$1" --regid="$2" --clear-groups \
      --bounding-set=-all --inh-caps=-all --ambient-caps=-all --no-new-privs \
      bash -euc '
        [[ "$(id -u)" == "$1" && "$(id -g)" == "$2" && "$(id -G)" == "$2" ]]
        for field in CapInh CapPrm CapEff CapBnd CapAmb; do
          grep -Eq "^${field}:[[:space:]]+0+$" /proc/self/status
        done
        grep -Eq "^NoNewPrivs:[[:space:]]+1$" /proc/self/status
        case "$3" in
          denied)
            if ( : < protected ) 2>/dev/null; then exit 1; fi
            if ( : >> protected ) 2>/dev/null; then exit 1; fi
            if ( cd private ) 2>/dev/null; then exit 1; fi ;;
          allowed)
            cd "$4"
            [[ "$(< protected)" == fixture ]]
            printf appended >> protected
            umask 077
            mkdir -p private/child/grandchild
            printf created > private/child/grandchild/file
            mv private/child/grandchild/file private/child/grandchild/renamed
            printf reopened >> private/child/grandchild/renamed
            [[ "$(< private/child/grandchild/renamed)" == createdreopened ]]
            [[ "$(stat -c %a private/child/grandchild/renamed)" == 600 ]]
            [[ "$(stat -c %a private/child/grandchild)" == 700 ]]
            # Expose the file boundary from an inherited cwd; a private parent
            # must not conceal a filesystem ignoring the child-file restriction.
            chmod 0711 private/child/grandchild ;;
          inherited)
            if ( : < renamed ) 2>/dev/null; then exit 1; fi
            if ( : >> renamed ) 2>/dev/null; then exit 1; fi
            if ( : > unauthorized ) 2>/dev/null; then exit 1; fi ;;
          *) exit 1 ;;
        esac
      ' _ "$1" "$2" "$3" "${scratch}"
  }
  # Expected-deny stages return success only after identity/cap setup executes.
  probe_owned_identity "${service_uid}" "${service_gid}" denied &&
    chown "${service_uid}:${service_gid}" -- protected private &&
    [[ "$(stat -c %u:%g protected)" == "${service_uid}:${service_gid}" ]] &&
    [[ "$(stat -c %u:%g private)" == "${service_uid}:${service_gid}" ]] &&
    probe_owned_identity "${service_uid}" "${service_gid}" allowed &&
    probe_owned_identity "${unrelated_uid}" "${unrelated_gid}" denied || return 1
  cd private/child/grandchild &&
    probe_owned_identity "${unrelated_uid}" "${unrelated_gid}" inherited || return 1
  cd "${dir}" && cleanup_owned_scratch || return 1
  scratch=""
)

initialize_private_owned_paths() (
  set -o pipefail
  local dir path target index success=0 unrelated_uid=65534 unrelated_gid=65534
  [[ "${service_uid}" != "${unrelated_uid}" ]] || unrelated_uid=65533
  [[ "${service_gid}" != "${unrelated_gid}" ]] || unrelated_gid=65533
  local -a missing created=() mount_fields
  trap 'if [[ "${success}" != 1 ]]; then for ((index=${#created[@]}-1;index>=0;index--)); do rmdir -- "${created[index]}" 2>/dev/null || true; done; fi' EXIT
  # Validate all existing roots before any ownership or mode changes.
  for dir in "${private_owned_roots[@]}"; do
    if ! validate_private_owned_path "${dir}"; then
      log "private-owned path '${dir}' is not an accessible private service-owned tree with safe ancestors. Existing data was not changed; use dedicated path overrides or the default ACL strategy on compatible storage."
      return 1
    fi
  done
  for dir in "${private_owned_roots[@]}"; do
    missing=(); path="${dir}"
    while [[ ! -e "${path}" ]]; do missing+=("${path}"); path="$(dirname -- "${path}")"; done
    for ((index=${#missing[@]}-1;index>=0;index--)); do
      path="${missing[index]}"
      ( umask 077; mkdir -- "${path}" ) || return 1
      created+=("${path}")
      if ! chown "${service_uid}:${service_gid}" -- "${path}" ||
         [[ "$(stat -c %u:%g:%a -- "${path}")" != "${service_uid}:${service_gid}:700" ]]; then
        log "cannot provision private-owned new directory '${path}' with enforced service ownership and mode 0700"
        return 1
      fi
    done
    if private_owned_image_home "${dir}"; then
      chown "${service_uid}:${service_gid}" -- "${dir}" && chmod 0700 -- "${dir}" || return 1
    fi
    if ! validate_private_owned_path "${dir}" || ! preflight_private_owned_permissions "${dir}"; then
      log "private-owned directory '${dir}' failed actual service access or privacy checks"
      return 1
    fi
    # Audit traverses nested mounts, but never follows symlink directories.
    # Probe each filesystem, including same-device bind mounts, independently.
    while IFS=' ' read -r -a mount_fields; do
      target="${mount_fields[4]}"
      printf -v target '%b' "${target//\\/\\0}" || return 1
      case "${target}" in "${dir}"/*)
        preflight_private_owned_permissions "${target}" || return 1 ;; esac
    done < /proc/self/mountinfo
    # Actual existing-file opens do not change content. Pipefail also rejects a
    # failed tree walk rather than treating an empty producer as verified data.
    setpriv --reuid="${service_uid}" --regid="${service_gid}" \
      --clear-groups --bounding-set=-all --inh-caps=-all --ambient-caps=-all --no-new-privs \
      bash -euco pipefail 'find -P "$1" -print0 | while IFS= read -r -d "" path; do
        if [[ -d "$path" ]]; then ( cd "$path" ); else : < "$path"; : >> "$path"; fi
      done' _ "${dir}" || return 1
    find -P "${dir}" -print0 | setpriv \
      --reuid="${unrelated_uid}" --regid="${unrelated_gid}" --clear-groups \
      --bounding-set=-all --inh-caps=-all --ambient-caps=-all --no-new-privs \
      bash -euc '
        [[ "$(id -u)" == "$1" && "$(id -g)" == "$2" && "$(id -G)" == "$2" ]]
        for field in CapInh CapPrm CapEff CapBnd CapAmb; do
          grep -Eq "^${field}:[[:space:]]+0+$" /proc/self/status
        done
        grep -Eq "^NoNewPrivs:[[:space:]]+1$" /proc/self/status
        while IFS= read -r -d "" path; do
          if ( : < "$path" ) 2>/dev/null; then exit 1; fi
          if ( : >> "$path" ) 2>/dev/null; then exit 1; fi
          if ( cd "$path" ) 2>/dev/null; then exit 1; fi
        done' _ "${unrelated_uid}" "${unrelated_gid}" || return 1
  done
  success=1
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
  local strategy="${SCENEWORKS_PERMISSION_STRATEGY:-acl}"
  case "${strategy}" in acl|private-owned) ;; *)
    log "SCENEWORKS_PERMISSION_STRATEGY must be acl or private-owned"
    return 1 ;; esac
  command -v setpriv >/dev/null || return 1
  if [[ "${strategy}" == acl ]]; then
  command -v setfacl >/dev/null || {
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
  else
    export HOME=/home/sceneworks
    local -a private_owned_roots=("$@" "${HOME}")
    initialize_private_owned_paths || {
      log "private-owned permission initialization failed; services were not started. No root service fallback."
      return 1
    }
    umask 077
  fi
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
