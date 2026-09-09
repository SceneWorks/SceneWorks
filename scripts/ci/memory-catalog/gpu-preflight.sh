#!/usr/bin/env bash
# Refuse to start a candle walk on a contaminated GPU, and reap our OWN orphans first (sc-22738).
#
# THE INCIDENT. Run 34272596969 spent 51 guarded renders — roughly an hour of GPU time — producing
# 51 copies of one sentence: "pure compute processes [6308] are resident on the profiled GPU; the
# peak is contaminated". Pid 6308 was a `memory-candle-adapter.exe` orphaned when the PREVIOUS
# dispatch (34271044903) was cancelled. The runner's own "Cleaning up orphan processes" reaps the
# step's direct children; the harness launches the adapter DETACHED (deliberately — so a Ctrl-C
# cannot reach it mid-command-buffer), so a cancelled walk can leave one holding VRAM.
#
# THE FALSE POSITIVE THAT FOLLOWED. The first spelling of this script censused with
# `nvidia-smi --query-compute-apps`, which on Windows WDDM returns EVERY process holding a device
# context — `explorer.exe`, `WindowsTerminal.exe`, `StartMenuExperienceHost.exe`, three pids whose
# names read `[Insufficient Permissions]` — all with `[N/A]` memory. Run 34297841666 refused twenty
# desktop graphics contexts as "foreign compute processes" and never started the walk, while the
# engine's own guard on the very same host had flagged exactly ONE pid. The engine discriminates by
# PROCESS TYPE, not by presence: `candle-gen/src/testkit.rs::pure_compute_pids` reads
# `nvidia-smi pmon -i <gpu> -c 1 -s um` and refuses only rows of type `C`, explicitly passing over
# `G` and `C+G` because "WDDM desktop processes are reported as C+G even with zero SM/memory
# activity". This script now asks the SAME question, with the SAME command, so a preflight refusal
# and an engine refusal cannot disagree.
#
# WHAT THIS DOES, IN ORDER.
#   1. Census the processes on the profiled GPU by type — `pmon` for the type,
#      `--query-compute-apps` for the full process path and memory that `pmon` truncates — and
#      print the whole thing, graphics rows included, so a refusal is diagnosable from the log.
#   2. Kill the ones that are OURS BY NAME — `memory-candle-adapter*` / `memory-mlx-adapter*`. Only
#      those, and only by name: this box is shared with other self-hosted lanes and killing a pid
#      merely because it is inconvenient would take a colleague's build down with it.
#   3. Census again. Any PURE-COMPUTE (`C`) row left is somebody else's, so FAIL FAST with a
#      `::error` naming the pid, the process and its memory — before the walk spends 60 cells
#      learning the same thing. Graphics and `C+G` desktop contexts are reported and ignored.
#
# WHY THIS IS NOT A GATE. It gates no code, no merge and no measurement: it is the same claim the
# engine's stable-idle guard makes (a peak sampled beside a foreign process is not this model's
# peak), made once at the top instead of once per anchor. Read `FEATURE_DEVELOPMENT.md`'s "Gate
# teardown" before reading anything else into it.
#
# WHY IT RUNS UNDER GIT BASH BUT SHELLS OUT TO POWERSHELL TO KILL. `nvidia-smi` is a plain console
# executable and reads fine from bash. Terminating a Windows process is not: Git Bash's `kill`
# addresses MSYS pids, not Windows ones, so it cannot signal a native process the runner never
# started. `Stop-Process` is the spelling that works, and it is invoked per pid, by pid, only after
# THIS script has matched the process by name.
# shellcheck source=scripts/ci/memory-catalog/common.sh
source "$(dirname "$0")/common.sh"

# The one device the capture profiles. Without `-i` a second card's renders would read as
# contamination here.
device_args=()
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  device_args=(-i "$CUDA_VISIBLE_DEVICES")
fi
# Only a single ordinal can be matched against `pmon`'s GPU column; a list (or an unset value)
# leaves the cross-device check to `-i` alone.
expected_gpu=""
if [[ "${CUDA_VISIBLE_DEVICES:-}" =~ ^[0-9]+$ ]]; then
  expected_gpu="$CUDA_VISIBLE_DEVICES"
fi

is_windows() {
  case "$(uname -s 2>/dev/null || echo unknown)" in
    MINGW* | MSYS* | CYGWIN* | Windows_NT) return 0 ;;
    *) return 1 ;;
  esac
}

# `pid, process_name, used_memory` for every process nvidia-smi will name. On Linux this is
# compute-only; on Windows WDDM it is every GPU-attached process, which is why it supplies NAMES
# here and never the classification.
apps_census() {
  nvidia-smi "${device_args[@]}" \
    --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true
}

# `pid<TAB>type<TAB>memory` per process, from the same `pmon` invocation the engine's stable-idle
# guard uses. Column positions come from pmon's own `# gpu pid type fb ...` header rather than
# being assumed: driver versions differ on whether `ccpm` sits between `fb` and `sm`.
pmon_census() {
  nvidia-smi pmon "${device_args[@]}" -c 1 -s um 2>/dev/null |
    awk -v want="$expected_gpu" '
      /^#/ {
        if ($2 == "gpu") { for (i = 2; i <= NF; i++) { col[$i] = i - 1 } }
        next
      }
      {
        gi = ("gpu" in col) ? col["gpu"] : 1
        pi = ("pid" in col) ? col["pid"] : 2
        ti = ("type" in col) ? col["type"] : 3
        fi = ("fb" in col) ? col["fb"] : 0
        if ($pi !~ /^[0-9]+$/) { next }
        if (want != "" && $gi != want) {
          printf "MISMATCH\t%s\t%s\n", $gi, $pi
          next
        }
        mem = (fi > 0 && fi <= NF && $fi ~ /^[0-9]+$/) ? ($fi " MiB") : "[N/A]"
        printf "%s\t%s\t%s\n", $pi, $ti, mem
      }
    ' || true
}

# Full name and memory for a pid, from the apps census; empty when nvidia-smi would not name it.
apps_row() {
  local pid="$1" apps="$2"
  printf '%s\n' "$apps" | awk -F',' -v p="$pid" '
    {
      gsub(/^[ \t]+|[ \t]+$/, "", $1)
      if ($1 != p) { next }
      name = $2; mem = $3
      gsub(/^[ \t]+|[ \t]+$/, "", name)
      gsub(/^[ \t]+|[ \t]+$/, "", mem)
      printf "%s\t%s\n", name, mem
      exit
    }
  '
}

# One census: `pid<TAB>type<TAB>name<TAB>memory` per row, types straight from pmon and names from
# the apps census wherever it has one (pmon truncates `command` to a bare, clipped basename).
census() {
  local apps pmon pid type mem name enriched
  apps="$(apps_census)"
  pmon="$(pmon_census)"
  while IFS=$'\t' read -r pid type mem; do
    [[ -z "$pid" ]] && continue
    if [[ "$pid" == "MISMATCH" ]]; then
      echo "::warning title=Census returned another GPU::nvidia-smi pmon reported physical GPU ${type} (pid ${mem}) for a census scoped to GPU ${expected_gpu}; that row is ignored" >&2
      continue
    fi
    name="[unnamed]"
    enriched="$(apps_row "$pid" "$apps")"
    if [[ -n "$enriched" ]]; then
      name="${enriched%%$'\t'*}"
      # The apps census is authoritative on memory too, except where it has none to give.
      local apps_mem="${enriched#*$'\t'}"
      [[ -n "$apps_mem" && "$apps_mem" != "[N/A]" ]] && mem="$apps_mem"
    fi
    printf '%s\t%s\t%s\t%s\n' "$pid" "$type" "$name" "$mem"
  done <<< "$pmon"
}

# The census as a job-log block, dropping the blank line an empty census leaves behind.
report() {
  local pid type name mem
  while IFS=$'\t' read -r pid type name mem; do
    [[ -z "$pid" ]] && continue
    printf '  pid %s, type %s, %s, %s\n' "$pid" "$type" "$name" "$mem"
  done
}

if ! command -v nvidia-smi >/dev/null 2>&1; then
  # NOT fatal. The walk's own preflight refuses an unset CUDA_VISIBLE_DEVICES and the engine's
  # stable-idle guard still refuses a contaminated peak per anchor; this step is the cheap way to
  # find out early, not the thing that makes the measurement valid.
  echo "::warning title=nvidia-smi unavailable::cannot census the profiled GPU before the walk; a contaminated device will only be caught per-anchor by the engine's stable-idle guard"
  exit 0
fi

if ! nvidia-smi pmon "${device_args[@]}" -c 1 -s um >/dev/null 2>&1; then
  # Without pmon there is no type column, and without a type column every desktop graphics context
  # reads as compute — the exact false positive that stopped run 34297841666. Refusing to guess is
  # the only honest option: the engine's per-anchor guard still holds the line.
  echo "::warning title=nvidia-smi pmon unavailable::cannot classify GPU processes by type before the walk; a contaminated device will only be caught per-anchor by the engine's stable-idle guard"
  exit 0
fi

# ONE census, printed and then reaped from. Two calls would let the log show a process the reaping
# loop never saw (or the reverse) whenever one exits between them.
before="$(census)"
echo "processes on GPU ${CUDA_VISIBLE_DEVICES:-<all>} before the walk (type C is compute; C+G and G are desktop contexts the engine ignores):"
report <<< "$before"

reaped=0
while IFS=$'\t' read -r pid type name memory; do
  [[ -z "$pid" ]] && continue
  # The binaries this campaign itself launches, matched on the process-name field `nvidia-smi`
  # prints (a full path on Windows, hence the `*` on both ends). Spelled as literal `case`
  # alternatives rather than held in a variable: bash parses `|` as case syntax BEFORE expanding
  # the pattern, so a variable carrying alternatives would match a literal pipe and reap nothing.
  case "$name" in
    *memory-candle-adapter* | *memory-mlx-adapter*)
      echo "::warning title=Reaping an orphaned adapter::pid ${pid} (type ${type}, ${name}, ${memory}) is one of this campaign's own binaries left over from an earlier run; terminating it before the walk"
      if command -v powershell >/dev/null 2>&1; then
        powershell -NoProfile -Command "Stop-Process -Id ${pid} -Force -ErrorAction SilentlyContinue" || true
      else
        kill -9 "$pid" 2>/dev/null || true
      fi
      reaped=$(( reaped + 1 ))
      ;;
    *)
      ;;
  esac
done <<< "$before"

if (( reaped > 0 )); then
  # Stop-Process returns before the driver has released the context; give it a moment so the
  # re-census does not report a process that is already on its way out.
  sleep 10
  echo "reaped ${reaped} orphaned adapter process(es)"
fi

remaining="$(census)"
if [[ -n "$(echo "$remaining" | tr -d '[:space:]')" ]]; then
  echo "processes still on GPU ${CUDA_VISIBLE_DEVICES:-<all>}:"
  report <<< "$remaining"
fi

# Only what the engine would refuse. `pure_compute_pids` counts type `C`, passes over `C+G` and
# `G`, and errors on anything else; an unrecognised type is warned about here rather than refused,
# because a census oddity is not evidence of a busy GPU and stopping the walk on one would be the
# false positive this script exists to stop making.
foreign=0
while IFS=$'\t' read -r pid type name memory; do
  [[ -z "$pid" ]] && continue
  case "$type" in
    C)
      case "$name" in
        *memory-candle-adapter* | *memory-mlx-adapter*)
          echo "::error title=Our own adapter survived the reap::pid ${pid} (${name}, ${memory}) is one of this campaign's binaries and is STILL resident after the reap; the walk is not started."
          ;;
        *)
          echo "::error title=Profiled GPU is not idle::pid ${pid} (${name}, ${memory}) is a pure-compute process on GPU ${CUDA_VISIBLE_DEVICES:-<all>}. Every capture's stable-idle baseline would be contaminated and every anchor would be refused, so the walk is not started. This process is not one of this campaign's binaries -- find out whose it is before killing it."
          ;;
      esac
      foreign=$(( foreign + 1 ))
      ;;
    "C+G" | G)
      # A WDDM desktop context. It holds a device context and no compute work; the engine's stable
      # baseline absorbs its fixed residency.
      ;;
    *)
      echo "::warning title=Unrecognised GPU process type::pid ${pid} (${name}, ${memory}) has nvidia-smi process type '${type}', which the engine's stable-idle guard does not recognise; it is not counted as contamination here"
      ;;
  esac
done <<< "$remaining"

if (( foreign > 0 )); then
  exit 1
fi

echo "profiled GPU carries no foreign compute process; starting the walk"
