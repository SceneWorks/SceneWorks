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
# WHAT THIS DOES, IN ORDER.
#   1. Census the compute processes on the profiled GPU.
#   2. Kill the ones that are OURS BY NAME — `memory-candle-adapter*` / `memory-mlx-adapter*`. Only
#      those, and only by name: this box is shared with other self-hosted lanes and killing a pid
#      merely because it is inconvenient would take a colleague's build down with it.
#   3. Census again. Anything left is somebody else's, so FAIL FAST with a `::error` naming the pid,
#      the process and its memory — before the walk spends 60 cells learning the same thing.
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

# `pid, process_name, used_memory` for every compute process on the profiled GPU, or empty when
# nvidia-smi cannot be reached. `-i $CUDA_VISIBLE_DEVICES` keeps the census on the ONE device the
# capture profiles; without it a second card's renders would read as contamination here.
census() {
  local device_args=()
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    device_args=(-i "$CUDA_VISIBLE_DEVICES")
  fi
  nvidia-smi "${device_args[@]}" \
    --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true
}

# Indent a census for the job log, dropping the blank line an empty census leaves behind.
indent() {
  while IFS= read -r line; do
    [[ -z "${line//[[:space:]]/}" ]] && continue
    printf '  %s\n' "$line"
  done
}

if ! command -v nvidia-smi >/dev/null 2>&1; then
  # NOT fatal. The walk's own preflight refuses an unset CUDA_VISIBLE_DEVICES and the engine's
  # stable-idle guard still refuses a contaminated peak per anchor; this step is the cheap way to
  # find out early, not the thing that makes the measurement valid.
  echo "::warning title=nvidia-smi unavailable::cannot census the profiled GPU before the walk; a contaminated device will only be caught per-anchor by the engine's stable-idle guard"
  exit 0
fi

# ONE census, printed and then reaped from. Two calls would let the log show a process the reaping
# loop never saw (or the reverse) whenever one exits between them.
before="$(census)"
echo "compute processes on GPU ${CUDA_VISIBLE_DEVICES:-<all>} before the walk:"
indent <<< "$before"

reaped=0
while IFS=, read -r pid name memory; do
  pid="$(echo "$pid" | tr -d '[:space:]')"
  name="$(echo "$name" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
  memory="$(echo "$memory" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
  [[ -z "$pid" ]] && continue
  # The binaries this campaign itself launches, matched on the process-name field `nvidia-smi`
  # prints (a full path on Windows, hence the `*` on both ends). Spelled as literal `case`
  # alternatives rather than held in a variable: bash parses `|` as case syntax BEFORE expanding
  # the pattern, so a variable carrying alternatives would match a literal pipe and reap nothing.
  case "$name" in
    *memory-candle-adapter* | *memory-mlx-adapter*)
      echo "::warning title=Reaping an orphaned adapter::pid ${pid} (${name}, ${memory}) is one of this campaign's own binaries left over from an earlier run; terminating it before the walk"
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
  echo "compute processes still on GPU ${CUDA_VISIBLE_DEVICES:-<all>}:"
  indent <<< "$remaining"
  while IFS= read -r line; do
    [[ -z "$(echo "$line" | tr -d '[:space:]')" ]] && continue
    echo "::error title=Profiled GPU is not idle::${line} is resident on GPU ${CUDA_VISIBLE_DEVICES:-<all>}. Every capture's stable-idle baseline would be contaminated and every anchor would be refused, so the walk is not started. This process is not one of this campaign's binaries -- find out whose it is before killing it."
  done <<< "$remaining"
  exit 1
fi

echo "profiled GPU is idle; starting the walk"
