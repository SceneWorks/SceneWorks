#!/usr/bin/env bash
# SW-7202: run the describer test under its wrapper chain and record the guard's verdict.
#
#     guarded_run.sh <evidence dir> <test log> <test name> -- <wrapper command...>
#
# The wrapper command ends in tree_guard.py, which writes the test's output to <test log> and
# prints to stdout a `GUARD: ...` line only when it stops the tree, and always one
# `exit=... root_lifetime_max_gib=...` summary line. Both kinds of line go to <evidence dir>/guard.txt
# and the step summary; the wrapper's stderr is kept in <evidence dir>/wrapper-stderr.txt.
# Exits 0 only when the command exits 0, the log shows exactly one passing test, and the guard
# printed its summary line; otherwise it names which of those failed.
set -euo pipefail

if [[ $# -lt 5 || "$4" != "--" ]]; then
  echo "usage: guarded_run.sh <evidence dir> <test log> <test name> -- <command...>" >&2
  exit 2
fi
out="$1" log="$2" test_name="$3"
shift 4

set +e
guard="$("$@" 2> >(tee "$out/wrapper-stderr.txt" >&2))"
rc=$?
set -e
# Let the stderr tee finish writing before anything reads the file.
wait
[[ -f "$log" ]] && cat "$log"
lines="$(grep -E '^(GUARD:|exit=)' <<<"$guard" || true)"
printf '%s\n' "$lines" | sed '/^$/d' | tee "$out/guard.txt"

failed=()
[[ "$rc" == 0 ]] || failed+=("the wrapper chain exited $rc")
[[ -f "$log" ]] && grep -q "test result: ok. 1 passed" "$log" || failed+=("the log has no 'test result: ok. 1 passed'")
grep -q '^exit=' <<<"$lines" || failed+=("tree_guard printed no 'exit=' summary line, so no peak was recorded")
if (( ${#failed[@]} )); then
  printf '::error::%s: %s\n' "$test_name" "$(IFS=';'; echo "${failed[*]}")"
  echo "wrapper stdout (non-guard lines):"
  grep -vE '^(GUARD:|exit=)' <<<"$guard" || true
  echo "wrapper stderr ($out/wrapper-stderr.txt):"
  cat "$out/wrapper-stderr.txt" 2>/dev/null || true
  exit 1
fi
{
  echo
  sed 's/^/    /' <<<"$lines"
  echo
  echo "The MLX floor is ceil(root_lifetime_max GiB / 0.9)."
} >> "${GITHUB_STEP_SUMMARY:-/dev/null}"
