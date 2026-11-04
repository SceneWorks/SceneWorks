#!/usr/bin/env bash
# Match macos-mlx.yml: validate the provisioned interpreter; never install into /Users/runner.
set -euo pipefail
py="$(command -v python3.12)"
"$py" -I -B -c 'import platform, sys; assert sys.implementation.name == "cpython" and sys.version_info[:2] == (3, 12) and sys.platform == "darwin" and platform.machine() == "arm64", "native arm64 Mac CPython 3.12 required"'
: "${RUNNER_TEMP:?}" "${GITHUB_JOB:?}" "${GITHUB_RUN_ID:?}" "${GITHUB_RUN_ATTEMPT:?}" "${GITHUB_ENV:?}" "${GITHUB_PATH:?}"
shim="$RUNNER_TEMP/cv-python-$GITHUB_JOB-$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT"
mkdir "$shim"
ln -s "$py" "$shim/python3"
ln -s "$py" "$shim/python"
printf '%s\n' "$shim" >> "$GITHUB_PATH"
printf 'CV_NATIVE_PYTHON=%s\n' "$py" >> "$GITHUB_ENV"
