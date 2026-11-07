#!/usr/bin/env bash
# Supply the real software AV-mux path required by parity-rust, without letting a slow
# runner-local mirror consume its build budget. No persistent APT sources/settings are edited.
set -euo pipefail

scratch="$(mktemp -d "${TMPDIR:-/tmp}/sceneworks-ci-ffmpeg.XXXXXX")"
apt_used=0
cleanup() {
  if [[ "$apt_used" == 1 ]]; then sudo rm -rf -- "$scratch"; else rm -rf -- "$scratch"; fi
}
trap cleanup EXIT

probe() {
  local version
  version="$(ffmpeg -version 2>/dev/null)" || return 1
  [[ "$version" == 'ffmpeg version '* ]] || return 1
  # Exercise both codecs, the MP4 mux, faststart, and decoding both streams. A version
  # string alone does not establish that the installed binary supports these tests.
  ffmpeg -nostdin -hide_banner -loglevel error -y \
    -f lavfi -i 'color=size=16x16:rate=10' \
    -f lavfi -i 'anullsrc=r=8000:cl=mono' \
    -t 0.2 -c:v libx264 -preset ultrafast -pix_fmt yuv420p -threads 1 \
    -c:a aac -movflags +faststart "$scratch/probe.mp4" || return 1
  ffmpeg -nostdin -hide_banner -loglevel error -i "$scratch/probe.mp4" \
    -map 0:v:0 -map 0:a:0 -f null - || return 1
  printf '%s\n' "${version%%$'\n'*}"
}

if probe; then
  echo 'Using the existing ffmpeg after a real software AV-mux/decode probe.'
  exit 0
fi

# The override is only a fixture seam; the workflow uses the runner's /etc/os-release.
# shellcheck source=/dev/null
source "${CI_FFMPEG_OS_RELEASE_FILE:-/etc/os-release}"
if [[ "${ID:-}" != ubuntu || ! "${VERSION_CODENAME:-}" =~ ^[a-z][a-z0-9]*$ ]] ||
   [[ "$(dpkg --print-architecture)" != amd64 ]]; then
  echo 'A working ffmpeg is required; automatic installation supports Ubuntu amd64 only.' >&2
  exit 1
fi

chmod 755 "$scratch"
mkdir -p "$scratch/lists/partial"
cat > "$scratch/ubuntu.sources" <<EOF
Types: deb
URIs: https://archive.ubuntu.com/ubuntu
Suites: $VERSION_CODENAME $VERSION_CODENAME-updates
Components: main restricted universe multiverse
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg

Types: deb
URIs: https://security.ubuntu.com/ubuntu
Suites: $VERSION_CODENAME-security
Components: main restricted universe multiverse
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg
EOF
apt_options=(
  -o "Dir::Etc::sourcelist=$scratch/ubuntu.sources"
  -o 'Dir::Etc::sourceparts=-'
  -o "Dir::State::lists=$scratch/lists"
  -o 'APT::Update::Error-Mode=any'
  -o 'Acquire::Retries=1'
  -o 'Acquire::http::Timeout=20'
  -o 'Acquire::https::Timeout=20'
  # This job installs packages only; translations and desktop command/app metadata
  # waste the download budget before APT reaches FFmpeg's package indexes.
  -o 'Acquire::Languages=none'
  -o 'Acquire::IndexTargets::deb::CNF::DefaultEnabled=false'
  -o 'Acquire::IndexTargets::deb::DEP-11::DefaultEnabled=false'
  -o 'Acquire::AllowInsecureRepositories=false'
  -o 'APT::Get::AllowUnauthenticated=false'
)
# Transport inactivity timeouts do not bound a continuously slow download. Hard-bound
# metadata/download phases separately, before dpkg runs, so expiry cannot interrupt unpacking.
apt_used=1
sudo timeout --signal=TERM --kill-after=10s 180s apt-get "${apt_options[@]}" update
sudo timeout --signal=TERM --kill-after=10s 180s apt-get "${apt_options[@]}" \
  install -y --no-install-recommends --download-only ffmpeg
sudo apt-get "${apt_options[@]}" install -y --no-install-recommends --no-download ffmpeg
hash -r
probe
