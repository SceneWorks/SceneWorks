# sc-3767 — Bundled ffmpeg: provenance & licensing

Epic 3482 (Python Eradication). The Rust worker shells out to `ffmpeg` for frame
sampling (person-track), frame extraction, timeline export, and video-gen audio
mux — all through `media_jobs::run_ffmpeg`, which uses `SCENEWORKS_FFMPEG` when set
else `ffmpeg` on PATH. The desktop ships no system ffmpeg and used to borrow the
venv's `imageio-ffmpeg` binary; once the Mac Python venv is stripped (sc-3492) that
source disappears, so we bundle a static ffmpeg as an app resource instead — the
same pattern as the onnxruntime dylib (sc-3487).

## What we ship

- **Binary:** `ffmpeg` 7.1, macОS arm64 (Mach-O, links only system frameworks +
  `libc++`/`libSystem` — no bundled third-party dylibs).
- **Source:** the binary bundled inside the `imageio-ffmpeg==0.6.0` macOS arm64
  wheel (`imageio_ffmpeg/binaries/ffmpeg-macos-aarch64-v7.1`), staged by
  `apps/desktop/scripts/stage-ffmpeg.py` (invoked from `build-sidecar.mjs` on macOS
  only) into the `apps/desktop/ffmpeg/` Tauri resource dir.
- **Identity:** this is the **exact** binary the desktop already runs today via the
  venv — sha256 `6d175a4743ca50256e89a8cdd731100f9cee33bd79aeea46894d209410dc6617`
  (wheel-RECORD base64 `bRdaR0PKUCVuiajN1zEQD5zuM715rupGiU0glBDcZhc=`). Bundling
  changes only *where* it resolves from, not *what* runs — zero behavior change.
- **Upstream provenance:** imageio-ffmpeg's macOS builds come from
  [evermeet.cx](https://evermeet.cx/ffmpeg/). Build configuration (from
  `ffmpeg -version`):
  `--enable-gpl --enable-libx264 --enable-libx265 --enable-libvmaf --enable-libvidstab
  --enable-libaom --enable-libsvtav1 --enable-libopus --enable-libmp3lame --enable-libvpx
  --enable-libwebp --enable-libass --enable-libfreetype ...`

## Licensing — GPLv3

This is a **GPLv3** build, not LGPL: `--enable-gpl` plus GPL-licensed components
(libx264, libx265, libvidstab) place the combined ffmpeg binary under GPLv3.

**An LGPL build is not an option for SceneWorks:** the video-gen encode path
hard-requires `libx264` for H.264 mp4 output (`crates/sceneworks-worker/src/
video_jobs.rs` — `-c:v libx264`), which only exists in a GPL build. Dropping to
LGPL would break video generation and timeline export.

SceneWorks invokes ffmpeg as a **separate executable** over a command-line
interface (`std::process::Command`), at arm's length — the GPL'd ffmpeg is an
independent aggregated program, not linked into the app, so its copyleft does not
extend to SceneWorks' own code.

### Redistribution obligations (now that we bundle it)

Today the GPL binary is pip-downloaded onto the user's machine at first-run; once
it's an app resource, **we redistribute it**, which triggers GPLv3 §6:

- **License text:** ship the GPL v3 license alongside the binary (ffmpeg's own
  `COPYING.GPLv3`).
- **Corresponding source / written offer:** the exact source is FFmpeg 7.1 from
  <https://ffmpeg.org/releases/> (`ffmpeg-7.1.tar.xz`) built per the evermeet
  configuration above; evermeet publishes its build scripts at
  <https://evermeet.cx/ffmpeg/>. Provide this as a written offer in the app's
  third-party notices / About → Licenses.
- The `imageio-ffmpeg` *wrapper* (the Python package the wheel comes from) is
  BSD-2-Clause — relevant only as the delivery vehicle; it imposes no extra
  obligation on the binary.

> **Action for packaging (tracked, not yet wired):** the desktop's third-party
> license bundle / About screen must include the ffmpeg GPLv3 text + the written
> offer above before the bundled-ffmpeg build ships to users. This doc is the
> source of truth for that notice.

## Cross-platform

- **macOS desktop:** bundled static ffmpeg (this doc).
- **Windows / Linux desktop:** `build-sidecar.mjs` writes a placeholder into the
  `ffmpeg/` resource dir (Tauri errors on a resource glob matching no files); no
  binary is staged. `resolve_bundled_ffmpeg()` finds no resource and no
  imageio-ffmpeg → returns `None` → `SCENEWORKS_FFMPEG` stays unset → PATH ffmpeg.
- **Server / Docker:** never run `setup.rs`; always PATH ffmpeg. Unchanged.

## Resolution order (`apps/desktop/src/setup.rs::resolve_bundled_ffmpeg`)

1. Bundled resource `…/Resources/ffmpeg/ffmpeg` (packaged app) — **preferred**.
2. Venv `imageio-ffmpeg` binary (dev / pre-bundle fallback).
3. Neither → `None` → caller leaves `SCENEWORKS_FFMPEG` unset → worker uses PATH.

The `SCENEWORKS_FFMPEG` override contract is unchanged, so no Rust-worker change is
needed — `media_jobs::run_ffmpeg` already honors it.
