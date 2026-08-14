@echo off
rem =============================================================================
rem use-real-rust-toolchain.cmd  (sc-13166)
rem
rem Prepend the REAL rustup toolchain bin to PATH so local Windows builds resolve
rem cargo/rustc/clippy directly, BYPASSING the fragile %USERPROFILE%\.cargo\bin
rem rustup proxies. On the self-hosted CUDA box those 13 proxies were observed as
rem 0-byte symlinks to rustup.exe (rustup prefers symlinks but falls back to
rem hardlinks where symlinks are unavailable, so that shape is not guaranteed);
rem when rustup.exe itself goes missing the proxies resolve to nothing and `cargo`
rem dies with
rem   "failed to run `cargo metadata` ... The system cannot find the file
rem    specified. (os error 2)"
rem
rem WHO DELETES rustup.exe -- corrected. This comment used to blame Defender; that
rem was wrong, and so was the self-update story that replaced it. It is
rem `Swatinem/rust-cache`'s POST step (inference PR #467). Not a Windows problem
rem and not an antivirus problem -- it was diagnosed on a Mac.
rem CANONICAL WRITEUP + the invariant that now enforces it (run by `npm run check`):
rem   scripts/lib/rust-cache-bin.mjs
rem Read that before re-deriving any of this; it is deliberately not filed under a
rem Windows-only path, because the bug is not Windows-specific.
rem
rem CI already dodges this via .github/actions/prepare-rust-runner (it prepends
rem the real toolchain bin to $GITHUB_PATH); this is the local-build equivalent.
rem Both remain worth keeping: they bypass %USERPROFILE%\.cargo\bin entirely, so
rem they are immune no matter what prunes that directory.
rem
rem CALL it (do NOT run in a child shell) so the PATH change reaches the caller:
rem     call "%~dp0scripts\win\use-real-rust-toolchain.cmd"
rem
rem It reads the pinned channel from rust-toolchain.toml at the repo root
rem (two levels up from this script), defaulting to "stable", and NEVER invokes
rem rustup (rustup.exe is itself one of the broken proxies). Sets ERRORLEVEL 1 if
rem no real toolchain is found. Leaves SW_RUST_TC_BIN set to the chosen bin dir.
rem =============================================================================
set "SW_REPO_ROOT=%~dp0..\..\"
set "SW_RUST_CHANNEL=stable"
if exist "%SW_REPO_ROOT%rust-toolchain.toml" (
  for /f "tokens=2 delims== " %%c in ('findstr /r /c:"^ *channel *=" "%SW_REPO_ROOT%rust-toolchain.toml"') do set "SW_RUST_CHANNEL=%%~c"
)
set "SW_RUST_TC_BIN=%USERPROFILE%\.rustup\toolchains\%SW_RUST_CHANNEL%-x86_64-pc-windows-msvc\bin"
if not exist "%SW_RUST_TC_BIN%\cargo.exe" (
  rem channel parse missed or toolchain dir named differently -- fall back to any
  rem installed toolchain that has a real (non-proxy) cargo.exe.
  for /d %%d in ("%USERPROFILE%\.rustup\toolchains\*") do if exist "%%d\bin\cargo.exe" set "SW_RUST_TC_BIN=%%d\bin"
)
if not exist "%SW_RUST_TC_BIN%\cargo.exe" (
  echo [rust] ERROR: no real Rust toolchain under "%USERPROFILE%\.rustup\toolchains" ^(sc-13166^).
  echo         Reinstall it:  rustup toolchain install %SW_RUST_CHANNEL% --force
  exit /b 1
)
set "PATH=%SW_RUST_TC_BIN%;%PATH%"
echo [rust] real toolchain bin on PATH, bypassing .cargo\bin ^(sc-13166^): %SW_RUST_TC_BIN%
exit /b 0
