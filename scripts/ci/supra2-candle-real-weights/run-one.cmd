@echo off
rem Epic 24212 (sc-24217): run ONE #[ignore]d sceneworks-worker lib test by its exact path on the
rem Windows real-weights card and require that EXACTLY one test ran and passed, so a renamed or
rem filtered-out test can never read green. Called from windows-candle.yml `supra2-worker-smoke`
rem inside the vcvars64 environment, after the lib test binary is already built.
rem   %1  the test path (module::path::test_name)
rem   %2  the log file the test output is written to (and echoed)
cargo test -p sceneworks-worker --features backend-candle --release --lib %1 -- --ignored --exact --nocapture --test-threads=1 > "%~2" 2>&1
set rc=%errorlevel%
type "%~2"
if not "%rc%"=="0" (
  echo ::error::%1 failed with exit code %rc%
  exit /b 1
)
findstr /C:"test result: ok. 1 passed" "%~2" >nul
if errorlevel 1 (
  echo ::error::%1 did not run exactly one passing test
  exit /b 1
)
exit /b 0
