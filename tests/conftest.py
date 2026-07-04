from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


# The retired Python worker's test suite (16 `tests/test_worker_*.py` /
# `test_*_adapters.py` files + `worker_runtime_shared.py`) was deleted in sc-8863
# once its live gates had been re-expressed against the Rust worker (sc-8861 e2e
# pure-HTTP client + sc-9513 engine-wiring guards), so `apps/worker` is no longer
# imported by any test. The surviving tests (Rust-API contract/smoke, manifest
# audits, the convert-script unit test) only need `packages/shared` on the path.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "shared"))


def running_in_ci() -> bool:
    """GitHub Actions (and most CI providers) set ``CI=true``."""
    return os.getenv("CI", "").strip().lower() in {"1", "true", "yes"}


# The CI gate steps this guard protects run a single marker at a time
# (`-m e2e`, `-m parity`). We only enforce "must not be all-skipped" for those,
# so the light-dep default lane and local ad-hoc filtered runs are unaffected.
_GUARDED_MARKERS = ("e2e", "parity")


def _guarded_marker(config: pytest.Config) -> str | None:
    markexpr = str(getattr(config.option, "markexpr", "") or "")
    for marker in _GUARDED_MARKERS:
        # Match the CI invocation `-m e2e` / `-m parity` (not `not e2e`).
        if markexpr.strip() == marker:
            return marker
    return None


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Fail an e2e/parity CI gate that exercised nothing.

    pytest exits 0 when every collected test skips, so a future workflow change
    that dropped the required toolchain (cargo, ffmpeg, ...) would turn one of
    these gates into a silent no-op that still reports success. In CI, require
    at least one collected test in a guarded gate to actually run (sc-8935 /
    F-133). Individual fixtures also upgrade a missing-cargo skip to a failure
    via `require_tool`, so this is defense in depth for any other skip source.
    """
    if not running_in_ci():
        return
    marker = _guarded_marker(session.config)
    if marker is None:
        return
    # Nothing collected (e.g. no test file matched) is a separate signal handled
    # by pytest's own exit code 5; only guard the all-skipped case here.
    collected = getattr(session, "testscollected", 0)
    if collected == 0:
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    ran = 0
    if reporter is not None:
        ran = len(reporter.stats.get("passed", [])) + len(reporter.stats.get("failed", []))
        ran += len(reporter.stats.get("error", []))
    if ran == 0:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        if reporter is not None:
            reporter.write_line(
                f"ERROR: the `-m {marker}` CI gate collected {collected} test(s) but ran none "
                "(all skipped) -- the required toolchain is missing, so this gate exercised "
                "nothing. Failing instead of reporting a silent green (sc-8935 / F-133).",
                red=True,
            )


def require_tool(name: str, purpose: str) -> None:
    """Guard a test on an external tool: fail in CI (where the toolchain is
    provisioned by earlier workflow steps, so a miss means the workflow broke),
    skip locally (so a developer without the tool isn't blocked). This keeps a
    dropped CI toolchain from silently no-op'ing the e2e gate (sc-8935 / F-133)."""
    import shutil

    if shutil.which(name) is not None:
        return
    message = f"{name} is required for {purpose}"
    if running_in_ci():
        pytest.fail(f"{message} but is missing in CI -- the workflow toolchain regressed.")
    pytest.skip(message)
