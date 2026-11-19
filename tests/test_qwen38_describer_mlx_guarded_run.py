"""SW-7202: the Qwen3.8 describer MLX lane's guarded run (scripts/ci/qwen38-describer-mlx/guarded_run.sh)
records tree_guard's peak summary and names why it failed. A fake test process stands in for the
describer binary: no model, no weights, no GPU. macOS-only, like tree_guard (`proc_pid_rusage`).
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GUARDED_RUN = ROOT / "scripts" / "ci" / "qwen38-describer-mlx" / "guarded_run.sh"
TREE_GUARD = ROOT / "scripts" / "ci" / "iris-mlx-real-weights" / "tree_guard.py"
LOCK = ROOT / "scripts" / "ci" / "metal_execution_lock.py"
TEST_NAME = "qwen38_describer_gpu_smoke::qwen38_describer_describes_a_synthetic_scene_on_mlx"
PASSED = "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 3691 filtered out"

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="macOS-only lane tooling")

# The runner's `shell: bash` may be macOS /bin/bash 3.2 or a newer bash on PATH: run under both.
BASHES = sorted({"/bin/bash", shutil.which("bash") or "/bin/bash"})
BASH = BASHES[0]
BASH_PATH = "/usr/bin:/bin"


@pytest.fixture(autouse=True, params=BASHES)
def _bash(request):
    global BASH
    BASH = request.param

# The metal_execution_lock.py wrapper on a private lock path (never the host's canonical lease):
# acquire, print the lease record to stderr, run the command through run_owned, exit with its code.
LOCK_SHIM = textwrap.dedent(
    """
    import importlib.util, json, sys
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("lock_shim", sys.argv[1])
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m)
    # tree_guard re-validates an inherited lease against the CANONICAL path; this private lease
    # must not be handed down under that variable, so the guard runs as an unleased supervisor.
    m.INHERITED_FD = "SCENEWORKS_TEST_PRIVATE_LEASE_FD"
    with m._acquire(Path(sys.argv[2]), 5) as lease:
        print("METAL_EXECUTION_LEASE " + json.dumps(lease.record()), file=sys.stderr, flush=True)
        sys.exit(m.run_owned(sys.argv[4:], lease))
    """
)


def fake_test(body: str) -> list[str]:
    return [sys.executable, "-c", textwrap.dedent(body)]


def run(tmp_path: Path, child: list[str], cap_gib: str = "4", wall_s: str = "60", lock: bool = True):
    out = tmp_path / "evidence"
    out.mkdir()
    log = out / "describer.log"
    summary = tmp_path / "step-summary.md"
    chain = [sys.executable, str(TREE_GUARD), cap_gib, wall_s, str(log), "--", *child]
    if lock:
        chain = [sys.executable, "-I", "-c", LOCK_SHIM, str(LOCK), str(tmp_path / "metal.lock"), "--", *chain]
    done = subprocess.run(
        [BASH, str(GUARDED_RUN), str(out), str(log), TEST_NAME, "--", *chain],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env={"PATH": BASH_PATH, "GITHUB_STEP_SUMMARY": str(summary)},
    )
    guard = (out / "guard.txt").read_text() if (out / "guard.txt").exists() else ""
    return done, out, guard, summary.read_text() if summary.exists() else ""


def test_a_passing_test_under_the_lock_and_guard_records_the_peak(tmp_path):
    # The run that failed on nax-macos (44464702617): the test passed, the guard never stopped it,
    # so the guard printed no `GUARD:` line, only its `exit=` summary.
    done, out, guard, summary = run(tmp_path, fake_test(f"""
        held = bytearray(64 << 20)
        for i in range(0, len(held), 4096): held[i] = 1
        print("running 1 test"); print({PASSED!r})
    """))
    assert done.returncode == 0, done.stdout + done.stderr
    assert PASSED in done.stdout, "the test log is echoed into the job log"
    lines = guard.splitlines()
    assert len(lines) == 1 and lines[0].startswith("exit=0 "), guard
    fields = dict(re.findall(r"(\w+)=(\S+)", lines[0]))
    assert int(fields["root_lifetime_max_bytes"]) >= 64 << 20, fields
    assert "root_lifetime_max_gib" in fields and "stopped" not in fields
    assert lines[0] in summary and "ceil(root_lifetime_max GiB / 0.9)" in summary
    assert "METAL_EXECUTION_LEASE " in (out / "wrapper-stderr.txt").read_text()


def test_a_nonzero_exit_names_the_code_and_shows_the_wrapper_stderr(tmp_path):
    done, out, guard, summary = run(tmp_path, fake_test(f"""
        import sys
        print({PASSED!r}); sys.exit(3)
    """))
    assert done.returncode == 1
    assert f"::error::{TEST_NAME}: the wrapper chain exited 3" in done.stdout, done.stdout
    assert "METAL_EXECUTION_LEASE " in done.stdout, "the failure prints the wrapper's stderr"
    assert guard.startswith("exit=3 "), "the peak is still recorded on a failure"
    assert summary == ""


def test_a_zero_exit_without_a_passing_test_is_a_named_failure(tmp_path):
    done, _, guard, _ = run(tmp_path, fake_test("print('test result: ok. 0 passed')"))
    assert done.returncode == 1
    assert "the log has no 'test result: ok. 1 passed'" in done.stdout, done.stdout
    assert guard.startswith("exit=0 ")


def test_a_cap_stop_keeps_both_guard_lines(tmp_path):
    done, _, guard, _ = run(tmp_path, fake_test("""
        import time
        held = bytearray(256 << 20)
        for i in range(0, len(held), 4096): held[i] = 1
        time.sleep(60)
    """), cap_gib="0.125")
    assert done.returncode == 1
    lines = guard.splitlines()
    assert lines[0].startswith("GUARD: tree footprint") and lines[0].endswith("SIGTERM"), guard
    assert lines[1].startswith("exit=") and 'stopped="tree footprint' in lines[1], guard
    assert "the wrapper chain exited" in done.stdout


def test_a_wrapper_that_never_reaches_the_guard_says_no_peak_was_recorded(tmp_path):
    out = tmp_path / "evidence"
    out.mkdir()
    done = subprocess.run(
        [BASH, str(GUARDED_RUN), str(out), str(out / "describer.log"), TEST_NAME, "--",
         sys.executable, "-c", "import sys; print('Metal execution refused: busy', file=sys.stderr); sys.exit(1)"],
        capture_output=True, text=True, timeout=60, check=False, env={"PATH": BASH_PATH},
    )
    assert done.returncode == 1
    assert "the wrapper chain exited 1" in done.stdout
    assert "tree_guard printed no 'exit=' summary line, so no peak was recorded" in done.stdout
    assert "Metal execution refused: busy" in (out / "wrapper-stderr.txt").read_text()
