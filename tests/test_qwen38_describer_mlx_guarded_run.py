"""SW-7202: the Qwen3.8 describer MLX lane's guarded run (scripts/ci/qwen38-describer-mlx/guarded_run.sh)
records tree_guard's peak summary and names why it failed, for the describer smoke and for the
MiniMax-H3-Image prompt expander smoke (`QWEN38_JOB=expander`). A fake test process stands in for
the smoke binary: no model, no weights, no GPU. macOS-only, like tree_guard (`proc_pid_rusage`).
"""

from __future__ import annotations

import json
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
# tree_guard's sampling interval, read from its source so the fake child's hold follows it.
TREE_GUARD_POLL_SECONDS = float(re.search(r"^POLL_SECONDS = ([0-9.]+)$", TREE_GUARD.read_text(), re.M).group(1))
RUN_SH = ROOT / "scripts" / "ci" / "qwen38-describer-mlx" / "run.sh"
TEST_NAME = "qwen38_describer_gpu_smoke::qwen38_describer_describes_a_synthetic_scene_on_mlx"
EXPANDER_TEST_NAME = "h3_prompt_expander_gpu_smoke::h3_prompt_expander_expands_a_prompt_on_mlx"
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


def run(tmp_path: Path, child: list[str], cap_gib: str = "4", wall_s: str = "60", lock: bool = True,
        test_name: str = TEST_NAME, log_name: str = "describer.log"):
    out = tmp_path / "evidence"
    out.mkdir()
    log = out / log_name
    summary = tmp_path / "step-summary.md"
    chain = [sys.executable, str(TREE_GUARD), cap_gib, wall_s, str(log), "--", *child]
    if lock:
        chain = [sys.executable, "-I", "-c", LOCK_SHIM, str(LOCK), str(tmp_path / "metal.lock"), "--", *chain]
    done = subprocess.run(
        [BASH, str(GUARDED_RUN), str(out), str(log), test_name, "--", *chain],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env={"PATH": BASH_PATH, "GITHUB_STEP_SUMMARY": str(summary)},
    )
    guard = (out / "guard.txt").read_text() if (out / "guard.txt").exists() else ""
    return done, out, guard, summary.read_text() if summary.exists() else ""


@pytest.mark.parametrize("test_name,log_name", [(TEST_NAME, "describer.log"),
                                                (EXPANDER_TEST_NAME, "expander.log")])
def test_a_passing_test_under_the_lock_and_guard_records_the_peak(tmp_path, test_name, log_name):
    # The run that failed on nax-macos (44464702617): the test passed, the guard never stopped it,
    # so the guard printed no `GUARD:` line, only its `exit=` summary.
    #
    # tree_guard reads the root's kernel high-water mark (`ri_lifetime_max_phys_footprint`) only
    # while the root is alive, once per POLL_SECONDS. A child that touches its 64 MiB and exits
    # inside one poll interval is never read after the allocation, so the recorded peak is the
    # interpreter's start-up footprint (~8 MB). Hold the touched memory for eight poll intervals,
    # so at least one reading lands after the allocation; the high-water mark keeps it from then on.
    hold_s = 8 * TREE_GUARD_POLL_SECONDS
    done, out, guard, summary = run(tmp_path, fake_test(f"""
        import time
        held = bytearray(64 << 20)
        for i in range(0, len(held), 4096): held[i] = 1
        time.sleep({hold_s!r})
        print("running 1 test"); print({PASSED!r})
    """), test_name=test_name, log_name=log_name)
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


def run_script(tmp_path: Path, **env: str):
    return subprocess.run(
        [BASH, str(RUN_SH)], capture_output=True, text=True, timeout=60, check=False, cwd=ROOT,
        env={"PATH": BASH_PATH, "QWEN38_EVIDENCE_DIR": str(tmp_path / "evidence"), **env},
    )


@pytest.mark.parametrize("job,tier,fresh,error", [
    ("expander", "bf16", "false", "the prompt expander has no published bf16 tier; use q4 or q8"),
    ("captioner", "q4", "false", "QWEN38_JOB must be describer or expander, got captioner"),
    # A fresh hub is the expander's isolation proof; the describer keeps its provisioned cache.
    ("describer", "q4", "true", "QWEN38_FRESH_HUB applies only to the expander job"),
    ("expander", "q4", "yes", "QWEN38_FRESH_HUB must be true or false, got yes"),
    # Without the run identity there is no run-scoped path to create, or to remove.
    ("expander", "q4", "true", "a fresh hub needs the workflow run id and attempt"),
])
def test_run_refuses_a_bad_job_tier_or_fresh_hub_request_before_any_work(tmp_path, job, tier, fresh, error):
    done = run_script(tmp_path, QWEN38_JOB=job, QWEN38_TIER=tier, QWEN38_FRESH_HUB=fresh,
                      QWEN38_MAC_RUNNER="nax-macos-2")
    assert done.returncode == 1
    assert f"::error::{error}" in done.stdout, done.stdout + done.stderr
    assert not (tmp_path / "evidence").exists(), "nothing was started"


def test_run_names_each_jobs_smoke_and_the_smokes_exist_in_the_worker():
    script = RUN_SH.read_text()
    worker = ROOT / "crates" / "sceneworks-worker" / "src"
    for name in (TEST_NAME, EXPANDER_TEST_NAME):
        assert f"test_name={name}" in script, name
        module, function = name.split("::")
        assert f"async fn {function}()" in (worker / f"{module}.rs").read_text(), name


# A stand-in `huggingface_hub`: `model_info` lists the pinned bf16 files at QWEN38_FAKE_SIZES, and
# `snapshot_download` records its call and writes each listed file at its size (sparse, no data).
FAKE_HUB = textwrap.dedent(
    """
    import json, os
    from pathlib import Path
    from types import SimpleNamespace
    SIZES = json.loads(os.environ["QWEN38_FAKE_SIZES"])
    class HfApi:
        def model_info(self, repo, revision, files_metadata, token):
            return SimpleNamespace(sha=revision, siblings=[SimpleNamespace(rfilename=n, size=s) for n, s in SIZES.items()])
    def snapshot_download(repo_id, revision, cache_dir, allow_patterns, **_):
        Path(os.environ["QWEN38_FAKE_CALLS"]).open("a").write(json.dumps([repo_id, revision, allow_patterns]) + "\\n")
        snap = Path(cache_dir) / ("models--" + repo_id.replace("/", "--")) / "snapshots" / revision
        for name, size in SIZES.items():
            path = snap / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as file:
                file.truncate(size)
    """
)


# Loaded at interpreter start from the fake package's directory: the provisioner's free-space
# reading (`shutil.disk_usage`) reports QWEN38_FAKE_FREE_BYTES, so the test never depends on the
# host's disk. The production check itself is unchanged.
FAKE_FREE_SPACE = textwrap.dedent(
    """
    import os, shutil
    _real = shutil.disk_usage
    def disk_usage(path):
        usage = _real(path)
        return type(usage)(usage.total, usage.used, int(os.environ["QWEN38_FAKE_FREE_BYTES"]))
    shutil.disk_usage = disk_usage
    """
)


def describer_provision_script() -> str:
    """The describer branch's embedded provisioning Python in run.sh."""
    blocks = [block.split("\nPY\n", 1)[0] for block in RUN_SH.read_text().split("<<'PY'")[1:]]
    (script,) = [block.split("\n", 1)[1] for block in blocks if 'provision.plan("bf16")' in block]
    return script


@pytest.mark.parametrize("shard_missing", [True, False])
def test_a_bf16_retry_with_its_metadata_but_a_missing_shard_downloads_it(tmp_path, shard_missing):
    sys.path.insert(0, str(ROOT / "scripts" / "ci"))
    import qwen38_describer_provision as provision
    sys.path.pop(0)
    shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    sizes = {name: 1000 for name in provision.BF16_FILES}
    shard_bytes = (provision.BF16_ESTIMATED_BYTES - sum(sizes.values())) // 2
    sizes |= {name: shard_bytes for name in shards}
    hub = tmp_path / "hub"
    snap = hub / ("models--" + provision.BF16_REPO.replace("/", "--")) / "snapshots" / provision.BF16_REVISION
    snap.mkdir(parents=True)
    # An earlier, interrupted run: every metadata file and the index landed, one shard did not.
    for name, size in sizes.items():
        if shard_missing and name == shards[1]:
            continue
        with (snap / name).open("wb") as file:
            file.truncate(size)
    fake = tmp_path / "fake"
    (fake / "huggingface_hub").mkdir(parents=True)
    (fake / "huggingface_hub" / "__init__.py").write_text(FAKE_HUB)
    (fake / "sitecustomize.py").write_text(FAKE_FREE_SPACE)
    calls = tmp_path / "calls.jsonl"
    done = subprocess.run(
        [sys.executable, "-B", "-", ], input=describer_provision_script(), capture_output=True,
        text=True, timeout=60, check=False, cwd=ROOT,
        env={"PATH": BASH_PATH, "PYTHONPATH": str(fake), "QWEN38_HUB": str(hub), "QWEN38_TIER": "bf16",
             "QWEN38_FREE_GB": "500", "QWEN38_FAKE_SIZES": json.dumps(sizes),
             "QWEN38_FAKE_CALLS": str(calls),
             # Exactly the bf16 minimum, whatever the host has free.
             "QWEN38_FAKE_FREE_BYTES": str(provision.BF16_MIN_FREE_BYTES)},
    )
    assert done.returncode == 0, done.stdout + done.stderr
    downloads = calls.read_text().splitlines() if calls.exists() else []
    if shard_missing:
        assert [json.loads(line)[:2] for line in downloads] == [[provision.BF16_REPO, provision.BF16_REVISION]]
        assert "model-*.safetensors" in json.loads(downloads[0])[2]
        assert (snap / shards[1]).stat().st_size == shard_bytes
    else:
        assert downloads == [], "a complete snapshot is not downloaded again"
        assert "Already present" in done.stdout, done.stdout
