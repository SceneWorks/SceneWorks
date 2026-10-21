"""The Iris MLX real-weight lane's own tooling (sc-25690): the process-tree phys_footprint guard
and the encoder-free hub clone. Both are macOS-only (`proc_pid_rusage`, APFS `cp -c`); they need no
GPU and no weights.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LANE = ROOT / "scripts" / "ci" / "iris-mlx-real-weights"
GIB = 1 << 30
MIB = 1 << 20

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="macOS-only lane tooling")

# A parent holding ~96 MiB that spawns a child holding ~160 MiB; both touch every page, then idle.
TREE = textwrap.dedent(
    """
    import subprocess, sys, time
    child = subprocess.Popen([sys.executable, "-c",
        "import time\\nheld = bytearray(160 << 20)\\n"
        "for i in range(0, len(held), 4096): held[i] = 1\\ntime.sleep({hold})"])
    held = bytearray(96 << 20)
    for i in range(0, len(held), 4096):
        held[i] = 1
    print("tree up", flush=True)
    time.sleep({hold})
    child.wait()
    """
)


def run_guard(tmp_path: Path, cap_gib: float, wall_s: float, hold: float):
    log = tmp_path / "child.log"
    done = subprocess.run(
        [
            sys.executable,
            str(LANE / "tree_guard.py"),
            str(cap_gib),
            str(wall_s),
            str(log),
            "--",
            sys.executable,
            "-c",
            TREE.format(hold=hold),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    summary = [line for line in done.stdout.splitlines() if line.startswith("exit=")]
    assert len(summary) == 1, done.stdout + done.stderr
    fields = dict(re.findall(r'(\w+)=("[^"]*"|\S+)', summary[0]))
    return done, fields, log.read_text()


def test_the_guard_sums_the_process_tree_in_bytes_and_reports_gib(tmp_path):
    done, fields, log = run_guard(tmp_path, cap_gib=4, wall_s=60, hold=2.5)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "tree up" in log, "the child's output goes to the log file"
    assert "stopped" not in fields

    sampled = int(fields["peak_tree_sampled_bytes"])
    root_max = int(fields["root_lifetime_max_bytes"])
    tree_max = int(fields["tree_lifetime_max_sum_bytes"])
    # The SUM over parent and child (96 + 160 MiB of touched pages), not the parent alone.
    assert sampled >= 256 * MIB, fields
    assert int(fields["processes"]) >= 2, fields
    # The root's own kernel high-water mark covers its 96 MiB but not the child's 160 MiB.
    assert 96 * MIB <= root_max < 256 * MIB, fields
    # The sum of high-water marks bounds the sampled tree peak from above.
    assert tree_max >= sampled, fields
    # One unit: the GiB figures are the byte figures / 2^30, and the cap is GiB too.
    assert float(fields["peak_tree_sampled_gib"]) == pytest.approx(sampled / GIB, abs=5e-4)
    assert float(fields["root_lifetime_max_gib"]) == pytest.approx(root_max / GIB, abs=5e-4)
    assert fields["cap_gib"] == "4"
    assert fields["unit"] == "GiB(2^30)"
    assert fields["source"] == "proc_pid_rusage.ri_phys_footprint"


def test_the_guard_terminates_the_tree_at_the_cap(tmp_path):
    # 0.125 GiB = 128 MiB: under the tree's 256 MiB, so the SUM (not either process) trips it.
    done, fields, _ = run_guard(tmp_path, cap_gib=0.125, wall_s=60, hold=60)
    assert done.returncode != 0
    assert "GUARD: tree footprint" in done.stdout and "> cap 0.125 GiB, SIGTERM" in done.stdout
    assert fields["stopped"].startswith('"tree footprint')
    assert float(fields["wall_s"]) < 30, "the tree was stopped, not waited out"


def test_the_guard_terminates_the_tree_at_the_wall_limit(tmp_path):
    done, fields, _ = run_guard(tmp_path, cap_gib=4, wall_s=1, hold=60)
    assert done.returncode != 0
    assert fields["stopped"] == '"wall time over 1s"'
    assert float(fields["wall_s"]) < 30


def test_the_encoder_free_hub_is_cloned_not_linked(tmp_path):
    # A hub repo in the HF cache layout: the snapshot file is a symlink to a blob.
    repo = tmp_path / "shared" / "hub" / "models--org--thing"
    blob = repo / "blobs" / "abc123"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"shared hub bytes")
    snapshot = repo / "snapshots" / "rev" / "sub"
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").symlink_to(blob)
    links_before = blob.stat().st_nlink

    dest = tmp_path / "no-encoder"
    done = subprocess.run(
        [sys.executable, str(LANE / "clone_hub.py"), str(dest), str(repo)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    cloned = dest / "models--org--thing" / "snapshots" / "rev" / "sub" / "model.safetensors"
    # A real file inside the destination tree (the worker canonicalizes the path), with the bytes.
    assert cloned.is_file() and not cloned.is_symlink()
    assert os.path.realpath(cloned).startswith(os.path.realpath(dest))
    assert cloned.read_bytes() == b"shared hub bytes"
    # Its own inode: writing through the encoder-free hub cannot reach the shared blob.
    assert cloned.stat().st_ino != blob.stat().st_ino
    assert blob.stat().st_nlink == links_before
    with open(cloned, "r+b") as handle:
        handle.write(b"MUTATED")
    assert blob.read_bytes() == b"shared hub bytes"
