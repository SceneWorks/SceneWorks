"""External phys_footprint guard for one Iris real-weight smoke (sc-25690), summed over the test
process and every descendant (the two-rank DDP smoke spawns rank processes).

    tree_guard.py <cap GiB> <wall seconds> <log file> -- <command...>

Runs the command with its output in the log file, polls the process tree every 0.25 s and SIGTERMs
the whole tree when its summed footprint crosses the cap or the wall clock runs out — the only kill
path (MLX is never SIGKILLed mid-buffer from here). Report-only beyond that: nothing is gated on
the peak.

Units and precision. Every reading is BYTES from the kernel ledger — `proc_pid_rusage`
RUSAGE_INFO_V4 `ri_phys_footprint` (the number Activity Monitor's "Memory" and `footprint` show,
including Metal / unified-memory buffers) — never the `footprint` tool's rounded "N GB" text. The
cap is GiB (2^30 bytes) and is compared to the byte reading, and the summary prints bytes and GiB.
Two peaks are reported, because a 0.25 s poll can miss a spike:

  * `peak_tree_sampled`: the largest SUM over the live tree seen at a poll (a lower bound of the
    true tree peak, 0.25 s resolution);
  * `root_lifetime_max`: the kernel's own high-water mark of the root process
    (`ri_lifetime_max_phys_footprint`, exact, no sampling) — the peak of a single-process smoke;
  * `tree_lifetime_max_sum`: the sum of every seen process's high-water mark (an upper bound of
    the tree peak: the processes need not peak together).
"""

import ctypes
from pathlib import Path
import os
import signal
import subprocess
import sys
import time


def inherited_metal_fds():
    if sys.platform != "darwin" or "SCENEWORKS_METAL_EXECUTION_LOCK_FD" not in os.environ:
        return ()
    import importlib.util
    source = Path(__file__).resolve().parents[1] / "metal_execution_fds.py"
    name = "sceneworks_metal_execution_fds"
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module.inherited_metal_fds()


GIB = 1 << 30
POLL_SECONDS = 0.25
# struct rusage_info_v4, as uint64 words (ri_uuid is words 0-1).
RUSAGE_INFO_V4 = 4
RI_PHYS_FOOTPRINT = 9
RI_LIFETIME_MAX_PHYS_FOOTPRINT = 30

_libc = None
_buf = (ctypes.c_uint64 * 64)()


def rusage(pid):
    """`(phys_footprint, lifetime_max_phys_footprint)` bytes of `pid`, or `None` once it is gone."""
    global _libc
    if _libc is None:
        _libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    if _libc.proc_pid_rusage(pid, RUSAGE_INFO_V4, ctypes.byref(_buf)) != 0:
        return None
    return int(_buf[RI_PHYS_FOOTPRINT]), int(_buf[RI_LIFETIME_MAX_PHYS_FOOTPRINT])


def tree_pids(root):
    """`root` and every live descendant."""
    listing = subprocess.run(
        ["/bin/ps", "-axo", "pid=,ppid="], capture_output=True, text=True, check=False
    ).stdout
    children = {}
    for line in listing.splitlines():
        parts = line.split()
        if len(parts) == 2:
            children.setdefault(int(parts[1]), []).append(int(parts[0]))
    seen, stack = [], [root]
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.append(pid)
        stack.extend(children.get(pid, []))
    return seen


def gib(value):
    return f"{value / GIB:.3f}"


def main(argv):
    if len(argv) < 5 or argv[3] != "--":
        print(__doc__, file=sys.stderr)
        return 2
    cap_gib, wall_s, log_path, command = float(argv[0]), float(argv[1]), argv[2], argv[4:]
    cap_bytes = int(cap_gib * GIB)
    start = time.time()
    with open(log_path, "wb") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, pass_fds=inherited_metal_fds())
    peak_sampled = 0
    lifetime_max = {}
    stopped = None
    while child.poll() is None:
        total = 0
        pids = tree_pids(child.pid)
        for pid in pids:
            reading = rusage(pid)
            if reading is None:
                continue
            total += reading[0]
            lifetime_max[pid] = max(lifetime_max.get(pid, 0), reading[1])
        peak_sampled = max(peak_sampled, total)
        if stopped is None:
            if total > cap_bytes:
                stopped = f"tree footprint {gib(total)} GiB > cap {cap_gib:g} GiB"
            elif time.time() - start > wall_s:
                stopped = f"wall time over {wall_s:g}s"
            if stopped is not None:
                print(f"GUARD: {stopped}, SIGTERM", flush=True)
                for pid in pids:
                    try:
                        os.kill(pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
        time.sleep(POLL_SECONDS)
    code = child.wait()
    root_max = lifetime_max.get(child.pid, 0)
    print(
        f"exit={code}"
        f" peak_tree_sampled_bytes={peak_sampled} peak_tree_sampled_gib={gib(peak_sampled)}"
        f" root_lifetime_max_bytes={root_max} root_lifetime_max_gib={gib(root_max)}"
        f" tree_lifetime_max_sum_bytes={sum(lifetime_max.values())}"
        f" tree_lifetime_max_sum_gib={gib(sum(lifetime_max.values()))}"
        f" processes={len(lifetime_max)} cap_gib={cap_gib:g} unit=GiB(2^30)"
        f" source=proc_pid_rusage.ri_phys_footprint poll_s={POLL_SECONDS:g}"
        f" wall_s={time.time() - start:.0f}" + (f' stopped="{stopped}"' if stopped else ""),
        flush=True,
    )
    return code if code is not None else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
