"""Preserve an existing canonical lease through Python supervisors; never acquire a fresh one.

Model launchers acquire OUTSIDE the physical watchdog so the model keeps its original sampled
process group. No namespace override, private fallback, new model limits, or GPU signalling.
"""
from __future__ import annotations
import importlib.util
import os
from pathlib import Path
import sys

_protocol = None


def inherited_metal_fds() -> tuple[int, ...]:
    raw = os.environ.get('SCENEWORKS_METAL_EXECUTION_LOCK_FD')
    if sys.platform != 'darwin' or raw is None:
        return ()
    global _protocol
    if _protocol is None:
        source = Path(__file__).with_name('metal_execution_lock.py')
        spec = importlib.util.spec_from_file_location('sceneworks_metal_execution_protocol', source)
        _protocol = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = _protocol
        spec.loader.exec_module(_protocol)
    # execution_lease takes the inherited branch: same-inode/kernel-flock validation, no wait or
    # fresh namespace. Closing its borrowed duplicate MUST NOT unlock the supervising owner.
    with _protocol.execution_lease(1) as lease:
        if not lease.borrowed:
            raise RuntimeError('Python supervisor requires an inherited Metal execution lease')
        return (int(raw),)
