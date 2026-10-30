#!/usr/bin/env python3
"""Host-global cooperative Metal execution lease, shared verbatim by both repositories.

No unlink, replacement, private fallback, GPU probing, model cancellation or resource caps.
All cooperating launchers use CANONICAL_PATH. Unwrapped user processes still need the existing
physical census/operator coordination: this is not a universal device scheduler.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import errno
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time

CANONICAL_PATH = Path('/Users/Shared/SceneWorks/gpu-locks/apple-metal.lock')
INHERITED_FD = 'SCENEWORKS_METAL_EXECUTION_LOCK_FD'
PROTOCOL = 'sceneworks-host-metal-flock-v1'


@dataclass
class Lease:
    fd: int
    borrowed: bool
    path: Path

    def environment(self, original: dict[str, str]) -> dict[str, str]:
        return dict(original, **{INHERITED_FD: str(self.fd)})

    def record(self) -> dict:
        info = os.fstat(self.fd)
        return {'protocol': PROTOCOL, 'path': str(self.path), 'pid': os.getpid(),
                'device': info.st_dev, 'inode': info.st_ino, 'borrowed': self.borrowed,
                'scope': 'cooperating launchers only; unwrapped processes require census/operator coordination'}


def _validate(path: Path, fd: int) -> None:
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError('Metal lease path/ancestor must not be a symlink')
    opened = os.fstat(fd)
    named = path.stat()
    if (not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)):
        raise ValueError('Metal lease must retain the same unique regular inode')


@contextmanager
def _acquire(path: Path, wait_seconds: float, inherited: str | None = None):
    """Private path seam permits CPU-only tests; production callers cannot select a namespace."""
    if type(wait_seconds) not in (float, int) or not math.isfinite(wait_seconds) or wait_seconds <= 0:
        raise ValueError('Metal acquisition wait must be explicit, finite and positive')
    if inherited is not None:
        if not inherited.isdecimal() or int(inherited) < 3:
            raise ValueError('invalid inherited Metal lease descriptor')
        fd = os.dup(int(inherited))
        try:
            _validate(path, fd)
            # A duplicated/inherited open-file description already holds the same kernel lock.
            # A forged separately-opened descriptor cannot bypass a holder of this inode.
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield Lease(fd, True, path)
        finally:
            os.close(fd)  # NEVER LOCK_UN a borrowed description; the outer owner retains it.
        return
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError('Metal lease path/ancestor must not be a symlink')
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, 'O_NOFOLLOW', 0)
    fd = os.open(path, flags, 0o666)
    acquired = False
    try:
        _validate(path, fd)
        started = time.monotonic()
        while not acquired:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except OSError as error:
                if error.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
                if time.monotonic() - started >= wait_seconds:
                    raise TimeoutError('approved Metal acquisition wait expired; no model was launched')
                time.sleep(min(0.1, max(0.001, wait_seconds / 10)))
        _validate(path, fd)
        yield Lease(fd, False, path)
    finally:
        if acquired:
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


@contextmanager
def execution_lease(wait_seconds: float):
    """Hold around an EXISTING owned-child-tree supervisor, which must finish before exiting."""
    with _acquire(CANONICAL_PATH, wait_seconds, os.environ.get(INHERITED_FD)) as lease:
        yield lease


def _processes() -> dict[int, tuple[int, int, str]]:
    result = subprocess.run(['/bin/ps', '-axo', 'pid=,ppid=,pgid=,stat='], check=True,
                            capture_output=True, text=True, encoding='utf-8')
    table = {}
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) != 4 or not all(field.isdecimal() for field in fields[:3]):
            raise ValueError('owned-tree process census is malformed')
        table[int(fields[0])] = (int(fields[1]), int(fields[2]), fields[3])
    return table


def run_owned(argv: list[str], lease: Lease) -> int:
    """Await root AND observed descendants, including changed groups; never signal a GPU child.

    Same session/ancestry ownership approach as the existing campaign supervisor. Launchers must
    not deliberately daemonize/double-fork away from ownership. SIGINT/TERM to this wrapper are
    deferred until the owned tree finishes; SIGKILL/host loss cannot be made safe by a file lock.
    """
    if not argv:
        raise ValueError('a fixed launcher command is required')
    interrupted = []
    previous = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(signum, lambda number, _frame: interrupted.append(number))
    child = None
    try:
        child = subprocess.Popen(argv, env=lease.environment(dict(os.environ)), start_new_session=True,
                                 pass_fds=(lease.fd,))
        known = {child.pid}
        while True:
            code = child.poll()
            try:
                table = _processes()
            except (OSError, ValueError, subprocess.SubprocessError):
                # Never release the lease while ownership is unknown, or kill a model to recover it.
                time.sleep(0.1)
                continue
            changed = True
            while changed:
                changed = False
                for pid, (parent, group, _state) in table.items():
                    if (parent in known or group == child.pid) and pid not in known:
                        known.add(pid)
                        changed = True
            active = {pid for pid in known if pid in table and not table[pid][2].startswith('Z')}
            if code is not None and not active:
                if interrupted:
                    return 128 + interrupted[0]
                return code if code >= 0 else 128 - code
            time.sleep(0.1)
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        # No terminate()/kill() here. Normal return occurs only after the owned tree is gone.


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-seconds', type=float, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    argv = args.command[1:] if args.command[:1] == ['--'] else args.command
    try:
        with execution_lease(args.wait_seconds) as lease:
            print('METAL_EXECUTION_LEASE ' + json.dumps(lease.record(), sort_keys=True), file=sys.stderr, flush=True)
            return run_owned(argv, lease)
    except (ValueError, OSError, TimeoutError) as error:
        print('Metal execution refused: ' + str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
