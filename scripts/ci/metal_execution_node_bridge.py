#!/usr/bin/env python3
"""CPU-only Node FD bridge to the unchanged canonical Metal execution protocol.

The Node parent opens the canonical file and passes that real descriptor to this child. A
borrowed execution_lease locks the SAME open-file-description; closing this child's duplicate
never LOCK_UNs the Node/worker references. No namespace override, GPU/model work, or PID signals.
"""
from __future__ import annotations
import argparse
import errno
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

SOURCE = Path(__file__).with_name('metal_execution_lock.py')
spec = importlib.util.spec_from_file_location('sceneworks_node_metal_protocol', SOURCE)
protocol = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = protocol
spec.loader.exec_module(protocol)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('prepare', 'acquire', 'inspect'), required=True)
    parser.add_argument('--wait-seconds', type=float, required=True)
    parser.add_argument('--marker-offset', type=int)
    args = parser.parse_args(argv)
    if not math.isfinite(args.wait_seconds) or args.wait_seconds <= 0:
        raise ValueError('explicit finite positive acquisition budget required')
    if args.mode == 'prepare':
        if protocol.INHERITED_FD in os.environ:
            raise ValueError('fresh canonical bootstrap cannot replace an inherited description')
        with protocol.execution_lease(args.wait_seconds):
            pass  # canonical helper alone creates/validates the fixed production namespace
        return 0
    raw = os.environ.get(protocol.INHERITED_FD)
    if raw is None or not raw.isdecimal() or int(raw) < 3:
        raise ValueError('real inherited descriptor required')
    fd = int(raw)
    if args.marker_offset is not None and not 0 < args.marker_offset <= (1 << 53) - 1:
        raise ValueError('invalid private owned-description marker')
    if args.mode == 'inspect':
        protocol._validate(protocol.CANONICAL_PATH, fd)
        if args.marker_offset is not None:
            os.lseek(fd, args.marker_offset, os.SEEK_SET)
        print(json.dumps({'offset': os.lseek(fd, 0, os.SEEK_CUR)}), flush=True)
        return 0
    deadline = time.monotonic() + args.wait_seconds
    while True:
        try:
            with protocol.execution_lease(args.wait_seconds) as lease:
                if not lease.borrowed:
                    raise ValueError('bridge must retain the Node parent description')
                # File OFFSET only, never file contents: detect an accidentally closed/reused
                # parent FD even if it was reopened on the same canonical inode. Borrowers never
                # alter a supervising owner's marker.
                if args.marker_offset is not None and os.lseek(fd, 0, os.SEEK_CUR) != args.marker_offset:
                    raise ValueError('owned open-file-description marker changed')
                print('METAL_EXECUTION_LEASE ' + json.dumps(lease.record(), sort_keys=True), flush=True)
                return 0
        except OSError as error:
            if error.errno not in (errno.EACCES, errno.EAGAIN):
                raise
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('approved Metal acquisition wait expired; no model was launched') from error
            time.sleep(min(0.1, remaining))


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, TimeoutError) as error:
        print('Metal execution refused: ' + str(error), file=sys.stderr)
        raise SystemExit(1)
