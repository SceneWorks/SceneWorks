#!/usr/bin/env python3
"""Fail-closed process-group guard for physical memory calibration.

Defense in depth only: this monitor never makes a safety-refused row admissible. Production Darwin
telemetry is the kernel-maintained phys_footprint from /usr/bin/footprint. Synthetic telemetry is
available only behind an explicit test-only flag.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

HARD_STOP_EXIT = 97


@dataclass(frozen=True)
class Identity:
    pid: int
    pgid: int
    state: str = field(compare=False)
    started: str


def process_identity(pid: int) -> Identity | None:
    result = subprocess.run(
        ["/bin/ps", "-ww", "-p", str(pid), "-o", "pgid=,state=,lstart="],
        capture_output=True, text=True, timeout=1, check=False,
    )
    fields = result.stdout.strip().split(None, 2)
    if result.returncode != 0 or len(fields) != 3 or fields[1].startswith("Z"):
        return None
    return Identity(pid, int(fields[0]), fields[1], fields[2])


def group_identities(pgid: int) -> list[Identity]:
    probe = subprocess.Popen(
        ["/bin/ps", "-ww", "-axo", "pid=,pgid=,state=,lstart="],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    stdout, stderr = probe.communicate(timeout=1)
    if probe.returncode != 0:
        raise RuntimeError(f"ps group census failed: {stderr.strip()}")
    members = []
    for line in stdout.splitlines():
        fields = line.strip().split(None, 3)
        if (len(fields) == 4 and int(fields[0]) != probe.pid and int(fields[1]) == pgid
                and not fields[2].startswith("Z")):
            members.append(Identity(int(fields[0]), pgid, fields[2], fields[3]))
    return members


def identity_is_live(identity: Identity) -> bool:
    current = process_identity(identity.pid)
    return bool(current and current.pgid == identity.pgid and current.started == identity.started)


def anchor_main() -> int:
    """TERM-resistant exact-identity anchor retained if the launch sentinel crashes."""
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    while True:
        signal.pause()


def sentinel_main(control_fd: int, command: list[str]) -> int:
    """Stable launch-owned PGID anchor; it outlives an early-exiting guarded root."""
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        raise RuntimeError("sentinel requires a guarded command")
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    def restore_child_signals() -> None:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)

    def cleanup_descendants() -> None:
        for sig, deadline in [(signal.SIGTERM, time.monotonic() + 0.5), (signal.SIGKILL, time.monotonic() + 1.0)]:
            members = [item for item in group_identities(os.getpgrp()) if item.pid != os.getpid()]
            if not members:
                break
            for member in members:
                try:
                    os.kill(member.pid, sig)
                except ProcessLookupError:
                    pass
            while time.monotonic() < deadline:
                if not any(identity_is_live(item) for item in members):
                    break
                time.sleep(0.02)
        survivors = [item for item in group_identities(os.getpgrp()) if item.pid != os.getpid()]
        if survivors:
            raise RuntimeError(f"sentinel retained live descendants: {survivors}")

    control = socket.socket(fileno=control_fd)
    anchor = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--anchor"])
    try:
        child = subprocess.Popen(command, preexec_fn=restore_child_signals)
    except BaseException:
        anchor.kill()
        anchor.wait()
        raise
    try:
        control.sendall(f"R {anchor.pid}\n".encode())
        acknowledged = control.recv(1) == b"G"
    except BaseException:
        cleanup_descendants()
        child.wait()
        raise
    if not acknowledged:
        cleanup_descendants()
        child.wait()
        return HARD_STOP_EXIT
    control.close()
    status = child.wait()
    # The command root may have exited after spawning descendants. The sentinel remains the exact
    # PGID anchor and removes every remaining member before propagating the root's status.
    cleanup_descendants()
    return 128 - status if status < 0 else status


class DarwinFootprintSampler:
    @staticmethod
    def parse_processes(pids: list[int], payload: object) -> int:
        requested = set(pids)
        if len(requested) != len(pids):
            raise RuntimeError("footprint request contains duplicate PIDs")
        processes = payload.get("processes") if isinstance(payload, dict) else None
        if not isinstance(processes, list):
            raise RuntimeError("footprint returned no process telemetry")
        observed: dict[int, int] = {}
        for process in processes:
            if not isinstance(process, dict):
                raise RuntimeError("footprint returned malformed process telemetry")
            pid = process.get("pid")
            auxiliary = process.get("auxiliary")
            value = auxiliary.get("phys_footprint") if isinstance(auxiliary, dict) else None
            if not isinstance(pid, int) or not isinstance(value, int) or value < 0:
                raise RuntimeError("footprint omitted PID or non-negative phys_footprint")
            if pid in observed:
                raise RuntimeError(f"footprint returned duplicate PID {pid}")
            observed[pid] = value
        if set(observed) != requested:
            missing = sorted(requested - set(observed))
            extra = sorted(set(observed) - requested)
            raise RuntimeError(f"footprint PID set mismatch: missing={missing}, extra={extra}")
        return sum(observed.values())

    def sample(self, pids: list[int], timeout: float) -> int:
        if sys.platform != "darwin":
            raise RuntimeError("Darwin phys_footprint telemetry is unavailable")
        if not pids:
            raise RuntimeError("owned group has no live members")
        fd, output = tempfile.mkstemp(prefix="sceneworks-footprint-", suffix=".json")
        os.close(fd)
        try:
            command = ["/usr/bin/footprint", "--noCategories", "-j", output]
            for pid in pids:
                command.extend(["-p", str(pid)])
            result = subprocess.run(
                command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                timeout=timeout, check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(f"footprint exited {result.returncode}: {result.stderr.strip()}")
            return self.parse_processes(pids, json.loads(Path(output).read_text()))
        finally:
            Path(output).unlink(missing_ok=True)


class SyntheticFileSampler:
    def __init__(self, path: Path):
        self.path = path

    def sample(self, pids: list[int], timeout: float) -> int:
        del pids, timeout
        value = int(self.path.read_text().strip())
        if value < 0:
            raise RuntimeError("synthetic footprint must be non-negative")
        return value


class OwnedGroup:
    def __init__(self, command: list[str], spawn_delay: float = 0.0):
        parent_control, child_control = socket.socketpair()
        sentinel = [
            sys.executable, str(Path(__file__).resolve()), "--sentinel",
            str(child_control.fileno()), "--", *command,
        ]
        def unblock_monitor_signals() -> None:
            signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT, signal.SIGTERM})

        try:
            self.child = subprocess.Popen(
                sentinel, start_new_session=True, preexec_fn=unblock_monitor_signals,
                pass_fds=(child_control.fileno(),),
            )
        except BaseException:
            parent_control.close()
            raise
        finally:
            child_control.close()
        self.pgid = self.child.pid
        if spawn_delay:
            time.sleep(spawn_delay)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            leader = process_identity(self.child.pid)
            if leader and leader.pgid == self.pgid:
                self.leader = leader
                try:
                    parent_control.settimeout(2)
                    ready = parent_control.makefile("rb").readline(64).decode().strip().split()
                    if len(ready) != 2 or ready[0] != "R":
                        raise RuntimeError("launch sentinel closed before readiness")
                    anchor = process_identity(int(ready[1]))
                    if not anchor or anchor.pgid != self.pgid:
                        raise RuntimeError("launch sentinel reported an invalid group anchor")
                    self.anchors = (leader, anchor)
                    self.retained = {leader, anchor}
                    self.retained.update(group_identities(self.pgid))
                    parent_control.sendall(b"G")
                    return
                except BaseException:
                    self.terminate(0.1)
                    raise
                finally:
                    parent_control.close()
            if self.child.poll() is not None:
                parent_control.close()
                raise RuntimeError("guarded command exited before establishing its process group")
            time.sleep(0.02)
        try:
            if self.child.poll() is None and os.getpgid(self.child.pid) == self.child.pid:
                os.killpg(self.child.pid, signal.SIGKILL)
            elif self.child.poll() is None:
                self.child.kill()
        except ProcessLookupError:
            pass
        self.child.wait(timeout=1)
        parent_control.close()
        raise RuntimeError("guarded command did not establish its process group")

    def refresh(self) -> list[Identity]:
        # Numeric PGID census is safe only while an exact launch-owned anchor proves the original
        # group still exists. The auxiliary anchor outlives a killed sentinel and cannot exit on
        # TERM/INT, closing the between-censuses descendant race without permitting PGID reuse.
        if any(identity_is_live(anchor) for anchor in self.anchors):
            self.retained.update(group_identities(self.pgid))
        return [identity for identity in self.retained if identity_is_live(identity)]

    def terminate(self, grace: float) -> None:
        live = self.refresh()
        if any(identity_is_live(anchor) for anchor in self.anchors):
            try:
                os.killpg(self.pgid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        else:
            for identity in live:
                try:
                    os.kill(identity.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline and any(identity_is_live(item) for item in live):
            live = self.refresh()
            time.sleep(0.02)
        live = self.refresh()
        if any(identity_is_live(anchor) for anchor in self.anchors):
            try:
                os.killpg(self.pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for identity in live:
            if identity_is_live(identity):
                try:
                    os.kill(identity.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        try:
            self.child.wait(timeout=2)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("owned process-group leader did not terminate") from error
        survivors = [identity for identity in self.retained if identity_is_live(identity)]
        if survivors:
            raise RuntimeError(f"owned process group retained live identities: {survivors}")


def emit(event_file: Path | None, event: dict[str, object]) -> None:
    line = json.dumps({"at": time.time(), **event}, separators=(",", ":"))
    if event_file:
        with event_file.open("a") as output:
            output.write(f"{line}\n")
            output.flush()
            os.fsync(output.fileno())
    else:
        print(line, file=sys.stderr, flush=True)


class MonitorSignal(Exception):
    def __init__(self, signum: int):
        self.signum = signum


def guard(args: argparse.Namespace) -> int:
    if args.telemetry_file:
        if not args.allow_synthetic_telemetry:
            raise RuntimeError("--telemetry-file requires --allow-synthetic-telemetry")
        sampler = SyntheticFileSampler(args.telemetry_file)
    else:
        sampler = DarwinFootprintSampler()
    if (args.synthetic_spawn_delay or args.synthetic_launch_ready_file) and not args.allow_synthetic_telemetry:
        raise RuntimeError("synthetic launch controls require --allow-synthetic-telemetry")
    hard_stop = None
    exit_status = HARD_STOP_EXIT
    previous_handlers = {
        signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
    }

    def interrupted(signum: int, _frame: object) -> None:
        raise MonitorSignal(signum)

    for signum in previous_handlers:
        signal.signal(signum, interrupted)
    # Block monitor signals across the sentinel spawn. When unblocked, any pending signal reaches
    # the installed handler only after `group` exists, so cleanup cannot lose the launch race.
    blocked = set(previous_handlers)
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, blocked)
    if args.synthetic_launch_ready_file:
        args.synthetic_launch_ready_file.write_text("ready\n")
    group = OwnedGroup(args.command, args.synthetic_spawn_delay)
    try:
        # A signal pending from the blocked launch window is delivered here, inside the cleanup
        # try, never in the gap between establishing the group and arming cleanup.
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        emit(args.event_file, {"event": "started", "pid": group.child.pid, "pgid": group.pgid})
        while True:
            live = group.refresh()
            status = group.child.poll()
            if status is not None and status < 0:
                hard_stop = f"launch_sentinel_lost:status_{status}"
                break
            if not live:
                return status if status is not None else 0
            if status is not None:
                # The census preceded poll; normal sentinel cleanup may have completed between
                # those observations. Refresh before treating a positive status as a failure.
                if not group.refresh():
                    return status
                hard_stop = f"launch_sentinel_failed_with_live_group:status_{status}"
                break
            started = time.monotonic()
            try:
                footprint = sampler.sample([item.pid for item in live], args.telemetry_timeout)
            except Exception as error:  # fail closed on timeout, parse failure, or source loss
                if not group.refresh() and group.child.poll() is not None:
                    return group.child.returncode
                hard_stop = f"telemetry_lost:{type(error).__name__}:{error}"
                break
            elapsed = time.monotonic() - started
            if elapsed > args.telemetry_timeout:
                hard_stop = f"telemetry_stale:{elapsed:.3f}s"
                break
            emit(args.event_file, {"event": "sample", "physicalFootprintBytes": footprint})
            if footprint >= args.max_footprint_bytes:
                hard_stop = f"physical_footprint_at_or_above_{args.max_footprint_bytes}:observed_{footprint}"
                break
            time.sleep(args.sample_interval)
    except MonitorSignal as caught:
        hard_stop = f"monitor_signal_{signal.Signals(caught.signum).name}"
        exit_status = 128 + caught.signum
    except BaseException as error:
        hard_stop = f"monitor_failure:{type(error).__name__}:{error}"
        exit_status = HARD_STOP_EXIT
    finally:
        if hard_stop is not None:
            try:
                emit(args.event_file, {"event": "hard_stop", "reason": hard_stop})
            except Exception:
                pass
            group.terminate(args.term_grace)
            try:
                emit(args.event_file, {"event": "terminated", "reason": hard_stop})
            except Exception:
                pass
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    return exit_status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-footprint-bytes", type=int, required=True)
    parser.add_argument("--sample-interval", type=float, default=0.25)
    parser.add_argument("--telemetry-timeout", type=float, default=1.0)
    parser.add_argument("--term-grace", type=float, default=0.5)
    parser.add_argument("--event-file", type=Path)
    parser.add_argument("--telemetry-file", type=Path)
    parser.add_argument("--allow-synthetic-telemetry", action="store_true")
    parser.add_argument("--synthetic-spawn-delay", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--synthetic-launch-ready-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command:
        parser.error("a guarded command is required after --")
    for name in ["max_footprint_bytes", "sample_interval", "telemetry_timeout", "term_grace"]:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.synthetic_spawn_delay < 0:
        parser.error("--synthetic-spawn-delay must be non-negative")
    return args


if __name__ == "__main__":
    try:
        if sys.argv[1:2] == ["--anchor"]:
            raise SystemExit(anchor_main())
        if sys.argv[1:2] == ["--sentinel"]:
            raise SystemExit(sentinel_main(int(sys.argv[2]), sys.argv[3:]))
        raise SystemExit(guard(parse_args()))
    except Exception as error:
        print(f"memory calibration watchdog failed closed: {error}", file=sys.stderr)
        raise SystemExit(HARD_STOP_EXIT)
