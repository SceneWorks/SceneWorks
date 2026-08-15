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
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

HARD_STOP_EXIT = 97
ATTESTED_INITIAL_MEMORY_FREE_PERCENT = 70


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


def sentinel_main(
        control_fd: int, attestation_path: str | None, command: list[str]) -> int:
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
    child = None
    try:
        control.sendall(f"R {anchor.pid}\n".encode())
        acknowledged = control.recv(1) == b"G"
        released = acknowledged and control.recv(1) == b"S"
        if released:
            environment = os.environ.copy()
            if attestation_path is not None:
                environment["SCENEWORKS_MEMORY_WATCHDOG_SOCKET"] = attestation_path
            child = subprocess.Popen(
                command, preexec_fn=restore_child_signals, env=environment,
            )
    except BaseException:
        cleanup_descendants()
        if child is not None:
            child.wait()
        raise
    if not released or child is None:
        cleanup_descendants()
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


@dataclass(frozen=True)
class HostPressure:
    memory_free_percent: int
    memory_free_bytes: int
    swap_free_bytes: int


class DarwinHostPressureSampler:
    def __init__(self, memory_bytes: int):
        self.memory_bytes = memory_bytes

    @staticmethod
    def actual_host_memory_bytes(timeout: float) -> int:
        result = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "hw.memsize"], capture_output=True, text=True,
            timeout=timeout, check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"hw.memsize exited {result.returncode}: {result.stderr.strip()}")
        try:
            value = int(result.stdout.strip())
        except ValueError as error:
            raise RuntimeError("hw.memsize did not report an integer") from error
        if value <= 0:
            raise RuntimeError("hw.memsize did not report positive installed memory")
        return value

    @staticmethod
    def parse_memory_free_percent(output: str) -> int:
        marker = "System-wide memory free percentage:"
        matches = [line for line in output.splitlines() if marker in line]
        if len(matches) != 1:
            raise RuntimeError("memory_pressure did not report one free percentage")
        raw = matches[0].split(marker, 1)[1].strip()
        if not raw.endswith("%"):
            raise RuntimeError("memory_pressure free percentage is malformed")
        value = int(raw[:-1])
        if value < 0 or value > 100:
            raise RuntimeError("memory_pressure free percentage is out of range")
        return value

    @staticmethod
    def parse_swap_free_bytes(output: str) -> int:
        import re
        match = re.search(r"\bfree\s*=\s*([0-9]+(?:\.[0-9]+)?)([MG])\b", output, re.IGNORECASE)
        if not match:
            raise RuntimeError("vm.swapusage did not report free swap")
        multiplier = 1024 ** (3 if match.group(2).upper() == "G" else 2)
        return int(float(match.group(1)) * multiplier)

    def sample(self, timeout: float) -> HostPressure:
        deadline = time.monotonic() + timeout
        def remaining() -> float:
            value = deadline - time.monotonic()
            if value <= 0:
                raise TimeoutError("aggregate host-pressure telemetry deadline expired")
            return value
        pressure = subprocess.run(
            ["/usr/bin/memory_pressure"], capture_output=True, text=True,
            timeout=remaining(), check=False,
        )
        if pressure.returncode != 0:
            raise RuntimeError(f"memory_pressure exited {pressure.returncode}: {pressure.stderr.strip()}")
        swap = subprocess.run(
            ["/usr/sbin/sysctl", "vm.swapusage"], capture_output=True, text=True,
            timeout=remaining(), check=False,
        )
        if swap.returncode != 0:
            raise RuntimeError(f"vm.swapusage exited {swap.returncode}: {swap.stderr.strip()}")
        percent = self.parse_memory_free_percent(pressure.stdout)
        return HostPressure(
            percent,
            self.memory_bytes * percent // 100,
            self.parse_swap_free_bytes(swap.stdout),
        )


class SyntheticHostPressureSampler:
    def __init__(self, path: Path):
        self.path = path

    def sample(self, timeout: float) -> HostPressure:
        del timeout
        payload = json.loads(self.path.read_text())
        return HostPressure(
            int(payload["memoryFreePercent"]),
            int(payload["memoryFreeBytes"]),
            int(payload["swapFreeBytes"]),
        )


class OwnedGroup:
    def __init__(
            self, command: list[str], spawn_delay: float = 0.0,
            attestation_path: str | None = None):
        parent_control, child_control = socket.socketpair()
        sentinel = [
            sys.executable, str(Path(__file__).resolve()), "--sentinel",
            str(child_control.fileno()),
            attestation_path if attestation_path is not None else "-",
            "--", *command,
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
                    self.control = parent_control
                    self.released = False
                    return
                except BaseException:
                    parent_control.close()
                    if hasattr(self, "anchors"):
                        self.terminate(0.1)
                    else:
                        try:
                            os.killpg(self.pgid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        self.child.wait(timeout=1)
                    raise
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

    def release(self) -> None:
        if self.released:
            raise RuntimeError("guarded command was already released")
        self.control.sendall(b"S")
        self.control.close()
        self.released = True

    def refresh(self) -> list[Identity]:
        # Numeric PGID census is safe only while an exact launch-owned anchor proves the original
        # group still exists. The auxiliary anchor outlives a killed sentinel and cannot exit on
        # TERM/INT, closing the between-censuses descendant race without permitting PGID reuse.
        if any(identity_is_live(anchor) for anchor in self.anchors):
            self.retained.update(group_identities(self.pgid))
        return [identity for identity in self.retained if identity_is_live(identity)]

    def terminate(self, grace: float) -> None:
        if hasattr(self, "control") and not self.released:
            self.control.close()
            self.released = True
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


def recv_line(sock: socket.socket, limit: int = 4096) -> str:
    payload = bytearray()
    while len(payload) < limit:
        chunk = sock.recv(1)
        if not chunk:
            raise RuntimeError("attestation channel closed before a complete line")
        if chunk == b"\n":
            return payload.decode()
        payload.extend(chunk)
    raise RuntimeError("attestation line exceeded its size bound")


def observe_group(
        group: OwnedGroup, sampler: object, host_sampler: object | None,
        timeout: float) -> tuple[list[Identity], int, HostPressure | None, float]:
    started = time.monotonic()
    deadline = started + timeout

    def remaining() -> float:
        value = deadline - time.monotonic()
        if value <= 0:
            raise TimeoutError("aggregate footprint and host-pressure deadline expired")
        return value

    live = group.refresh()
    if not live:
        raise RuntimeError("owned group has no live identities")
    footprint = sampler.sample([item.pid for item in live], remaining())
    pressure = host_sampler.sample(remaining()) if host_sampler is not None else None
    elapsed = time.monotonic() - started
    if elapsed > timeout:
        raise TimeoutError(f"aggregate telemetry stale after {elapsed:.3f}s")
    return live, footprint, pressure, elapsed


def guard(args: argparse.Namespace) -> int:
    attested_initial_memory_free_bytes = None
    if args.require_child_attestation:
        actual_host_memory = DarwinHostPressureSampler.actual_host_memory_bytes(
            args.telemetry_timeout,
        )
        if args.host_memory_bytes != actual_host_memory:
            raise RuntimeError(
                f"child attestation host memory {args.host_memory_bytes} does not match "
                f"hw.memsize {actual_host_memory}"
            )
        telemetry_resolution = (actual_host_memory + 99) // 100
        attested_initial_memory_free_bytes = 2 * args.max_footprint_bytes + telemetry_resolution
    if args.telemetry_file:
        if not args.allow_synthetic_telemetry:
            raise RuntimeError("--telemetry-file requires --allow-synthetic-telemetry")
        sampler = SyntheticFileSampler(args.telemetry_file)
    else:
        sampler = DarwinFootprintSampler()
    host_sampler = None
    if args.host_pressure_file:
        if not args.allow_synthetic_telemetry:
            raise RuntimeError("--host-pressure-file requires --allow-synthetic-telemetry")
        host_sampler = SyntheticHostPressureSampler(args.host_pressure_file)
    elif args.host_memory_bytes is not None:
        host_sampler = DarwinHostPressureSampler(args.host_memory_bytes)
    if (args.synthetic_spawn_delay or args.synthetic_launch_ready_file) and not args.allow_synthetic_telemetry:
        raise RuntimeError("synthetic launch controls require --allow-synthetic-telemetry")
    hard_stop = None
    exit_status = HARD_STOP_EXIT
    attestation_listener = None
    attestation_stream = None
    attestation_directory = None
    attestation_path = None
    if args.require_child_attestation:
        attestation_directory = Path(tempfile.mkdtemp(prefix="sceneworks-watchdog-attestation-"))
        attestation_path = attestation_directory / "watchdog.sock"
        attestation_listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        attestation_listener.bind(str(attestation_path))
        attestation_listener.listen(1)
    previous_handlers = {
        signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
    }
    interrupted_signum: int | None = None

    def interrupted(signum: int, _frame: object) -> None:
        nonlocal interrupted_signum
        interrupted_signum = signum
        raise MonitorSignal(signum)

    for signum in previous_handlers:
        signal.signal(signum, interrupted)
    # Block monitor signals across the sentinel spawn. When unblocked, any pending signal reaches
    # the installed handler only after `group` exists, so cleanup cannot lose the launch race.
    blocked = set(previous_handlers)
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, blocked)
    if args.synthetic_launch_ready_file:
        args.synthetic_launch_ready_file.write_text("ready\n")
    try:
        group = OwnedGroup(
            args.command, args.synthetic_spawn_delay,
            str(attestation_path) if attestation_path is not None else None,
        )
    except BaseException:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        if attestation_listener is not None:
            attestation_listener.close()
        if attestation_path is not None:
            attestation_path.unlink(missing_ok=True)
        if attestation_directory is not None:
            attestation_directory.rmdir()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        raise
    runtime_deadline = None
    attestation_nonce = None
    attestation_buffer = bytearray()
    child_reported_done = False

    def check_observation(footprint: int, pressure: HostPressure | None) -> str | None:
        if footprint >= args.max_footprint_bytes:
            return (
                f"physical_footprint_at_or_above_{args.max_footprint_bytes}:"
                f"observed_{footprint}"
            )
        if pressure is not None:
            if pressure.memory_free_bytes < args.min_memory_free_bytes:
                return (
                    f"host_memory_free_below_{args.min_memory_free_bytes}:"
                    f"observed_{pressure.memory_free_bytes}"
                )
            if pressure.swap_free_bytes < args.min_swap_free_bytes:
                return (
                    f"host_swap_free_below_{args.min_swap_free_bytes}:"
                    f"observed_{pressure.swap_free_bytes}"
                )
        return None

    def check_initial_observation(
            footprint: int, pressure: HostPressure | None) -> str | None:
        stopped = check_observation(footprint, pressure)
        if stopped is not None or not args.require_child_attestation:
            return stopped
        if pressure is None or attested_initial_memory_free_bytes is None:
            return "child_attestation_initial_host_pressure_was_not_sampled"
        if pressure.memory_free_percent < ATTESTED_INITIAL_MEMORY_FREE_PERCENT:
            return (
                f"initial_host_memory_free_percent_below_"
                f"{ATTESTED_INITIAL_MEMORY_FREE_PERCENT}:observed_"
                f"{pressure.memory_free_percent}"
            )
        if pressure.memory_free_bytes < attested_initial_memory_free_bytes:
            return (
                f"initial_host_memory_free_below_{attested_initial_memory_free_bytes}:"
                f"observed_{pressure.memory_free_bytes}"
            )
        return None

    def emit_sample(footprint: int, pressure: HostPressure | None, phase: str) -> None:
        event: dict[str, object] = {
            "event": "sample", "phase": phase, "physicalFootprintBytes": footprint,
        }
        if pressure is not None:
            event.update({
                "memoryFreePercent": pressure.memory_free_percent,
                "memoryFreeBytes": pressure.memory_free_bytes,
                "swapFreeBytes": pressure.swap_free_bytes,
            })
        emit(args.event_file, event)

    def bounded_telemetry_timeout() -> float:
        if runtime_deadline is None:
            return args.telemetry_timeout
        remaining = runtime_deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"runtime reached {args.max_runtime_seconds}s")
        return min(args.telemetry_timeout, remaining)

    try:
        # A signal pending from the blocked launch window is delivered here, inside the cleanup
        # try, never in the gap between establishing the group and arming cleanup.
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        emit(args.event_file, {"event": "started", "pid": group.child.pid, "pgid": group.pgid})
        try:
            _, footprint, pressure, _ = observe_group(
                group, sampler, host_sampler, args.telemetry_timeout,
            )
        except MonitorSignal:
            raise
        except Exception as error:
            hard_stop = f"initial_telemetry_lost:{type(error).__name__}:{error}"
        if hard_stop is None:
            hard_stop = check_initial_observation(footprint, pressure)
        if hard_stop is None:
            emit_sample(footprint, pressure, "before_child_release")
            group.release()
            runtime_deadline = (
                time.monotonic() + args.max_runtime_seconds
                if args.max_runtime_seconds is not None
                else None
            )
        if hard_stop is None and attestation_listener is not None:
            nonce = secrets.token_hex(32)
            attestation_nonce = nonce
            attestation = {
                "protocol": "sceneworks-memory-watchdog-v1",
                "nonce": nonce,
                "maxFootprintBytes": args.max_footprint_bytes,
                "maxRuntimeSeconds": args.max_runtime_seconds,
                "hostMemoryBytes": args.host_memory_bytes,
                "minInitialMemoryFreeBytes": attested_initial_memory_free_bytes,
                "minInitialMemoryFreePercent": ATTESTED_INITIAL_MEMORY_FREE_PERCENT,
                "minMemoryFreeBytes": args.min_memory_free_bytes,
                "minSwapFreeBytes": args.min_swap_free_bytes,
            }
            try:
                attestation_listener.settimeout(bounded_telemetry_timeout())
                attestation_stream, _ = attestation_listener.accept()
                attestation_stream.settimeout(bounded_telemetry_timeout())
                attestation_stream.sendall((json.dumps(attestation, separators=(",", ":")) + "\n").encode())
                if recv_line(attestation_stream) != f"ACK {nonce}":
                    raise RuntimeError("guarded child returned an invalid watchdog acknowledgement")
                _, footprint, pressure, _ = observe_group(
                    group, sampler, host_sampler, bounded_telemetry_timeout(),
                )
                hard_stop = check_initial_observation(footprint, pressure)
                if hard_stop is None:
                    emit_sample(footprint, pressure, "child_attested_before_allocation")
                    emit(args.event_file, {"event": "child_attested"})
                    attestation_stream.sendall(f"GO {nonce}\n".encode())
                    attestation_stream.setblocking(False)
            except MonitorSignal:
                raise
            except Exception as error:
                hard_stop = f"child_attestation_failed:{type(error).__name__}:{error}"
        while True:
            if hard_stop is not None:
                break
            if runtime_deadline is not None and time.monotonic() >= runtime_deadline:
                hard_stop = f"runtime_at_or_above_{args.max_runtime_seconds}s"
                break
            live = group.refresh()
            status = group.child.poll()
            if status is not None and status < 0:
                hard_stop = f"launch_sentinel_lost:status_{status}"
                break
            if not live:
                if attestation_stream is not None and not child_reported_done:
                    hard_stop = "child_exited_without_completion_attestation"
                    break
                return status if status is not None else 0
            if status is not None:
                # The census preceded poll; normal sentinel cleanup may have completed between
                # those observations. Refresh before treating a positive status as a failure.
                if not group.refresh():
                    if attestation_stream is not None and not child_reported_done:
                        hard_stop = "child_exited_without_completion_attestation"
                        break
                    return status
                hard_stop = f"launch_sentinel_failed_with_live_group:status_{status}"
                break
            try:
                _, footprint, pressure, _ = observe_group(
                    group, sampler, host_sampler, bounded_telemetry_timeout(),
                )
            except MonitorSignal:
                raise
            except Exception as error:  # fail closed on timeout, parse failure, or source loss
                failed_at_or_after_deadline = (
                    runtime_deadline is not None and time.monotonic() >= runtime_deadline
                )
                if not group.refresh() and group.child.poll() is not None:
                    return group.child.returncode
                if failed_at_or_after_deadline:
                    hard_stop = f"runtime_at_or_above_{args.max_runtime_seconds}s"
                else:
                    hard_stop = f"telemetry_lost:{type(error).__name__}:{error}"
                break
            if runtime_deadline is not None and time.monotonic() >= runtime_deadline:
                hard_stop = f"runtime_at_or_above_{args.max_runtime_seconds}s"
                break
            hard_stop = check_observation(footprint, pressure)
            emit_sample(footprint, pressure, "runtime")
            if hard_stop is not None:
                break
            if attestation_stream is not None and not child_reported_done:
                heartbeat = f"PING {attestation_nonce}\n".encode()
                try:
                    if attestation_stream.send(heartbeat) != len(heartbeat):
                        hard_stop = "child_attestation_heartbeat_was_partial"
                        break
                except (BlockingIOError, BrokenPipeError, ConnectionResetError):
                    hard_stop = "child_attestation_channel_lost_before_done"
                    break
                try:
                    chunk = attestation_stream.recv(4096)
                    if not chunk:
                        hard_stop = "child_attestation_channel_lost_before_done"
                        break
                    attestation_buffer.extend(chunk)
                except BlockingIOError:
                    pass
                except ConnectionResetError:
                    hard_stop = "child_attestation_channel_lost_before_done"
                    break
                if len(attestation_buffer) > 4096:
                    hard_stop = "child_completion_attestation_exceeded_size_bound"
                    break
                if b"\n" in attestation_buffer:
                    line, remainder = bytes(attestation_buffer).split(b"\n", 1)
                    if remainder or line.decode() != f"DONE {attestation_nonce}":
                        hard_stop = "child_returned_invalid_completion_attestation"
                        break
                    attestation_stream.setblocking(True)
                    attestation_stream.settimeout(args.telemetry_timeout)
                    attestation_stream.sendall(f"BYE {attestation_nonce}\n".encode())
                    child_reported_done = True
                    emit(args.event_file, {"event": "child_completed"})
            sleep_seconds = args.sample_interval
            if runtime_deadline is not None:
                sleep_seconds = min(sleep_seconds, max(0.0, runtime_deadline - time.monotonic()))
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
    except MonitorSignal as caught:
        hard_stop = f"monitor_signal_{signal.Signals(caught.signum).name}"
        exit_status = 128 + caught.signum
    except BaseException as error:
        if interrupted_signum is not None:
            hard_stop = f"monitor_signal_{signal.Signals(interrupted_signum).name}"
            exit_status = 128 + interrupted_signum
        else:
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
        if attestation_stream is not None:
            attestation_stream.close()
        if attestation_listener is not None:
            attestation_listener.close()
        if attestation_path is not None:
            attestation_path.unlink(missing_ok=True)
        if attestation_directory is not None:
            attestation_directory.rmdir()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    return exit_status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-footprint-bytes", type=int, required=True)
    parser.add_argument("--max-runtime-seconds", type=float)
    parser.add_argument("--host-memory-bytes", type=int)
    parser.add_argument("--min-memory-free-bytes", type=int)
    parser.add_argument("--min-swap-free-bytes", type=int)
    parser.add_argument("--sample-interval", type=float, default=0.25)
    parser.add_argument("--telemetry-timeout", type=float, default=1.0)
    parser.add_argument("--term-grace", type=float, default=0.5)
    parser.add_argument("--event-file", type=Path)
    parser.add_argument("--telemetry-file", type=Path)
    parser.add_argument("--host-pressure-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--allow-synthetic-telemetry", action="store_true")
    parser.add_argument("--require-child-attestation", action="store_true")
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
    if args.max_runtime_seconds is not None and args.max_runtime_seconds <= 0:
        parser.error("--max-runtime-seconds must be positive")
    pressure_values = [args.host_memory_bytes, args.min_memory_free_bytes, args.min_swap_free_bytes]
    if any(value is not None for value in pressure_values) and not all(
            value is not None for value in pressure_values):
        parser.error("host pressure guard requires memory size plus both free-byte floors")
    if args.host_pressure_file and not all(value is not None for value in pressure_values[1:]):
        parser.error("synthetic host pressure requires both free-byte floors")
    for value in pressure_values:
        if value is not None and value <= 0:
            parser.error("host pressure byte values must be positive")
    if args.synthetic_spawn_delay < 0:
        parser.error("--synthetic-spawn-delay must be non-negative")
    if args.require_child_attestation and (
            args.max_runtime_seconds is None or not all(value is not None for value in pressure_values)):
        parser.error("child attestation requires runtime and complete host-pressure bounds")
    if args.require_child_attestation and (
            args.allow_synthetic_telemetry
            or args.telemetry_file is not None
            or args.host_pressure_file is not None
            or args.synthetic_spawn_delay != 0
            or args.synthetic_launch_ready_file is not None):
        parser.error("child attestation requires production Darwin telemetry and launch controls")
    return args


if __name__ == "__main__":
    try:
        if sys.argv[1:2] == ["--anchor"]:
            raise SystemExit(anchor_main())
        if sys.argv[1:2] == ["--sentinel"]:
            raise SystemExit(sentinel_main(
                int(sys.argv[2]), None if sys.argv[3] == "-" else sys.argv[3],
                sys.argv[4:],
            ))
        raise SystemExit(guard(parse_args()))
    except Exception as error:
        print(f"memory calibration watchdog failed closed: {error}", file=sys.stderr)
        raise SystemExit(HARD_STOP_EXIT)
