#!/usr/bin/env python3
"""Fail-closed process-group guard for physical memory calibration.

Defense in depth only: this monitor never makes a safety-refused row admissible. Production Darwin
telemetry is the kernel-maintained phys_footprint from /usr/bin/footprint. Synthetic telemetry is
available only behind an explicit test-only flag.

Fail-closed is about the CEILINGS, not about every sample: a guarded capture spawns and reaps
short-lived helpers constantly, and a tick that fails because one of them raced its own exit — or
because `footprint` took longer than its budget on a 100 GB process — is not evidence that the
group is unobserved.

A hard stop therefore has exactly THREE triggers:

  a. a GOOD sample at or above the footprint ceiling, or below the free-memory/free-swap floor;
  b. loss of the guarded ROOT child from a sample it was enumerated for (`RootTelemetryLost`);
  c. sampler faults that persist for `--telemetry-fault-window` of WALL CLOCK with no good sample.

Every sampler failure path — a per-PID `footprint` timeout, a `footprint` non-zero exit, a parse
failure, the aggregate telemetry deadline, the host free-memory/swap probe, and the `/bin/ps`
process-group census — is routed through that one window (`tolerate_telemetry_fault`). None of them
may escalate on its own: a false hard stop is a process-group SIGKILL through a live Metal command
buffer, which is strictly worse for this host than a late stop.

The census is telemetry, not bookkeeping. `/bin/ps` is a source that can be slow or hang exactly
like `/usr/bin/footprint`, so it carries the same per-probe budget and the same tolerance: a census
that FAILS is an UNKNOWN view (the previous census stands), never evidence that the guarded root is
gone. Only a SUCCESSFUL census that lacks the root, or `child.poll()` reporting the sentinel's exit,
is loss of the guarded process. `monitor_failure` is reserved for an exception that is not a
telemetry source failing at all — a genuine bug in this monitor.
"""

from __future__ import annotations

import argparse
import hashlib
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
# WALL-CLOCK window a run of sampler faults may occupy before the guard declares telemetry lost.
# WHY wall clock and not a tick count: a tick count is a duration only if the cadence is known, and
# under load the cadence collapses — the sc-22738 false positive burned a three-tick tolerance in
# 1.2 seconds and SIGKILLed a 57-minute render that was sitting at 38 GB against a 94.8 GB ceiling.
# WHY 60 s: at a 2 s cadence with a 10 s per-probe budget, one fault costs at most ~12 s, so 60 s
# means the sampler failed on at least five full, independently budgeted attempts. That is a
# telemetry outage; anything shorter is a loaded host.
TELEMETRY_FAULT_WINDOW_SECONDS = 60.0
# Independently budgeted probes inside ONE telemetry tick: the process-group census, the
# `footprint` sample, and the host-pressure sample. Each gets a full `--telemetry-timeout`, so the
# aggregate staleness deadline is this multiple of it. WHY this is not one shared budget: when the
# census and the footprint sample drew from a single `--telemetry-timeout`, a slow census left
# `footprint` an arbitrary residue (0.31 s of a 1 s budget on the sc-22738 capture) and starved the
# host-pressure probe to zero, manufacturing the very faults the window then had to absorb. The
# aggregate is therefore a post-hoc staleness assertion and can never be shorter than one full
# sample.
TELEMETRY_PROBE_BUDGETS = 3
# Seconds between runtime ceiling checks. WHY 2 s: the guard exists to catch a footprint climbing
# toward a multi-GB ceiling, which no render crosses inside one interval, and every tick costs a
# whole-system `ps` census plus a `/usr/bin/footprint` walk of a 100 GB process. The sc-22738
# capture ran that at ~2.5 Hz, which is pure contention on the host being measured.
SAMPLE_INTERVAL_SECONDS = 2.0
# Per-probe telemetry budget. WHY 10 s: `/usr/bin/footprint` walks the VM map of every guarded PID,
# and on a 40-100 GB Metal process under load that measured well past one second. The budget must
# bound a hung probe, not a slow one.
TELEMETRY_TIMEOUT_SECONDS = 10.0
# How long a guarded child may take to connect and attest before it allocates. WHY 60 s: the child
# is a cold interpreter importing a generation framework before it says anything, and this window
# must comfortably exceed one full telemetry tick (`TELEMETRY_PROBE_BUDGETS` x the per-probe
# budget) so a single slow tick can never consume the whole startup allowance.
CHILD_ATTESTATION_TIMEOUT_SECONDS = 60.0
# Per-probe budget for the `/bin/ps` process-group census. WHY it is the telemetry budget and not
# the 1 s it used to hard-code: the census is a whole-system `ps` walk, and on this host under a
# 100 GB Metal render it measured past a second — `bernini:q8:mlx` was SIGKILLed 85 minutes into a
# steady 52 GB render, against a 94.8 GB ceiling, by a census that took longer than 1 s (sc-22738).
# A census is a telemetry source exactly like `/usr/bin/footprint`, so it gets the same budget; the
# budget exists to bound a HUNG probe, not a slow one. `guard` adopts the run's `--telemetry-timeout`
# through `set_census_timeout`; the module default is what the sentinel subprocess and direct
# callers use.
CENSUS_TIMEOUT_SECONDS = TELEMETRY_TIMEOUT_SECONDS
# Exceptions a telemetry SOURCE raises: the `ps` census, `/usr/bin/footprint`, `memory_pressure`,
# `sysctl`, their parsers, and the aggregate staleness deadline. Enumerated so that anything else —
# an AttributeError, a TypeError, a NameError — remains a monitor BUG and still fails closed through
# `monitor_failure` instead of being absorbed by the telemetry tolerance.
TELEMETRY_SOURCE_ERRORS = (
    # `subprocess.TimeoutExpired` (a probe exceeded its budget) and every other failed probe launch
    # or reap.
    subprocess.SubprocessError,
    # `/bin/ps` or `/usr/bin/footprint` could not be executed; `TimeoutError` (the aggregate
    # staleness deadline) is an `OSError` subclass.
    OSError,
    # Non-zero probe exit or malformed payload, raised by the probes themselves. `RootTelemetryLost`
    # is a subclass and is refused before this test is reached.
    RuntimeError,
    # `int()` and `json` parse failures over probe output; `json.JSONDecodeError` is a `ValueError`.
    ValueError,
)
PROVIDER_PHASE_PROTOCOL = "sceneworks-provider-phase-v1"
CAMPAIGN_ENTRY_PROVIDER_PHASES = (
    "common_load",
    "primary_conditioning",
    "primary_denoise",
    "primary_decode",
    "lifecycle_warm_repeat",
    "lifecycle_cancel",
    "lifecycle_cancel_recovery",
    "lifecycle_error",
    "lifecycle_error_recovery",
    "cleanup",
)
BOUNDED_CARRIER_PROVIDER_PHASES = (
    "common_load",
    "primary_conditioning",
    "primary_denoise",
    "primary_decode",
    "cleanup",
)
BOUNDED_CAMPAIGN_ENTRY_PROVIDER_PHASES = (
    "common_load",
    "primary_conditioning",
    "primary_denoise",
    "primary_decode",
    "cleanup",
)
PROVIDER_PHASE_PROFILES = {
    "campaign-entry": CAMPAIGN_ENTRY_PROVIDER_PHASES,
    "bounded-carrier": BOUNDED_CARRIER_PROVIDER_PHASES,
    "bounded-campaign-entry": BOUNDED_CAMPAIGN_ENTRY_PROVIDER_PHASES,
}


@dataclass(frozen=True)
class Identity:
    pid: int
    pgid: int
    state: str = field(compare=False)
    started: str


def set_census_timeout(seconds: float) -> None:
    """Adopt this run's per-probe telemetry budget as the census budget.

    Called once by `guard`. The census functions read the module value at call time, so this
    reaches `identity_is_live`, `OwnedGroup.refresh` and `root_pids` without threading a budget
    through every signature.
    """
    global CENSUS_TIMEOUT_SECONDS
    if seconds <= 0:
        raise RuntimeError("census timeout must be positive")
    CENSUS_TIMEOUT_SECONDS = seconds


def census_budget(timeout: float | None) -> float:
    return CENSUS_TIMEOUT_SECONDS if timeout is None else timeout


def process_identity(pid: int, timeout: float | None = None) -> Identity | None:
    result = subprocess.run(
        ["/bin/ps", "-ww", "-p", str(pid), "-o", "pgid=,state=,lstart="],
        capture_output=True, text=True, timeout=census_budget(timeout), check=False,
    )
    fields = result.stdout.strip().split(None, 2)
    if result.returncode != 0 or len(fields) != 3 or fields[1].startswith("Z"):
        return None
    return Identity(pid, int(fields[0]), fields[1], fields[2])


def group_identities(pgid: int, timeout: float | None = None) -> list[Identity]:
    probe = subprocess.Popen(
        ["/bin/ps", "-ww", "-axo", "pid=,pgid=,state=,lstart="],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    stdout, stderr = probe.communicate(timeout=census_budget(timeout))
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


def parent_pids(timeout: float | None = None) -> dict[int, int]:
    """pid → ppid for every live process, or `{}` when the census is unavailable.

    Used only to resolve the guarded ROOT process (the sentinel's one non-anchor child). An
    unavailable census leaves the root unresolved, which only widens tolerance — it never makes a
    sample admissible that the ceilings would refuse.
    """
    try:
        result = subprocess.run(
            ["/bin/ps", "-axo", "pid=,ppid="], capture_output=True, text=True,
            timeout=census_budget(timeout), check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    if result.returncode != 0:
        return {}
    parents: dict[int, int] = {}
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) == 2:
            try:
                parents[int(fields[0])] = int(fields[1])
            except ValueError:
                continue
    return parents


class RootTelemetryLost(RuntimeError):
    """The guarded root process was enumerated live but absent from its own footprint sample."""


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
    def parse_processes(pids: list[int], payload: object, required: object = ()) -> int:
        """Sum the guarded group's phys_footprint over the PIDs the sample actually described.

        A requested PID absent from the payload EXITED between the group census and the sample —
        compilers, `xcrun`, and other short-lived helpers do this constantly under a real capture.
        Its exit is not lost telemetry: it is dropped here and the next tick re-enumerates the
        group. Absence of a `required` PID (the guarded root) is `RootTelemetryLost`. A payload PID
        that was never requested is outside the enumerated group and remains a hard anomaly, as is
        a payload that describes none of the live PIDs.
        """
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
        extra = sorted(set(observed) - requested)
        if extra:
            raise RuntimeError(f"footprint reported PIDs outside the owned group: extra={extra}")
        lost_required = sorted(set(required) - set(observed))
        if lost_required:
            raise RootTelemetryLost(
                f"footprint lost the guarded root PIDs: missing={lost_required}")
        if not observed:
            raise RuntimeError("footprint described none of the live owned PIDs")
        return sum(observed.values())

    def sample(self, pids: list[int], timeout: float, required: object = ()) -> int:
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
            return self.parse_processes(
                pids, json.loads(Path(output).read_text()), required)
        finally:
            Path(output).unlink(missing_ok=True)


class SyntheticFileSampler:
    def __init__(self, path: Path):
        self.path = path

    def sample(self, pids: list[int], timeout: float, required: object = ()) -> int:
        del pids, timeout, required
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
    # Bypass-constructed instances (tests build the group with `__new__`) read the class default.
    root = None

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
        # This runs `/bin/ps` twice over, so it can time out like any other telemetry probe. Every
        # caller inside the guard loop routes that failure through `tolerate_telemetry_fault`; a
        # raised census is never an empty group.
        #
        # Numeric PGID census is safe only while an exact launch-owned anchor proves the original
        # group still exists. The auxiliary anchor outlives a killed sentinel and cannot exit on
        # TERM/INT, closing the between-censuses descendant race without permitting PGID reuse.
        if any(identity_is_live(anchor) for anchor in self.anchors):
            self.retained.update(group_identities(self.pgid))
        return [identity for identity in self.retained if identity_is_live(identity)]

    def root_pids(self, live: list[Identity]) -> list[int]:
        """The guarded root's PID while it is live: the sentinel's one non-anchor child.

        The root is the guarded command itself. Every other group member is a descendant it may
        spawn and reap at will, so only the root's disappearance from a sample it was enumerated
        for is telemetry loss. Resolved once and then held by exact identity; before the sentinel
        has spawned it, and after it exits, the required set is empty.
        """
        if self.root is None:
            anchors = {item.pid for item in self.anchors}
            parents = parent_pids()
            for identity in live:
                if identity.pid not in anchors and parents.get(identity.pid) == self.leader.pid:
                    self.root = identity
                    break
        return [self.root.pid] if self.root is not None and self.root in live else []

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


class EventChain:
    """Append-only event evidence with deletion/reorder/mutation detection."""

    def __init__(self, event_file: Path | None):
        self.event_file = event_file
        self.sequence = 0
        self.previous_hash = "0" * 64

    def emit(self, event: dict[str, object]) -> None:
        self.sequence += 1
        payload = {
            "eventSequence": self.sequence,
            "previousEventHash": self.previous_hash,
            "at": time.time(),
            **event,
        }
        event_hash = hashlib.sha256(json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True,
        ).encode()).hexdigest()
        line = json.dumps({**payload, "eventHash": event_hash}, separators=(",", ":"))
        self.previous_hash = event_hash
        if self.event_file:
            with self.event_file.open("a") as output:
                output.write(f"{line}\n")
                output.flush()
                os.fsync(output.fileno())
        else:
            print(line, file=sys.stderr, flush=True)


def identity_json(identity: Identity) -> dict[str, object]:
    return {
        "pid": identity.pid,
        "pgid": identity.pgid,
        "started": identity.started,
    }


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
    """One telemetry tick. `timeout` is the PER-PROBE budget, never a budget shared across probes.

    The census, the footprint sample and the host-pressure sample each get the full budget, so a
    slow census can no longer hand `footprint` a residue of it and a slow footprint can no longer
    starve the host-pressure probe to zero. The aggregate is a staleness assertion applied after
    the fact — `TELEMETRY_PROBE_BUDGETS` full budgets, so it is never shorter than one full sample.
    """
    started = time.monotonic()
    live = group.refresh()
    if not live:
        raise RuntimeError("owned group has no live identities")
    footprint = sampler.sample([item.pid for item in live], timeout, group.root_pids(live))
    pressure = host_sampler.sample(timeout) if host_sampler is not None else None
    elapsed = time.monotonic() - started
    aggregate = timeout * TELEMETRY_PROBE_BUDGETS
    if elapsed > aggregate:
        raise TimeoutError(
            f"aggregate telemetry stale after {elapsed:.3f}s of a {aggregate:.3f}s deadline")
    return live, footprint, pressure, elapsed


def guard(args: argparse.Namespace) -> int:
    # The census is a telemetry probe and takes this run's per-probe telemetry budget, not the 1 s
    # that SIGKILLed a healthy 85-minute render (sc-22738).
    set_census_timeout(args.telemetry_timeout)
    events = EventChain(args.event_file)
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
        try:
            # Darwin's sockaddr_un path is capped at 104 bytes. The caller may
            # put TMPDIR on a deliberately long external-volume path for large
            # build artifacts, so keep this tiny, mode-0700 rendezvous under the
            # system's short temporary root.
            short_temp_root = Path("/tmp")
            if not short_temp_root.is_dir():
                short_temp_root = Path(tempfile.gettempdir())
            attestation_directory = Path(tempfile.mkdtemp(
                prefix="sceneworks-watchdog-attestation-", dir=short_temp_root))
            attestation_path = attestation_directory / "watchdog.sock"
            attestation_listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            attestation_listener.bind(str(attestation_path))
            attestation_listener.listen(1)
        except BaseException:
            if attestation_listener is not None:
                attestation_listener.close()
            if attestation_path is not None:
                attestation_path.unlink(missing_ok=True)
            if attestation_directory is not None:
                attestation_directory.rmdir()
            raise
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
    provider_phases = (
        PROVIDER_PHASE_PROFILES[args.provider_phase_profile]
        if args.provider_phase_profile is not None
        else ()
    )
    runtime_deadline = None
    attestation_nonce = None
    attestation_buffer = bytearray()
    child_reported_done = False
    completion_released = False
    telemetry_faults = 0
    # Monotonic time of the FIRST failure of the current unrecovered fault run: the wall-clock
    # anchor the tolerance window is measured from. `None` means the last tick produced a good
    # sample.
    telemetry_fault_since: float | None = None
    # The first failure of an unrecovered fault run, kept so a deadline reached while that run is
    # still open cannot relabel a failure that started before it.
    telemetry_fault_reason: str | None = None
    # Fault run that reached the window, attached to the `hard_stop` event as its evidence.
    telemetry_fault_history: dict[str, object] | None = None
    # The most recent GOOD sample. A tolerated fault does not blank the guard's reading: this stays
    # the current reading for the ceiling test until a new good sample replaces it.
    last_good_footprint: int | None = None
    last_good_pressure: HostPressure | None = None
    provider_phase: dict[str, object] | None = None
    provider_phase_sequence = 0

    def process_attestation_lines() -> str | None:
        nonlocal attestation_buffer, child_reported_done
        nonlocal provider_phase, provider_phase_sequence
        while b"\n" in attestation_buffer:
            line, remainder = bytes(attestation_buffer).split(b"\n", 1)
            attestation_buffer = bytearray(remainder)
            try:
                decoded = line.decode()
            except UnicodeDecodeError:
                return "child_returned_non_utf8_attestation"
            fields = decoded.split()
            if fields[:1] == ["PHASE"]:
                if child_reported_done:
                    return "child_returned_provider_phase_after_completion"
                if not args.require_provider_phases or len(fields) != 4:
                    return "child_returned_invalid_provider_phase"
                if fields[1] != attestation_nonce:
                    return "child_returned_foreign_provider_phase_nonce"
                try:
                    sequence = int(fields[2])
                except ValueError:
                    return "child_returned_non_integer_provider_phase_sequence"
                expected_sequence = provider_phase_sequence + 1
                if sequence != expected_sequence or sequence > len(provider_phases):
                    return (
                        "child_returned_reordered_provider_phase:"
                        f"expected_{expected_sequence}:observed_{sequence}"
                    )
                expected_name = provider_phases[sequence - 1]
                if fields[3] != expected_name:
                    return (
                        "child_returned_invalid_provider_phase_name:"
                        f"expected_{expected_name}:observed_{fields[3]}"
                    )
                provider_phase_sequence = sequence
                provider_phase = {"sequence": sequence, "name": expected_name}
                events.emit({
                    "event": "provider_phase",
                    "providerPhase": provider_phase,
                    "authenticated": True,
                })
                try:
                    assert attestation_stream is not None
                    attestation_stream.setblocking(True)
                    attestation_stream.settimeout(args.telemetry_timeout)
                    attestation_stream.sendall(
                        f"PHASE_ACK {attestation_nonce} {sequence} {expected_name}\n".encode())
                except (OSError, TimeoutError) as error:
                    return f"provider_phase_ack_failed:{type(error).__name__}:{error}"
                finally:
                    if attestation_stream is not None:
                        attestation_stream.setblocking(False)
                continue
            if fields == ["DONE", str(attestation_nonce)]:
                if child_reported_done:
                    return "child_returned_duplicate_completion_attestation"
                if args.require_provider_phases and provider_phase_sequence != len(provider_phases):
                    return (
                        "child_completed_before_provider_phase_sequence:"
                        f"observed_{provider_phase_sequence}"
                    )
                child_reported_done = True
                continue
            return "child_returned_invalid_completion_attestation"
        if len(attestation_buffer) > 4096:
            return "child_attestation_message_exceeded_size_bound"
        return None

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
            if (args.min_swap_free_bytes is not None
                    and pressure.swap_free_bytes < args.min_swap_free_bytes):
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
        if pressure.memory_free_bytes < attested_initial_memory_free_bytes:
            return (
                f"initial_host_memory_free_below_{attested_initial_memory_free_bytes}:"
                f"observed_{pressure.memory_free_bytes}"
            )
        return None

    def emit_sample(footprint: int, pressure: HostPressure | None, phase: str) -> None:
        event: dict[str, object] = {
            "event": "sample", "phase": phase, "physicalFootprintBytes": footprint,
            "providerPhase": provider_phase,
        }
        if pressure is not None:
            event.update({
                "memoryFreePercent": pressure.memory_free_percent,
                "memoryFreeBytes": pressure.memory_free_bytes,
                "swapFreeBytes": pressure.swap_free_bytes,
            })
        events.emit(event)

    def tolerate_telemetry_fault(error: BaseException, phase: str) -> bool:
        """Whether this failed tick is a transient sampling fault rather than telemetry loss.

        THE single escalation decision for every sampler failure path — a per-PID `footprint`
        timeout, a `footprint` non-zero exit, a parse failure, the aggregate telemetry deadline,
        the `/bin/ps` process-group census, and the host free-memory/swap probe all arrive here.
        None of them may escalate on its own, and no one of them may spend a budget the others
        share: only losing the guarded root (`RootTelemetryLost`, which is not a sampling fault but
        the loss of the thing being guarded) or a fault run that occupies
        `--telemetry-fault-window` of WALL CLOCK with no good sample is telemetry loss.

        A tolerated fault leaves the last good sample standing as the current reading, so the
        ceiling test never sees an unknown footprint.

        An exception that is NOT a telemetry source failing is a bug in this monitor, not a fault
        to absorb: it is re-raised here so it reaches the `monitor_failure` handler and still fails
        closed.
        """
        nonlocal telemetry_faults, telemetry_fault_reason
        nonlocal telemetry_fault_since, telemetry_fault_history
        if isinstance(error, RootTelemetryLost):
            return False
        if not isinstance(error, TELEMETRY_SOURCE_ERRORS):
            raise error
        now = time.monotonic()
        telemetry_faults += 1
        if telemetry_fault_since is None:
            telemetry_fault_since = now
            telemetry_fault_reason = f"telemetry_lost:{type(error).__name__}:{error}"
        elapsed = now - telemetry_fault_since
        history: dict[str, object] = {
            "faults": telemetry_faults,
            "elapsedSeconds": round(elapsed, 3),
            "windowSeconds": args.telemetry_fault_window,
            "firstReason": telemetry_fault_reason,
        }
        if elapsed >= args.telemetry_fault_window:
            telemetry_fault_history = history
            return False
        events.emit({
            "event": "telemetry_fault", "phase": phase,
            "consecutiveFaults": telemetry_faults,
            "faultElapsedSeconds": history["elapsedSeconds"],
            "faultWindowSeconds": args.telemetry_fault_window,
            "lastGoodPhysicalFootprintBytes": last_good_footprint,
            "lastGoodMemoryFreeBytes": (
                last_good_pressure.memory_free_bytes if last_good_pressure is not None else None),
            "reason": f"{type(error).__name__}:{error}",
            "providerPhase": provider_phase,
        })
        return True

    def record_good_sample(footprint: int, pressure: HostPressure | None) -> None:
        """A good sample closes any open fault run and becomes the guard's current reading."""
        nonlocal telemetry_faults, telemetry_fault_reason, telemetry_fault_since
        nonlocal last_good_footprint, last_good_pressure
        telemetry_faults = 0
        telemetry_fault_since = None
        telemetry_fault_reason = None
        last_good_footprint = footprint
        last_good_pressure = pressure

    # The most recent SUCCESSFUL census. A census that FAILS leaves this view standing: the guard's
    # picture of the group is the last one it actually took, never an empty group.
    last_live: list[Identity] = sorted(group.retained, key=lambda item: item.pid)

    def observe_census(phase: str) -> tuple[list[Identity], bool, str | None]:
        """The guard loop's group census, routed through the one telemetry tolerance.

        Returns `(view, known, telemetry-loss reason)`. `known` is False when the census failed and
        the fault was tolerated: the previous view is returned, and NO caller may read it as the
        guarded root being gone. Root loss is only ever a SUCCESSFUL census that lacks it, or
        `child.poll()` reporting an exit.
        """
        nonlocal last_live
        try:
            last_live = group.refresh()
        except MonitorSignal:
            raise
        except Exception as error:
            if tolerate_telemetry_fault(error, phase):
                return last_live, False, None
            return last_live, False, f"telemetry_lost:{type(error).__name__}:{error}"
        return last_live, True, None

    def pause_runtime() -> None:
        sleep_seconds = args.sample_interval
        if runtime_deadline is not None:
            sleep_seconds = min(sleep_seconds, max(0.0, runtime_deadline - time.monotonic()))
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

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
        events.emit({
            "event": "started", "pid": group.child.pid, "pgid": group.pgid,
            "providerPhase": provider_phase,
            "processIdentities": [identity_json(item) for item in sorted(
                group.retained, key=lambda item: item.pid)],
        })
        # The pre-release observation takes the SAME tolerance as every other sampler failure path.
        # Nothing is rendering yet, but a `footprint` that timed out once is still not evidence
        # about this host's memory, and refusing a capture on it is the same false negative as
        # killing one.
        while hard_stop is None:
            try:
                _, footprint, pressure, _ = observe_group(
                    group, sampler, host_sampler, args.telemetry_timeout,
                )
            except MonitorSignal:
                raise
            except Exception as error:
                if tolerate_telemetry_fault(error, "before_child_release"):
                    time.sleep(args.sample_interval)
                    continue
                hard_stop = f"initial_telemetry_lost:{type(error).__name__}:{error}"
                break
            record_good_sample(footprint, pressure)
            hard_stop = check_initial_observation(footprint, pressure)
            break
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
                "minMemoryFreeBytes": args.min_memory_free_bytes,
            }
            if args.require_provider_phases:
                attestation.update({
                    "providerPhaseProtocol": PROVIDER_PHASE_PROTOCOL,
                    "providerPhaseProfile": args.provider_phase_profile,
                    "providerPhases": list(provider_phases),
                })
            if args.min_swap_free_bytes is not None:
                attestation["minSwapFreeBytes"] = args.min_swap_free_bytes
            try:
                child_attestation_deadline = (
                    time.monotonic() + args.child_attestation_timeout
                )
                startup_deadline = min(runtime_deadline, child_attestation_deadline)

                def startup_deadline_reason() -> str | None:
                    if time.monotonic() < startup_deadline:
                        return None
                    if runtime_deadline <= child_attestation_deadline:
                        return f"runtime_at_or_above_{args.max_runtime_seconds}s"
                    return (
                        "child_attestation_timeout_at_or_above_"
                        f"{args.child_attestation_timeout}s"
                    )

                def observe_startup() -> tuple[
                        str | None, int | None, HostPressure | None]:
                    stopped = startup_deadline_reason()
                    if stopped is not None:
                        return stopped, None, None
                    status = group.child.poll()
                    if status is not None:
                        if status < 0:
                            return f"launch_sentinel_lost:status_{status}", None, None
                        return (
                            "child_attestation_failed:guarded_child_exited:"
                            f"status_{status}", None, None,
                        )
                    remaining = startup_deadline - time.monotonic()
                    if remaining <= 0:
                        return startup_deadline_reason(), None, None
                    try:
                        _, current_footprint, current_pressure, _ = observe_group(
                            group, sampler, host_sampler,
                            min(
                                args.telemetry_timeout,
                                remaining,
                            ),
                        )
                    except MonitorSignal:
                        raise
                    except Exception as error:
                        failed_at_or_after_startup_deadline = (
                            time.monotonic() >= startup_deadline
                        )
                        status = group.child.poll()
                        if status is not None and status < 0:
                            return f"launch_sentinel_lost:status_{status}", None, None
                        if status is not None:
                            return (
                                "child_attestation_failed:guarded_child_exited:"
                                f"status_{status}", None, None,
                            )
                        if failed_at_or_after_startup_deadline:
                            return startup_deadline_reason(), None, None
                        if tolerate_telemetry_fault(error, "awaiting_child_attestation"):
                            return None, None, None
                        return (
                            "child_attestation_telemetry_lost:"
                            f"{type(error).__name__}:{error}", None, None,
                        )
                    record_good_sample(current_footprint, current_pressure)
                    stopped = startup_deadline_reason()
                    if stopped is None:
                        stopped = check_initial_observation(
                            current_footprint, current_pressure,
                        )
                    return stopped, current_footprint, current_pressure

                def pause_startup() -> None:
                    remaining = startup_deadline - time.monotonic()
                    if remaining > 0:
                        time.sleep(min(args.sample_interval, remaining))

                attestation_listener.setblocking(False)
                while attestation_stream is None and hard_stop is None:
                    try:
                        attestation_stream, _ = attestation_listener.accept()
                        break
                    except BlockingIOError:
                        pass
                    hard_stop, footprint, pressure = observe_startup()
                    if footprint is not None:
                        emit_sample(footprint, pressure, "awaiting_child_attestation")
                    if hard_stop is None:
                        pause_startup()
                if hard_stop is None:
                    remaining = startup_deadline - time.monotonic()
                    if remaining <= 0:
                        hard_stop = startup_deadline_reason()
                    else:
                        attestation_stream.settimeout(min(args.telemetry_timeout, remaining))
                if hard_stop is None:
                    attestation_stream.sendall((
                        json.dumps(attestation, separators=(",", ":")) + "\n"
                    ).encode())
                    attestation_stream.setblocking(False)
                    acknowledgement = bytearray()
                    while hard_stop is None and b"\n" not in acknowledgement:
                        try:
                            chunk = attestation_stream.recv(4096)
                        except BlockingIOError:
                            chunk = None
                        if chunk == b"":
                            hard_stop = (
                                "child_attestation_failed:channel_closed_before_ack"
                            )
                            break
                        if chunk:
                            acknowledgement.extend(chunk)
                            if len(acknowledgement) > 4096:
                                hard_stop = (
                                    "child_attestation_failed:ack_exceeded_size_bound"
                                )
                                break
                        if b"\n" in acknowledgement:
                            line, remainder = bytes(acknowledgement).split(b"\n", 1)
                            if remainder or line.decode() != f"ACK {nonce}":
                                hard_stop = (
                                    "child_attestation_failed:invalid_acknowledgement"
                                )
                            break
                        hard_stop, footprint, pressure = observe_startup()
                        if footprint is not None:
                            emit_sample(footprint, pressure, "awaiting_child_ack")
                        if hard_stop is None:
                            pause_startup()
                if hard_stop is None:
                    hard_stop, footprint, pressure = observe_startup()
                    if hard_stop is None:
                        emit_sample(footprint, pressure, "child_attested_before_allocation")
                        events.emit({
                            "event": "child_attested", "providerPhase": provider_phase,
                        })
                        remaining = startup_deadline - time.monotonic()
                        if remaining <= 0:
                            hard_stop = startup_deadline_reason()
                if hard_stop is None:
                    attestation_stream.settimeout(min(args.telemetry_timeout, remaining))
                    try:
                        attestation_stream.sendall(f"GO {nonce}\n".encode())
                    finally:
                        attestation_stream.setblocking(False)
            except MonitorSignal:
                raise
            except Exception as error:
                if hard_stop is None:
                    hard_stop = (
                        f"child_attestation_failed:{type(error).__name__}:{error}"
                    )
        while True:
            if hard_stop is not None:
                break
            if runtime_deadline is not None and time.monotonic() >= runtime_deadline:
                # An open fault run predates the deadline: the deadline does not relabel it.
                hard_stop = (
                    telemetry_fault_reason
                    or f"runtime_at_or_above_{args.max_runtime_seconds}s"
                )
                break
            live, _, census_lost = observe_census("runtime")
            if census_lost is not None:
                hard_stop = census_lost
                break
            status = group.child.poll()
            if status is not None and status < 0:
                hard_stop = f"launch_sentinel_lost:status_{status}"
                break
            # An empty view can only come from a census that SUCCEEDED: a tolerated census failure
            # returns the previous view, which is never empty while the group exists.
            if not live:
                if attestation_stream is not None and not child_reported_done:
                    hard_stop = "child_exited_without_completion_attestation"
                    break
                return status if status is not None else 0
            if status is not None:
                # The census preceded poll; normal sentinel cleanup may have completed between
                # those observations. Refresh before treating a positive status as a failure — and
                # decide nothing at all on a census that failed, which is an unknown view.
                view, view_known, census_lost = observe_census("runtime")
                if census_lost is not None:
                    hard_stop = census_lost
                    break
                if not view_known:
                    pause_runtime()
                    continue
                if not view:
                    if attestation_stream is not None and not child_reported_done:
                        hard_stop = "child_exited_without_completion_attestation"
                        break
                    return status
                hard_stop = f"launch_sentinel_failed_with_live_group:status_{status}"
                break
            if attestation_stream is not None and not child_reported_done:
                try:
                    while True:
                        chunk = attestation_stream.recv(4096)
                        if not chunk:
                            hard_stop = "child_attestation_channel_lost_before_done"
                            break
                        attestation_buffer.extend(chunk)
                        if len(chunk) < 4096:
                            break
                except BlockingIOError:
                    pass
                except ConnectionResetError:
                    hard_stop = "child_attestation_channel_lost_before_done"
                if hard_stop is None:
                    hard_stop = process_attestation_lines()
                if hard_stop is not None:
                    break
                if args.require_provider_phases and provider_phase is None:
                    time.sleep(min(args.sample_interval, bounded_telemetry_timeout()))
                    continue
            if attestation_stream is not None and child_reported_done and not completion_released:
                attestation_stream.setblocking(True)
                attestation_stream.settimeout(args.telemetry_timeout)
                attestation_stream.sendall(f"BYE {attestation_nonce}\n".encode())
                completion_released = True
                events.emit({
                    "event": "child_completed", "providerPhase": provider_phase,
                })
                attestation_stream.setblocking(False)
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
                # The exit check needs its own census, and that census is telemetry too: a failed one
                # is an unknown view, never proof the group is gone — and never `monitor_failure`,
                # which is what a bare `refresh()` raising inside this handler used to produce.
                view, view_known, census_lost = observe_census("runtime")
                if view_known and not view and group.child.poll() is not None:
                    return group.child.returncode
                if failed_at_or_after_deadline:
                    hard_stop = (
                        telemetry_fault_reason
                        or f"runtime_at_or_above_{args.max_runtime_seconds}s"
                    )
                    break
                if census_lost is not None:
                    hard_stop = census_lost
                    break
                if tolerate_telemetry_fault(error, "runtime"):
                    pause_runtime()
                    continue
                hard_stop = f"telemetry_lost:{type(error).__name__}:{error}"
                break
            record_good_sample(footprint, pressure)
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
                    while True:
                        chunk = attestation_stream.recv(4096)
                        if not chunk:
                            hard_stop = "child_attestation_channel_lost_before_done"
                            break
                        attestation_buffer.extend(chunk)
                        if len(chunk) < 4096:
                            break
                except BlockingIOError:
                    pass
                except ConnectionResetError:
                    hard_stop = "child_attestation_channel_lost_before_done"
                    break
                if hard_stop is None:
                    hard_stop = process_attestation_lines()
                if hard_stop is not None:
                    break
                if child_reported_done and not completion_released:
                    attestation_stream.setblocking(True)
                    attestation_stream.settimeout(args.telemetry_timeout)
                    attestation_stream.sendall(f"BYE {attestation_nonce}\n".encode())
                    completion_released = True
                    events.emit({
                        "event": "child_completed", "providerPhase": provider_phase,
                    })
                    attestation_stream.setblocking(False)
            pause_runtime()
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
                events.emit({
                    "event": "hard_stop", "reason": hard_stop,
                    "providerPhase": provider_phase,
                    # Present only when a fault run reached the window: the evidence that this stop
                    # is telemetry loss and not a single unlucky probe.
                    "telemetryFaultHistory": telemetry_fault_history,
                    "processIdentities": [identity_json(item) for item in sorted(
                        group.retained, key=lambda item: item.pid)],
                })
            except Exception:
                pass
            group.terminate(args.term_grace)
            try:
                events.emit({
                    "event": "terminated", "reason": hard_stop,
                    "providerPhase": provider_phase,
                    "processIdentities": [identity_json(item) for item in sorted(
                        group.retained, key=lambda item: item.pid)],
                })
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
    parser.add_argument("--sample-interval", type=float, default=SAMPLE_INTERVAL_SECONDS)
    parser.add_argument("--telemetry-timeout", type=float, default=TELEMETRY_TIMEOUT_SECONDS)
    parser.add_argument(
        "--telemetry-fault-window", type=float, default=TELEMETRY_FAULT_WINDOW_SECONDS)
    parser.add_argument(
        "--child-attestation-timeout", type=float,
        default=CHILD_ATTESTATION_TIMEOUT_SECONDS)
    parser.add_argument("--term-grace", type=float, default=0.5)
    parser.add_argument("--event-file", type=Path)
    parser.add_argument("--telemetry-file", type=Path)
    parser.add_argument("--host-pressure-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--allow-synthetic-telemetry", action="store_true")
    parser.add_argument("--require-child-attestation", action="store_true")
    parser.add_argument("--require-provider-phases", action="store_true")
    parser.add_argument("--provider-phase-profile", choices=tuple(PROVIDER_PHASE_PROFILES))
    parser.add_argument("--synthetic-spawn-delay", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--synthetic-launch-ready-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command:
        parser.error("a guarded command is required after --")
    for name in [
            "max_footprint_bytes", "sample_interval", "telemetry_timeout",
            "telemetry_fault_window", "child_attestation_timeout", "term_grace"]:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_runtime_seconds is not None and args.max_runtime_seconds <= 0:
        parser.error("--max-runtime-seconds must be positive")
    pressure_values = [args.host_memory_bytes, args.min_memory_free_bytes]
    if any(value is not None for value in pressure_values) and not all(
            value is not None for value in pressure_values):
        parser.error("host pressure guard requires memory size plus a free-memory floor")
    if args.min_swap_free_bytes is not None and not all(
            value is not None for value in pressure_values):
        parser.error("a swap floor requires the complete host-pressure guard")
    if args.host_pressure_file and args.min_memory_free_bytes is None:
        parser.error("synthetic host pressure requires a free-memory floor")
    for value in [*pressure_values, args.min_swap_free_bytes]:
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
    if args.require_provider_phases and not args.require_child_attestation:
        parser.error("provider phases require child attestation")
    if args.require_provider_phases != (args.provider_phase_profile is not None):
        parser.error("provider phases require exactly one named phase profile")
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
