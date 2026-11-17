"""Cooperative exact-UUID CUDA lease and owned Windows child boundary.

No actor scheduler, any-idle fallback, model proof, or new memory cap. Caller
budgets/argv/device semantics remain explicit. Metadata probes create no contexts.
"""
from __future__ import annotations
import argparse
import csv
from contextlib import nullcontext
import hashlib
import importlib.util
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from urllib.parse import urlsplit

ACTIVE_KEY = "SW_CI_CUDA_EXECUTION_ACTIVE"

PIN = "91ffb4a819aa2990c652db2d881ba0b9b10126ce"
GUARD_SHA = "301f0d0dd18b55aa9eaf5f3ffebac8122004f7bde59aaac4e7c5103e3451bf3f"
WINDOWS_SHA = "b50647bb1566df498f8cf4b57fa9288278e8ee1301de37afec3da478783123dd"


# The ONE admission rule, for every actor (rule b8b59242): with the exact-UUID lock held, admit the
# card when a short window shows enough FREE memory for the job and a quiet, stable device. Nothing
# else is required: the card may drive a display, and other (non-CI) processes such as the desktop
# app may hold CUDA contexts or loaded models. CI jobs never share a card with each other: the lock
# is held from admission until the owned child tree has drained.
SHARED_POLICY = "shared-free-memory"
# Free memory a job needs when its workflow does not say (--min-free-mib). A floor that refuses a
# nearly full card, not a measured per-job need; a job that knows its peak passes it.
DEFAULT_MIN_FREE_MIB = 16384
SHARED_WINDOW_SAMPLES = 10            # one inventory per second after the UUID lock is held
SHARED_WINDOW_INTERVAL_SECONDS = 1.0
SHARED_MAX_SPREAD_MIB = 256           # max - min device memory.used across the window
SHARED_MAX_GROWTH_MIB = 128           # last - first device memory.used across the window
SHARED_MAX_UTILIZATION_PERCENT = 30   # an idle-holding model or display composition stays far below this
SHARED_MAX_UTILIZATION_SPREAD = 15
# A wait prints a line when it starts and then this often, so a long admission wait is never silent.
WAIT_REPORT_INTERVAL_SECONDS = 60


class WaitReporter:
    """Prints each admission wait to the job log (stderr, flushed): start, every interval, end.

    It only reports; it never changes what is admitted. `state` is the last wait seen, for a cancel.
    Lines carry device index/UUID, memory and process identities only, never environment values.
    """
    def __init__(self, stream=None, interval=WAIT_REPORT_INTERVAL_SECONDS):
        self.stream, self.interval = stream, interval
        self.reset()

    def reset(self):
        self.wait, self.last_print, self.reported, self.state, self.seen = None, None, False, None, set()

    def emit(self, event, fields):
        try:
            print("cuda-execution: " + json.dumps({"event": event, **fields}, default=str),
                  file=self.stream or sys.stderr, flush=True)
        except Exception:  # a closed or broken log stream must never change admission
            pass

    def waiting(self, wait, now, started, deadline, devices, detail=dict):
        state = {"wait": wait, "elapsed_seconds": round(now - started, 1),
                 "remaining_budget_seconds": round(max(0, deadline - now), 1),
                 "devices": [{"index": row.get("index"), "uuid": row.get("uuid")} for row in devices]}
        # A new kind of wait prints at once; after that one line per interval, whatever the kind.
        changed = wait not in self.seen
        if changed or now - self.last_print >= self.interval:
            try:
                state.update(detail())
            except Exception as error:  # reporting must never break admission
                state["detail_error"] = f"{type(error).__name__}: {error}"
            self.emit("wait-start" if changed else "waiting", state)
            self.wait, self.last_print, self.reported = wait, now, True
            self.seen.add(wait)
        elif self.state:
            state = {**self.state, **{key: state[key] for key in ("elapsed_seconds", "remaining_budget_seconds")}}
        self.state = state

    def finished(self, scope, waited, devices=()):
        if self.reported:
            self.emit("wait-end", {"scope": scope, "wait_seconds": round(waited, 1),
                                   "devices": [{"index": row.get("index"), "uuid": row.get("uuid")} for row in devices]})
        self.reset()


class DeviceIdentity:
    """Only physical index/UUID inventory; never process, utilization or memory census."""
    def __init__(self, executable, run=subprocess.run):
        self.executable, self.run = executable, run

    def inventory(self, out, label, deadline):
        remaining = min(30, deadline - time.monotonic())
        if remaining <= 0:
            raise TimeoutError("device identity deadline exhausted")
        result = self.run([self.executable, "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True, check=True, timeout=remaining)
        devices = []
        for row in csv.reader(result.stdout.splitlines()):
            if len(row) != 2 or not re.fullmatch(r"[0-9]+", row[0].strip()) or not re.fullmatch(r"GPU-[A-Za-z0-9-]+", row[1].strip()):
                raise RuntimeError("invalid physical device identity")
            devices.append({"index": int(row[0].strip()), "uuid": row[1].strip()})
        if (not devices or len({d["index"] for d in devices}) != len(devices)
                or len({d["uuid"].casefold() for d in devices}) != len(devices)):
            raise RuntimeError("ambiguous physical device identity")
        return devices, []


def require_physical_index(binding, index, environment=None, probe_visible=None):
    """Refuse unless every bound device is physical GPU `index` (the workflow's GPU restriction).

    A `--probe-visible` binding is probed with that mask, but the child inherits the job's own
    CUDA_VISIBLE_DEVICES, so the two must agree or the lock would guard a card the child never uses.
    """
    if index is None:
        return
    if probe_visible is not None and (environment or {}).get("CUDA_VISIBLE_DEVICES") != probe_visible:
        raise RuntimeError(f"--probe-visible {probe_visible} differs from the job's CUDA_VISIBLE_DEVICES; "
                           "the child would not run on the card this lock guards")
    found = [row.get("index") for row in binding.get("devices", [])]
    if not found or any(value != index for value in found):
        raise RuntimeError(f"bound CUDA device(s) are physical GPU {found}, not the required GPU {index}; "
                           "check CUDA_DEVICE_ORDER=PCI_BUS_ID and CUDA_VISIBLE_DEVICES")


def gpu_identities(xml_text):
    """Every GPU's full driver identity for the receipt: bus, UUID, name, display, memory, processes."""
    import xml.etree.ElementTree as ET
    rows = []
    for gpu in ET.fromstring(xml_text).findall("gpu"):
        text = lambda path: (gpu.findtext(path) or "").strip()
        rows.append({"pci_bus_id": gpu.get("id"), "uuid": text("uuid"), "name": text("product_name"),
                     "display_attached": text("display_attached"), "display_active": text("display_active"),
                     "display_mode": text("display_mode"), "driver_model": text("driver_model/current_dm"),
                     "memory_total": text("fb_memory_usage/total"), "memory_used": text("fb_memory_usage/used"),
                     "memory_free": text("fb_memory_usage/free"), "utilization": text("utilization/gpu_util"),
                     "processes": [{"pid": (p.findtext("pid") or "").strip(), "type": (p.findtext("type") or "").strip(),
                                    "name": (p.findtext("process_name") or "").strip(),
                                    "used_memory": (p.findtext("used_memory") or "").strip()}
                                   for p in gpu.findall("processes/process_info")]})
    return rows


def shared_decision(samples, uuid, min_free_mib):
    """Pure admission rule over a window of (devices, processes) samples of one leased UUID.

    Admits on free memory and stability alone. Other processes on the card -- display/graphics
    contexts, a desktop app's compute context or loaded model -- are recorded, never refused: on WDDM
    their per-process memory is N/A, and what the job needs is the card's free memory, which already
    accounts for them. Refuses only too little free memory, a moving memory.used, or a busy device.
    """
    reasons = []
    rows = []
    for devices, processes in samples:
        matches = [row for row in devices if row["uuid"].casefold() == uuid.casefold()]
        if len(matches) != 1:
            return False, ["leased GPU UUID missing from a sample"], {}
        rows.append((matches[0], [row for row in processes if row[0].casefold() == uuid.casefold()]))
    if len(rows) < 2:
        return False, ["sampling window too short"], {}
    used = [device["used_mib"] for device, _ in rows]
    utilization = [device["utilization_percent"] for device, _ in rows]
    last = rows[-1][0]
    free = last["total_mib"] - last["used_mib"]
    summary = {"samples": len(rows), "index": last.get("index"), "uuid": last.get("uuid"),
               "total_mib": last["total_mib"], "baseline_used_mib": max(used), "min_used_mib": min(used),
               "free_mib": free, "min_free_mib": min_free_mib, "max_utilization_percent": max(utilization),
               "co_tenants": sorted({(row[1], row[2] if len(row) > 2 else "unknown",
                                      row[3] if len(row) > 3 else "unknown")
                                     for _, processes in rows for row in processes})}
    if free < min_free_mib:
        reasons.append(f"free {free} MiB < required {min_free_mib} MiB")
    if max(used) - min(used) > SHARED_MAX_SPREAD_MIB:
        reasons.append(f"memory.used unstable: spread {max(used) - min(used)} MiB > {SHARED_MAX_SPREAD_MIB}")
    if used[-1] - used[0] > SHARED_MAX_GROWTH_MIB:
        reasons.append(f"memory.used growing: +{used[-1] - used[0]} MiB over the window")
    if max(utilization) > SHARED_MAX_UTILIZATION_PERCENT:
        reasons.append(f"utilization {max(utilization)}% > {SHARED_MAX_UTILIZATION_PERCENT}%")
    if max(utilization) - min(utilization) > SHARED_MAX_UTILIZATION_SPREAD:
        reasons.append(f"utilization unstable: spread {max(utilization) - min(utilization)} points")
    return not reasons, sorted(set(reasons)), summary


def sample_shared_window(guard, out, deadline, uuid, min_free_mib, phase, clock=time.monotonic, sleep=time.sleep):
    samples = []
    for index in range(SHARED_WINDOW_SAMPLES):
        if index:
            sleep(SHARED_WINDOW_INTERVAL_SECONDS)
        label = f"{phase}-shared-{index + 1}"
        samples.append(guard.inventory(out, label, deadline))
    admitted, reasons, summary = shared_decision(samples, uuid, min_free_mib)
    (out / f"{phase}-shared-window.json").write_text(json.dumps(
        {"admitted": admitted, "reasons": reasons, "summary": summary}, indent=2, default=list),
        encoding="utf-8", newline="\n")
    return admitted, reasons, summary


def resource_receipt(actor, command):
    return {"schema_version": 1, "kind": "cuda-execution-resource-lease-v1", "model_inference_proof": False,
            "resource_completed": False, "actor": actor, "argv": command,
            "admission_policy": SHARED_POLICY, "shared_device": True, "isolated_device": False,
            "post_child_idle_verified": False,
            "observation_note": ("shared GPU: other non-CI processes may hold memory on the card; a memory "
                                 "figure is attributable only as a delta above a baseline taken under this lock"),
            "guard_source_revision": PIN, "guard_sha256": GUARD_SHA, "windows_job_sha256": WINDOWS_SHA}


def post_child_sample(guard, out, devices, baselines, clock=time.monotonic):
    """One inventory after the owned child drained, still under the lock: how far the card's
    memory.used moved from the admission baseline over the run (diagnostic; never a refusal)."""
    try:
        current, _ = guard.inventory(out, "after-child", clock() + 30)
    except Exception as error:  # a diagnostic must never change the child's verdict
        return {"error": f"{type(error).__name__}: {error}"}
    rows = []
    for device in devices:
        matches = [row for row in current if row["uuid"].casefold() == device["uuid"].casefold()]
        baseline = (baselines.get(device["uuid"]) or {}).get("baseline_used_mib")
        used = matches[0]["used_mib"] if len(matches) == 1 else None
        rows.append({"index": device.get("index"), "uuid": device["uuid"], "used_mib": used,
                     "admission_baseline_used_mib": baseline,
                     "drift_mib": used - baseline if used is not None and baseline is not None else None})
    return {"devices": rows}


def complete_resource(receipt, result, devices, guard, out, baselines):
    receipt["post_child"] = post_child_sample(guard, out, devices, baselines)
    receipt["resource_completed"] = (result["returncode"] == 0 and result["timeout"] is False
                                     and result.get("owned_tree_empty") is True)
    return result["returncode"] if result["returncode"] is not None else 124


def component(name: str, expected: str):
    file = Path(__file__).resolve().parent / "vendor" / name
    if hashlib.sha256(file.read_bytes()).hexdigest() != expected:
        raise RuntimeError(f"pinned CUDA ownership component changed: {name}")
    spec = importlib.util.spec_from_file_location("nvfp4_" + file.stem, file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def unlock(handle, locker):
    try:
        handle.seek(0)
        locker.locking(handle.fileno(), locker.LK_UNLCK, 1)
    finally:
        handle.close()


def execute_owned(args, out, deadline, job_type, env,
                  popen=subprocess.Popen, clock=time.monotonic, sleep=time.sleep, capture=True):
    """Resume only inside the non-breakaway Job and join every descendant."""
    job = job_type()
    child = None
    assigned = False
    started = clock()
    child_environment = env.copy()
    child_environment[ACTIVE_KEY] = str(os.getpid())
    try:
        stdout_scope = (out / "proof.stdout.log").open("wb") if capture else nullcontext()
        stderr_scope = (out / "proof.stderr.log").open("wb") if capture else nullcontext()
        with stdout_scope as stdout, stderr_scope as stderr:
            child = popen(args, stdout=stdout, stderr=stderr, env=child_environment,
                          creationflags=0x00000004)  # CREATE_SUSPENDED
            try:
                job.assign_and_resume(child)
                assigned = True
            except BaseException:
                child.kill()  # no code/descendant has run before assignment/resume
                child.wait(timeout=10)
                raise
            remaining = deadline - clock()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(args, 0)
            code = child.wait(timeout=remaining)
            while job.members():
                if clock() >= deadline:
                    raise subprocess.TimeoutExpired(args, clock() - started)
                sleep(0.05)
            return {"returncode": code, "timeout": False, "seconds": clock() - started,
                    "owned_tree_empty": True}
    except subprocess.TimeoutExpired:
        job.terminate_and_reap(child, 10)
        return {"returncode": None, "timeout": True, "seconds": clock() - started,
                "owned_tree_empty": True}
    except BaseException:
        if assigned and child is not None and job.members():
            job.terminate_and_reap(child, 10)
        raise
    finally:
        job.close()


def release_all(handles, locker):
    first_error = None
    for handle in reversed(handles):
        try:
            unlock(handle, locker)
        except BaseException as error:
            first_error = first_error or error
    if first_error is not None:
        raise first_error


class AdmissionRefused(RuntimeError):
    def __init__(self, device, reasons, summary):
        super().__init__(f'GPU {device.get("index")} not admitted: ' + "; ".join(reasons))
        self.device, self.reasons, self.summary = device, reasons, summary


def claim_uuids(uuids, guard, out, deadline, public, locker,
                required_indices=None, phase_names=None, clock=time.monotonic, sleep=time.sleep,
                min_free_mib=DEFAULT_MIN_FREE_MIB, reporter=None):
    """Hold every UUID's host-global lock, then admit on the shared free-memory/stability window.

    A refusal releases the locks (so another CI job is never starved behind a card that cannot fit
    this one), reports the reason through `reporter`, and retries until the caller's deadline.
    """
    reporter = reporter or WaitReporter()
    reporter.reset()
    expected = sorted(uuids, key=str.casefold)
    if not isinstance(min_free_mib, int) or not 0 < min_free_mib < 1 << 20:
        raise RuntimeError("admission needs a positive free-memory requirement in MiB")
    if not expected or len({value.casefold() for value in expected}) != len(expected):
        raise RuntimeError("explicit unique CUDA UUID set is required")
    if not public.is_absolute():
        raise RuntimeError("PUBLIC must name the host-global absolute lock namespace")
    started = clock()
    while clock() < deadline:
        try:
            devices, _ = guard.inventory(out, "uuid-admission", deadline)
        except RuntimeError as error:  # a transient driver inventory error: log it and retry
            reporter.waiting("admission", clock(), started, deadline, [],
                             lambda: {"refused": {"reasons": [str(error)]}})
            sleep(min(1, max(0, deadline - clock())))
            continue
        selected = []
        for identifier in expected:
            matches = [row for row in devices if row["uuid"].casefold() == identifier.casefold()]
            if len(matches) != 1:
                raise RuntimeError("declared GPU UUID changed or disappeared; no device fallback")
            device = matches[0]
            if required_indices and device["index"] != required_indices[identifier]:
                raise RuntimeError("declared physical GPU index/UUID binding changed")
            if not re.fullmatch(r"GPU-[A-Za-z0-9-]+", device["uuid"]):
                raise RuntimeError("unrecognized physical GPU UUID")
            selected.append(device)
        handles = []
        busy = None
        directory = public / "SceneWorks" / "gpu-locks"
        directory.mkdir(parents=True, exist_ok=True)
        handle = None
        try:
            for device in selected:
                handle = (directory / f'{device["uuid"]}.lock').open("a+b")
                if handle.seek(0, 2) == 0:
                    handle.write(b"0")
                    handle.flush()
                handle.seek(0)
                try:
                    locker.locking(handle.fileno(), locker.LK_NBLCK, 1)
                except OSError:
                    handle.close()
                    busy = device
                    break
                handles.append(handle)
        except BaseException:
            if handle is not None and handle not in handles:
                handle.close()
            release_all(handles, locker)
            raise
        if busy is not None:
            release_all(handles, locker)
            reporter.waiting("lock", clock(), started, deadline, selected,
                             lambda: {"held_lock": {"index": busy.get("index"), "uuid": busy.get("uuid"),
                                                    "path": str(directory / f'{busy["uuid"]}.lock')}})
            sleep(min(1, max(0, deadline - clock())))
            continue
        try:
            baselines = {}
            for device in selected:
                phase = (phase_names or {}).get(device["uuid"], f'claimed-device-{device["index"]}')
                admitted, reasons, summary = sample_shared_window(guard, out, deadline, device["uuid"],
                                                                  min_free_mib, phase, clock, sleep)
                if not admitted:
                    raise AdmissionRefused(device, reasons, summary)
                baselines[device["uuid"]] = summary
            reporter.finished("gpu", clock() - started, selected)
            return selected, handles, baselines, clock() - started
        except RuntimeError as error:  # a refusal, or a transient driver inventory error: retry
            release_all(handles, locker)
            refused = getattr(error, "device", {})
            reporter.waiting("admission", clock(), started, deadline, selected,
                             lambda: {"refused": {"index": refused.get("index"), "uuid": refused.get("uuid"),
                                                  "reasons": getattr(error, "reasons", [str(error)]),
                                                  "window": getattr(error, "summary", None)}})
            sleep(min(1, max(0, deadline - clock())))
        except BaseException:
            release_all(handles, locker)
            raise
    raise TimeoutError("no admission within the budget: the card never held the lock with enough free "
                       "memory and a stable window (the last refusal reason is in the wait log)")


def integers(value):
    if not re.fullmatch(r"[0-9]+(?:,[0-9]+)*", value or ""):
        raise RuntimeError("explicit nonnegative CUDA ordinal/index list is required")
    result = [int(part) for part in value.split(",")]
    if len(set(result)) != len(result):
        raise RuntimeError("duplicate CUDA selection")
    return result


def resolve_binding(guard, out, deadline, environment, physical_indices=None,
                    cuda_ordinals="0", probe_visible=None, run=subprocess.run):
    devices, _ = guard.inventory(out, "binding", deadline)
    if physical_indices is not None:
        selected = []
        for index in integers(physical_indices):
            matches = [row for row in devices if row["index"] == index]
            if len(matches) != 1:
                raise RuntimeError("actor's explicit NVML-to-UUID binding is unavailable")
            selected.append(matches[0])
        return {"kind": "actor-forced-nvml-uuid", "devices": selected,
                "key": [(row["index"], row["uuid"]) for row in selected]}
    probe_env = environment.copy()
    if probe_visible is not None:
        probe_env["CUDA_VISIBLE_DEVICES"] = probe_visible
    remaining = min(30, deadline - time.monotonic())
    if remaining <= 0:
        raise TimeoutError("CUDA metadata mapping deadline exhausted")
    result = run([sys.executable, "-I", str(Path(__file__).with_name("windows_cuda_metadata.py"))],
                 env=probe_env, capture_output=True, text=True, timeout=remaining, check=True)
    packet = json.loads(result.stdout)
    if (packet.get("context_created") is not False
            or packet.get("abi_source", {}).get("checksum") != "1cea5f10a99e025c1b44ae2354c2d8326b25ddbd0baf76bde8e55cfd4018a2cc"):
        raise RuntimeError("unverified context-free CUDA ordinal metadata")
    visible = packet.get("devices", [])
    ordinals = [row["ordinal"] for row in visible] if cuda_ordinals == "all" else integers(cuda_ordinals)
    if not ordinals:
        raise RuntimeError("no declared CUDA-visible device; never skip/pass")
    selected = []
    mapping = []
    for ordinal in ordinals:
        logical = [row for row in visible if row["ordinal"] == ordinal]
        if len(logical) != 1:
            raise RuntimeError("declared actual CUDA ordinal is unavailable")
        matches = [row for row in devices if row["uuid"].casefold() == logical[0]["uuid"].casefold()]
        if len(matches) != 1:
            raise RuntimeError("actual CUDA ordinal UUID cannot be matched to NVML")
        selected.append(matches[0])
        mapping.append({"cuda_ordinal": ordinal, "uuid": matches[0]["uuid"], "nvml_index": matches[0]["index"]})
    if len({row["uuid"].casefold() for row in selected}) != len(selected):
        raise RuntimeError("actual CUDA ordinal mapping is ambiguous")
    return {"kind": "context-free-cuda-ordinal-uuid", "devices": selected,
            "probe": packet, "mapping": mapping,
            "key": [(row["cuda_ordinal"], row["uuid"]) for row in mapping]}


def prepared_components():
    guard = component("veda_cuda_guard.py", GUARD_SHA)
    windows = component("media_campaign_windows.py", WINDOWS_SHA)
    trusted = windows.trusted_nvidia_smi()
    original = guard.command
    def command(args, deadline=None):
        if args[0] != "nvidia-smi":
            raise RuntimeError("unexpected physical inventory command")
        return original([trusted, *args[1:]], deadline)
    guard.command = command
    return guard, windows


def require_outermost(environment):
    # This marker is only a refusal signal, never authority to bypass/reuse a lock.
    if ACTIVE_KEY in environment:
        raise RuntimeError("nested CUDA execution boundary refused; wrap the closed transaction exactly once")


def service_identity(url):
    parsed = urlsplit(url)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment):
        raise RuntimeError("an actual credential-free loopback service URL is required")
    host = parsed.hostname.casefold()
    try:
        loopback = host == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = False
    if not loopback or parsed.port is None or not 0 < parsed.port < 65536:
        raise RuntimeError("actual loopback service port must be explicit; no default/port substitution")
    return {"url": url, "port": parsed.port}


def claim_service(url, public, deadline, locker, clock=time.monotonic, sleep=time.sleep, reporter=None):
    reporter = reporter or WaitReporter()
    reporter.reset()
    identity = service_identity(url)
    if not public.is_absolute():
        raise RuntimeError("PUBLIC must name the host-global absolute service namespace")
    directory = public / "SceneWorks" / "service-locks"
    directory.mkdir(parents=True, exist_ok=True)
    file = directory / f'loopback-port-{identity["port"]}.lock'
    started = clock()
    while clock() < deadline:
        handle = file.open("a+b")
        try:
            if handle.seek(0, 2) == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            locker.locking(handle.fileno(), locker.LK_NBLCK, 1)
        except OSError:
            handle.close()
            reporter.waiting("service", clock(), started, deadline, (),
                             lambda: {"port": identity["port"], "lock_path": str(file)})
            sleep(min(1, max(0, deadline - clock())))
            continue
        except BaseException:
            handle.close()
            raise
        reporter.finished("service", clock() - started)
        return handle, {**identity, "lock_path": str(file), "wait_seconds": clock() - started}
    raise TimeoutError("host service port transaction wait exhausted the existing execution budget")


def release_transaction(gpu_handles, service_handle, locker):
    try:
        release_all(gpu_handles, locker)
    finally:
        if service_handle is not None:
            unlock(service_handle, locker)


def cancel_handler(receipt, reporter):
    """On a cancel signal, print the current wait and the partial receipt, then unwind once through
    main's existing release/receipt path. Later signals, and any signal after `disarm()` (main calls it
    as soon as its body exits), only print: they never interrupt lock release or the receipt."""
    seen = []
    def handle(signum, _frame):
        partial = {key: value for key, value in receipt.items() if key != "argv"}
        try:
            name = signal.Signals(signum).name
        except ValueError:
            name = str(signum)
        reporter.emit("cancel", {"signal": name, "wait": reporter.state, "partial_receipt": partial})
        if not seen:
            seen.append(signum)
            raise KeyboardInterrupt(f"cancelled by signal {signum}")
    handle.disarm = lambda: seen.append(None)
    return handle


def install_cancel_handler(handler, signals=signal):
    """Install `handler` for SIGTERM/SIGINT/SIGBREAK (Actions cancel on Windows); return the previous ones."""
    previous = {}
    for name in ("SIGTERM", "SIGINT", "SIGBREAK"):
        number = getattr(signals, name, None)
        if number is not None:
            try:
                previous[number] = signals.signal(number, handler)
            except (OSError, ValueError):
                # Signal not supported or unavailable here (e.g. off the main thread); try the next one.
                continue
    return previous


def restore_signal_handlers(previous, signals=signal):
    for number, handler in previous.items():
        try:
            signals.signal(number, handler)
        except (OSError, ValueError, TypeError):
            # Best-effort restore: the previous handler or the signal may no longer be valid.
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--budget-minutes", type=float, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--physical-indices")
    modes.add_argument("--cuda-ordinals", default="0")
    parser.add_argument("--probe-visible")
    parser.add_argument("--min-free-mib", type=int, default=DEFAULT_MIN_FREE_MIB,
                        help=f"free memory (MiB) the job needs on each held card; default {DEFAULT_MIN_FREE_MIB}")
    parser.add_argument("--require-physical-index", type=int,
                        help="refuse unless every bound device is this physical (NVML, PCI bus order) GPU index")
    parser.add_argument("--service-url", help="actual loopback service URL; optional host transaction lock")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if (os.name != "nt" or not re.fullmatch(r"[A-Za-z0-9_.-]+", args.actor)
            or not math.isfinite(args.budget_minutes) or args.budget_minutes <= 0):
        raise RuntimeError("explicit Windows actor/budget is required")
    environment = os.environ.copy()
    require_outermost(environment)
    if not 0 < args.min_free_mib < 1 << 20:
        raise RuntimeError("--min-free-mib must be a positive MiB count")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise RuntimeError("original actor argv is required")
    run_id, attempt = os.environ["GITHUB_RUN_ID"], os.environ["GITHUB_RUN_ATTEMPT"]
    if not run_id.isdecimal() or not attempt.isdecimal():
        raise RuntimeError("actual workflow run/attempt identity is required")
    temporary_root = Path(os.environ["RUNNER_TEMP"])
    if not temporary_root.is_absolute():
        raise RuntimeError("RUNNER_TEMP must be the actual absolute owned workflow directory")
    out = temporary_root / f"sceneworks-cuda-execution-{run_id}-{attempt}-{args.actor}"
    out.mkdir(parents=True, exist_ok=False)
    import msvcrt
    deadline = time.monotonic() + args.budget_minutes * 60
    receipt = resource_receipt(args.actor, command)
    reporter = WaitReporter()
    cancel = cancel_handler(receipt, reporter)
    previous_handlers = install_cancel_handler(cancel)
    handles = []
    service_handle = None
    code = 1
    try:
        if args.service_url is not None:
            service_handle, receipt["service_transaction"] = claim_service(
                args.service_url, Path(os.environ["PUBLIC"]), deadline, msvcrt, reporter=reporter)
        guard, windows = prepared_components()
        binding = resolve_binding(guard, out, deadline, environment, args.physical_indices,
                                  args.cuda_ordinals, args.probe_visible)
        receipt["binding"] = binding
        require_physical_index(binding, args.require_physical_index, environment, args.probe_visible)
        xml_file = out / "binding-nvidia-smi.xml"
        if xml_file.exists():
            receipt["gpu_identities"] = gpu_identities(xml_file.read_text(encoding="utf-8"))
        devices, handles, baselines, waited = claim_uuids([row["uuid"] for row in binding["devices"]],
            guard, out, deadline, Path(os.environ["PUBLIC"]), msvcrt,
            required_indices={row["uuid"]: row["index"] for row in binding["devices"]},
            min_free_mib=args.min_free_mib, reporter=reporter)
        receipt["lock_wait_seconds"] = waited
        receipt["admission"] = baselines
        rechecked = resolve_binding(guard, out, deadline, environment, args.physical_indices,
                                    args.cuda_ordinals, args.probe_visible)
        if rechecked["key"] != binding["key"]:
            raise RuntimeError("actual CUDA UUID binding changed before owned launch")
        # Legacy actors retain their original console/evidence streams; only the
        # fixed NVFP4 sealer opts into separate bounded collector-owned logs.
        reporter.emit("launch", {"lock_wait_seconds": round(waited, 1),
                                 "service_wait_seconds": round(receipt.get("service_transaction", {}).get("wait_seconds", 0), 1),
                                 "remaining_budget_seconds": round(max(0, deadline - time.monotonic()), 1),
                                 "devices": [{"index": row.get("index"), "uuid": row.get("uuid")} for row in devices],
                                 "admission": baselines})
        result = execute_owned(command, out, deadline, windows.WindowsJob, environment, capture=False)
        receipt["child"] = result
        code = complete_resource(receipt, result, devices, guard, out, baselines)
    except BaseException as error:
        cancel.disarm()
        receipt["error"] = f"{type(error).__name__}: {error}"
    finally:
        cancel.disarm()  # from here a cancel only prints; release, receipt and summary always finish
        try:
            # Global order: service first, then exact GPU set; release in reverse.
            release_transaction(handles, service_handle, msvcrt)
        except BaseException as error:
            receipt["resource_completed"] = False
            receipt["error"] = f"resource release failed: {type(error).__name__}: {error}"
            code = 1
        (out / "resource-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    # The job log must say WHY on its own: the receipt file sits outside any uploaded artifact.
    summary = {"resource_completed": receipt["resource_completed"], "model_inference_proof": False,
               "resource_receipt": str(out / "resource-receipt.json")}
    summary.update({key: receipt[key] for key in ("error", "child") if key in receipt})
    print(json.dumps(summary), flush=True)
    restore_signal_handlers(previous_handlers)
    return code if code != 0 or receipt["resource_completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
