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
import subprocess
import sys
import time
from urllib.parse import urlsplit

ACTIVE_KEY = "SW_CI_CUDA_EXECUTION_ACTIVE"

PIN = "91ffb4a819aa2990c652db2d881ba0b9b10126ce"
GUARD_SHA = "301f0d0dd18b55aa9eaf5f3ffebac8122004f7bde59aaac4e7c5103e3451bf3f"
WINDOWS_SHA = "b50647bb1566df498f8cf4b57fa9288278e8ee1301de37afec3da478783123dd"


STRICT_POLICY = "strict-idle"
FUNCTIONAL_POLICY = "functional-coexistence"
# SW-7033 Tier 2 A/B only: physical GPU1 is the host's display GPU, so it is never strictly idle
# while the desktop session runs. Admit it when it has room for the largest route and is quiet.
SHARED_DISPLAY_POLICY = "shared-display"
SHARED_DISPLAY_ACTORS = ("tier2-ab-cuda",)
SHARED_WINDOW_SAMPLES = 10            # one inventory per second after the UUID lock is held
SHARED_WINDOW_INTERVAL_SECONDS = 1.0
SHARED_MAX_SPREAD_MIB = 256           # max - min device memory.used across the window
SHARED_MAX_GROWTH_MIB = 128           # last - first device memory.used across the window
SHARED_MAX_UTILIZATION_PERCENT = 30   # display composition stays far below this
SHARED_MAX_UTILIZATION_SPREAD = 15
# WDDM reports per-process used_memory as N/A, so a foreign compute ("C"/unknown) process is
# bounded by the DEVICE memory.used instead: a context holder fits under this, a real job does not.
SHARED_FOREIGN_COMPUTE_DEVICE_MAX_MIB = 8192
# Where the driver does report per-process memory, any foreign compute process above this refuses.
SHARED_FOREIGN_COMPUTE_PROCESS_MAX_MIB = 1024


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


def require_admission_policy(policy, actor, environment, physical_indices, cuda_ordinals, probe_visible):
    if policy == STRICT_POLICY:
        return
    if policy == SHARED_DISPLAY_POLICY:
        if actor not in SHARED_DISPLAY_ACTORS:
            raise RuntimeError("shared-display admission is restricted to the Tier 2 A/B actor")
        if (environment.get("GITHUB_EVENT_NAME") != "workflow_dispatch"
                or not environment.get("GITHUB_WORKFLOW_REF", "").startswith(
                    "SceneWorks/SceneWorks/.github/workflows/memory-catalog-campaign.yml@")):
            raise RuntimeError("shared-display admission requires the dispatched campaign workflow")
        if (physical_indices is not None or cuda_ordinals != "0" or probe_visible not in (None, "1")
                or environment.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"
                or any(environment.get(key) != "1" for key in
                       ("SCENEWORKS_GPU_ID", "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"))):
            raise RuntimeError("shared-display admission requires logical CUDA 0 masked to physical GPU1")
        return
    if policy != FUNCTIONAL_POLICY or actor != "load-progress-acceptance":
        raise RuntimeError("functional coexistence is restricted to load-progress acceptance")
    workflows = ("memory-catalog-campaign.yml", "load-progress-acceptance.yml")
    if (environment.get("GITHUB_EVENT_NAME") != "workflow_dispatch"
            or not any(environment.get("GITHUB_WORKFLOW_REF", "").startswith(
                f"SceneWorks/SceneWorks/.github/workflows/{name}@") for name in workflows)):
        raise RuntimeError("functional coexistence requires the dedicated manual acceptance workflow")
    if (physical_indices is not None or cuda_ordinals != "0" or probe_visible != "1"
            or environment.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"
            or any(environment.get(key) != "1" for key in
                   ("SCENEWORKS_GPU_ID", "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"))):
        raise RuntimeError("functional acceptance requires logical CUDA 0 masked to physical GPU1")


def require_functional_binding(binding):
    devices = binding.get("devices", [])
    if (binding.get("kind") != "context-free-cuda-ordinal-uuid" or len(devices) != 1
            or devices[0].get("index") != 1
            or not isinstance(devices[0].get("uuid"), str)
            or not re.fullmatch(r"GPU-[A-Za-z0-9-]+", devices[0].get("uuid", ""))
            or binding.get("mapping") != [{"cuda_ordinal": 0, "uuid": devices[0]["uuid"], "nvml_index": 1}]):
        raise RuntimeError("functional acceptance physical GPU1/UUID mapping is unavailable")


def process_memory_mib(xml_text):
    """Per-(gpu uuid, pid) used_memory in MiB from the driver XML; None where WDDM says N/A."""
    import xml.etree.ElementTree as ET
    result = {}
    for gpu in ET.fromstring(xml_text).findall("gpu"):
        uuid = (gpu.findtext("uuid") or "").strip()
        for process in gpu.findall("processes/process_info"):
            text = (process.findtext("used_memory") or "").strip()
            match = re.fullmatch(r"([0-9]+) MiB", text)
            result[(uuid, (process.findtext("pid") or "").strip())] = int(match.group(1)) if match else None
    return result


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


def shared_display_decision(samples, uuid, min_free_mib):
    """Pure admission rule over a window of (devices, processes, per-process MiB) samples.

    Tolerates display/graphics processes (G, C+G) and a foreign compute process that only holds a
    context; refuses a large or growing foreign compute footprint, an unstable device, or too
    little free memory for the largest route plus its margin.
    """
    reasons = []
    rows = []
    for devices, processes, memory in samples:
        matches = [row for row in devices if row["uuid"].casefold() == uuid.casefold()]
        if len(matches) != 1:
            return False, ["leased GPU UUID missing from a sample"], {}
        device = matches[0]
        foreign = [row for row in processes if row[0].casefold() == uuid.casefold()
                   and (len(row) != 4 or row[3] not in ("G", "C+G"))]
        rows.append((device, foreign, memory or {}))
    if len(rows) < 2:
        return False, ["sampling window too short"], {}
    used = [device["used_mib"] for device, _, _ in rows]
    utilization = [device["utilization_percent"] for device, _, _ in rows]
    last, foreign_last, memory_last = rows[-1]
    free = last["total_mib"] - last["used_mib"]
    summary = {"samples": len(rows), "baseline_used_mib": max(used), "min_used_mib": min(used),
               "free_mib": free, "min_free_mib": min_free_mib, "max_utilization_percent": max(utilization),
               "foreign_compute": sorted({(row[1], row[2], row[3] if len(row) == 4 else "unknown")
                                          for _, foreign, _ in rows for row in foreign}),
               "per_process_memory_reported": any(value is not None for _, _, memory in rows for value in memory.values())}
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
    if any(foreign for _, foreign, _ in rows):
        if max(used) > SHARED_FOREIGN_COMPUTE_DEVICE_MAX_MIB:
            reasons.append(f"foreign compute process present with device memory.used {max(used)} MiB > "
                           f"{SHARED_FOREIGN_COMPUTE_DEVICE_MAX_MIB} MiB (per-process memory not attributable)")
        for _, foreign, memory in rows:
            for row in foreign:
                value = memory.get((row[0], row[1]))
                if value is not None and value > SHARED_FOREIGN_COMPUTE_PROCESS_MAX_MIB:
                    reasons.append(f"foreign compute pid {row[1]} uses {value} MiB")
    return not reasons, sorted(set(reasons)), summary


def sample_shared_window(guard, out, deadline, uuid, min_free_mib, phase, clock=time.monotonic, sleep=time.sleep):
    samples = []
    for index in range(SHARED_WINDOW_SAMPLES):
        if index:
            sleep(SHARED_WINDOW_INTERVAL_SECONDS)
        label = f"{phase}-shared-{index + 1}"
        devices, processes = guard.inventory(out, label, deadline)
        xml_file = out / f"{label}-nvidia-smi.xml"
        memory = process_memory_mib(xml_file.read_text(encoding="utf-8")) if xml_file.exists() else {}
        samples.append((devices, processes, memory))
    admitted, reasons, summary = shared_display_decision(samples, uuid, min_free_mib)
    (out / f"{phase}-shared-window.json").write_text(json.dumps(
        {"admitted": admitted, "reasons": reasons, "summary": summary}, indent=2, default=list),
        encoding="utf-8", newline="\n")
    return admitted, reasons, summary


def resource_receipt(actor, command, policy):
    receipt = {"schema_version": 1, "kind": "cuda-execution-resource-lease-v1", "model_inference_proof": False,
               "resource_completed": False, "actor": actor, "argv": command,
               "admission_policy": policy,
               "guard_source_revision": PIN, "guard_sha256": GUARD_SHA, "windows_job_sha256": WINDOWS_SHA}
    if policy == FUNCTIONAL_POLICY:
        receipt.update(shared_device=True, isolated_device=False, vram_measurement=False,
                       performance_measurement=False, post_child_idle_verified=False)
    if policy == SHARED_DISPLAY_POLICY:
        receipt.update(shared_device=True, isolated_device=False, post_child_idle_verified=False,
                       observation_note="shared display GPU: memory peaks are attributable only above the recorded baseline; times were taken alongside display work")
    return receipt


def complete_resource(receipt, result, devices, guard, out, baselines, policy):
    if policy == SHARED_DISPLAY_POLICY:
        receipt["post_child_idle_verified"] = False
        receipt["resource_completed"] = (result["returncode"] == 0 and result["timeout"] is False
                                         and result.get("owned_tree_empty") is True)
        return result["returncode"] if result["returncode"] is not None else 124
    if policy == STRICT_POLICY:
        for device in devices:
            guard.verify_idle(device, out, f'after-child-{device["index"]}', baselines[device["uuid"]])
        receipt["post_child_idle_verified"] = True
    else:
        receipt["post_child_idle_verified"] = False
    receipt["resource_completed"] = (result["returncode"] == 0 and not result["timeout"]
        and (policy == STRICT_POLICY or (result["timeout"] is False and result.get("owned_tree_empty") is True)))
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


def claim_uuids(uuids, guard, out, deadline, public, locker,
                required_indices=None, phase_names=None, clock=time.monotonic, sleep=time.sleep,
                admission_policy=STRICT_POLICY, min_free_mib=None):
    expected = sorted(uuids, key=str.casefold)
    if admission_policy == SHARED_DISPLAY_POLICY and min_free_mib is None:
        raise RuntimeError("shared-display admission needs --min-free-mib")
    if not expected or len({value.casefold() for value in expected}) != len(expected):
        raise RuntimeError("explicit unique CUDA UUID set is required")
    if not public.is_absolute():
        raise RuntimeError("PUBLIC must name the host-global absolute lock namespace")
    started = clock()
    while clock() < deadline:
        devices, processes = guard.inventory(out, "uuid-admission", deadline)
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
        if admission_policy == STRICT_POLICY and not all(guard.idle(device, processes) for device in selected):
            sleep(min(1, max(0, deadline - clock())))
            continue
        handles = []
        busy = False
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
                    busy = True
                    break
                handles.append(handle)
        except BaseException:
            if handle is not None and handle not in handles:
                handle.close()
            release_all(handles, locker)
            raise
        if busy:
            release_all(handles, locker)
            sleep(min(1, max(0, deadline - clock())))
            continue
        try:
            if admission_policy == FUNCTIONAL_POLICY:
                return selected, handles, {}, clock() - started
            if admission_policy == SHARED_DISPLAY_POLICY:
                baselines = {}
                for device in selected:
                    phase = (phase_names or {}).get(device["uuid"], f'claimed-device-{device["index"]}')
                    admitted, reasons, summary = sample_shared_window(guard, out, deadline, device["uuid"],
                                                                      min_free_mib, phase, clock, sleep)
                    if not admitted:
                        raise RuntimeError("shared display GPU not admitted: " + "; ".join(reasons))
                    baselines[device["uuid"]] = summary
                return selected, handles, baselines, clock() - started
            baselines = {}
            checked = []
            for device in selected:
                phase = (phase_names or {}).get(device["uuid"], f'claimed-device-{device["index"]}')
                current, baseline = guard.sample_baseline(device, out, phase)
                checked.append(current)
                baselines[device["uuid"]] = baseline
            return checked, handles, baselines, clock() - started
        except RuntimeError:
            release_all(handles, locker)
            sleep(min(1, max(0, deadline - clock())))
        except BaseException:
            release_all(handles, locker)
            raise
    if admission_policy == FUNCTIONAL_POLICY:
        raise TimeoutError("exact GPU UUID lock wait exhausted the existing execution budget")
    if admission_policy == SHARED_DISPLAY_POLICY:
        raise TimeoutError("shared display GPU never became quiet with enough free memory within the budget")
    raise TimeoutError("exact GPU UUID lock/idle wait exhausted the existing execution budget")


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


def prepared_components(policy=STRICT_POLICY):
    guard = component("veda_cuda_guard.py", GUARD_SHA)
    windows = component("media_campaign_windows.py", WINDOWS_SHA)
    trusted = windows.trusted_nvidia_smi()
    original = guard.command
    def command(args, deadline=None):
        if args[0] != "nvidia-smi":
            raise RuntimeError("unexpected physical inventory command")
        return original([trusted, *args[1:]], deadline)
    guard.command = command
    return (DeviceIdentity(trusted) if policy == FUNCTIONAL_POLICY else guard), windows


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


def claim_service(url, public, deadline, locker, clock=time.monotonic, sleep=time.sleep):
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
            sleep(min(1, max(0, deadline - clock())))
            continue
        except BaseException:
            handle.close()
            raise
        return handle, {**identity, "lock_path": str(file), "wait_seconds": clock() - started}
    raise TimeoutError("host service port transaction wait exhausted the existing execution budget")


def release_transaction(gpu_handles, service_handle, locker):
    try:
        release_all(gpu_handles, locker)
    finally:
        if service_handle is not None:
            unlock(service_handle, locker)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--budget-minutes", type=float, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--physical-indices")
    modes.add_argument("--cuda-ordinals", default="0")
    parser.add_argument("--probe-visible")
    parser.add_argument("--admission-policy", choices=(STRICT_POLICY, FUNCTIONAL_POLICY, SHARED_DISPLAY_POLICY), default=STRICT_POLICY)
    parser.add_argument("--min-free-mib", type=int, help="shared-display only: free memory the largest route needs plus its margin")
    parser.add_argument("--service-url", help="actual loopback service URL; optional host transaction lock")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if (os.name != "nt" or not re.fullmatch(r"[A-Za-z0-9_.-]+", args.actor)
            or not math.isfinite(args.budget_minutes) or args.budget_minutes <= 0):
        raise RuntimeError("explicit Windows actor/budget is required")
    environment = os.environ.copy()
    require_outermost(environment)
    require_admission_policy(args.admission_policy, args.actor, environment,
                             args.physical_indices, args.cuda_ordinals, args.probe_visible)
    if (args.admission_policy == SHARED_DISPLAY_POLICY) != (args.min_free_mib is not None):
        raise RuntimeError("--min-free-mib is required with, and only with, shared-display admission")
    if args.min_free_mib is not None and not 0 < args.min_free_mib < 1 << 20:
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
    receipt = resource_receipt(args.actor, command, args.admission_policy)
    handles = []
    service_handle = None
    code = 1
    try:
        if args.service_url is not None:
            service_handle, receipt["service_transaction"] = claim_service(
                args.service_url, Path(os.environ["PUBLIC"]), deadline, msvcrt)
        guard, windows = prepared_components(args.admission_policy)
        binding = resolve_binding(guard, out, deadline, environment, args.physical_indices,
                                  args.cuda_ordinals, args.probe_visible)
        receipt["binding"] = binding
        if args.admission_policy == SHARED_DISPLAY_POLICY:
            xml_file = out / "binding-nvidia-smi.xml"
            if xml_file.exists():
                receipt["gpu_identities"] = gpu_identities(xml_file.read_text(encoding="utf-8"))
        if args.admission_policy == FUNCTIONAL_POLICY:
            require_functional_binding(binding)
        devices, handles, baselines, waited = claim_uuids([row["uuid"] for row in binding["devices"]],
            guard, out, deadline, Path(os.environ["PUBLIC"]), msvcrt,
            required_indices={row["uuid"]: row["index"] for row in binding["devices"]}
                if args.admission_policy == FUNCTIONAL_POLICY else None,
            admission_policy=args.admission_policy, min_free_mib=args.min_free_mib)
        receipt["lock_wait_seconds"] = waited
        if args.admission_policy == SHARED_DISPLAY_POLICY:
            receipt["shared_admission"] = baselines
        rechecked = resolve_binding(guard, out, deadline, environment, args.physical_indices,
                                    args.cuda_ordinals, args.probe_visible)
        if args.admission_policy == FUNCTIONAL_POLICY:
            require_functional_binding(rechecked)
            receipt["binding_rechecked"] = rechecked
        if rechecked["key"] != binding["key"]:
            raise RuntimeError("actual CUDA UUID binding changed before owned launch")
        # Legacy actors retain their original console/evidence streams; only the
        # fixed NVFP4 sealer opts into separate bounded collector-owned logs.
        result = execute_owned(command, out, deadline, windows.WindowsJob, environment, capture=False)
        receipt["child"] = result
        code = complete_resource(receipt, result, devices, guard, out, baselines, args.admission_policy)
    except BaseException as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
    finally:
        try:
            # Global order: service first, then exact GPU set; release in reverse.
            release_transaction(handles, service_handle, msvcrt)
        except BaseException as error:
            receipt["resource_completed"] = False
            receipt["error"] = f"resource release failed: {type(error).__name__}: {error}"
            code = 1
        (out / "resource-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"resource_completed": receipt["resource_completed"], "model_inference_proof": False,
                      "resource_receipt": str(out / "resource-receipt.json")}))
    return code if code != 0 or receipt["resource_completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
