"""Physical admission and owned-child policy preserved from CUDA S0 10c09c5e.

No guard widening: exact driver types, caps, timed samples, reserved/nonrising
baseline and host-global UUID lock. Coordinator separately owns the shared host.
"""
from __future__ import annotations
import csv
import json
import os
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET

MAX_SECONDS = 3600
ALLOWED_RUNNERS = {"cuda-windows", "cuda-windows-2"}
# Reserved display/WDDM baseline, not CUDA work: cap it at 2 GiB on this 96-GiB pool.
MAX_GRAPHICS_BASELINE_MIB = 2048
MAX_BASELINE_UTILIZATION_PERCENT = 5
BASELINE_SAMPLES = 5
BASELINE_INTERVAL_SECONDS = 0.5
MAX_SAMPLING_SECONDS = 5


def command(args: list[str], deadline: float | None = None) -> str:
    timeout = 30 if deadline is None else min(30, deadline - time.monotonic())
    if timeout <= 0:
        raise TimeoutError("GPU inventory sampling deadline exhausted")
    return subprocess.check_output(args, text=True, timeout=timeout).strip()


def inventory(out: Path | None = None, label: str = "snapshot",
              deadline: float | None = None) -> tuple[list[dict], list[list[str]]]:
    rows = command(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], deadline)
    devices = []
    for row in csv.reader(rows.splitlines()):
        i, uuid, name, driver, total, used, utilization = [v.strip() for v in row]
        devices.append(dict(index=int(i), uuid=uuid, name=name, driver=driver, total_mib=int(total), used_mib=int(used), utilization_percent=int(utilization)))
    xml = command(["nvidia-smi", "-q", "-x"], deadline)
    if out is not None:
        (out / f"{label}-nvidia-smi.xml").write_text(xml, encoding="utf-8", newline="\n")
    xml_uuids = {(gpu.findtext("uuid") or "").strip() for gpu in ET.fromstring(xml).findall("gpu")}
    if xml_uuids != {device["uuid"] for device in devices}:
        raise RuntimeError("driver XML and CSV GPU identities differ")
    processes = typed_processes(xml)
    # Cover permission-opaque processes using driver-supplied type, never executable names.
    # A PID present in the CSV but absent from the typed snapshot remains unknown/refused.
    rows = command(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader,nounits"], deadline)
    known = {(p[0], p[1]) for p in processes}
    typed_by_pid = {}
    for uuid, pid, _, _ in processes:
        typed_by_pid.setdefault(pid, set()).add(uuid)
    for row in csv.reader(rows.splitlines()):
        if not row:
            continue
        row = [v.strip() for v in row]
        if len(row) != 3:
            raise RuntimeError("unrecognized GPU process inventory")
        if row[1] in typed_by_pid and row[0] not in typed_by_pid[row[1]]:
            raise RuntimeError("driver XML and compute-app GPU UUID/PID differ")
        if (row[0], row[1]) not in known:
            processes.append(row + ["unknown"])
    return devices, processes


def typed_processes(xml: str) -> list[list[str]]:
    result = []
    for gpu in ET.fromstring(xml).findall("gpu"):
        uuid = (gpu.findtext("uuid") or "").strip()
        if not uuid:
            raise RuntimeError("driver XML GPU identity missing")
        for process in gpu.findall("processes/process_info"):
            pid = (process.findtext("pid") or "").strip()
            if not pid:
                raise RuntimeError("driver XML process PID missing")
            result.append([uuid, pid, process.findtext("process_name") or "unknown", (process.findtext("type") or "unknown").strip()])
    return result


def idle(device: dict, processes: list[list[str]]) -> bool:
    return (device["used_mib"] < MAX_GRAPHICS_BASELINE_MIB
            and device["utilization_percent"] < MAX_BASELINE_UTILIZATION_PERCENT
            and all(len(row) == 4 and row[3] in {"G", "C+G"} and row[1]
                    for row in processes if row[0] == device["uuid"]))


def process_identities(uuid: str, processes: list[list[str]]) -> list[tuple[str, str, str]]:
    return sorted((row[0], row[1], row[3]) for row in processes if row[0] == uuid)


def sample_baseline(device: dict, out: Path, phase: str,
                    baseline: dict | None = None) -> tuple[dict, dict]:
    """Admit one physical UUID only after bounded, stable driver observations."""
    samples = []
    started = time.monotonic()
    try:
        for index in range(BASELINE_SAMPLES):
            if index:
                time.sleep(BASELINE_INTERVAL_SECONDS)
            devices, processes = inventory(out, f"{phase}-sample-{index + 1}", started + MAX_SAMPLING_SECONDS)
            matches = [row for row in devices if row["uuid"] == device["uuid"]]
            sample = dict(sample_number=index + 1, elapsed_seconds=time.monotonic() - started,
                          devices=devices, processes=processes)
            samples.append(sample)
            if len(matches) != 1 or time.monotonic() - started > MAX_SAMPLING_SECONDS:
                raise RuntimeError(f"physical GPU identity or sampling deadline changed during {phase}")
            current = matches[0]
            identities = process_identities(device["uuid"], processes)
            if not idle(current, processes):
                raise RuntimeError(f"physical GPU occupied during {phase}")
            if index == 0 and baseline is None:
                baseline = dict(uuid=device["uuid"], reserved_baseline_mib=current["used_mib"],
                                process_identities=identities)
            if (identities != baseline["process_identities"]
                    or current["used_mib"] > baseline["reserved_baseline_mib"]
                    or (index and current["used_mib"] > samples[index - 1]["selected_used_mib"])):
                raise RuntimeError(f"physical GPU baseline changed during {phase}")
            sample["selected_used_mib"] = current["used_mib"]
        return current, baseline
    finally:
        (out / f"{phase}-samples.json").write_text(json.dumps(samples, indent=2), encoding="utf-8", newline="\n")


def claim_device(out: Path):
    import msvcrt
    devices, processes = inventory(out, "admission")
    (out / "admission-inventory.json").write_text(json.dumps(dict(devices=devices, processes=processes), indent=2), encoding="utf-8", newline="\n")
    # Public is host-global for both listeners, unlike runner.temp/workspaces.
    lock_dir = Path(os.environ["PUBLIC"]) / "SceneWorks" / "gpu-locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    for device in sorted(devices, key=lambda d: d["index"], reverse=True):
        if not idle(device, processes):
            continue
        handle = (lock_dir / f'{device["uuid"]}.lock').open("a+b")
        if handle.seek(0, 2) == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            handle.close()
            continue
        try:
            checked, baseline = sample_baseline(device, out, "claimed-device")
            return checked, handle, baseline
        except RuntimeError:
            handle.close()
        except Exception:
            handle.close()
            raise
    raise RuntimeError("no unclaimed idle physical CUDA device; refusing benchmark")


def verify_idle(device: dict, out: Path, phase: str, baseline: dict):
    sample_baseline(device, out, phase, baseline)


def persist_receipt(out: Path, receipt: dict):
    temporary = out / "receipt.json.tmp"
    temporary.write_text(json.dumps(receipt, indent=2), encoding="utf-8", newline="\n")
    temporary.replace(out / "receipt.json")


def execute_owned(name, args, out, deadline, job_type):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("CUDA validation one-hour session budget exhausted")
    started = time.monotonic()
    with (out / f"{name}.stdout.log").open("wb") as stdout, (out / f"{name}.stderr.log").open("wb") as stderr:
        job = job_type()
        child = None
        try:
            child = subprocess.Popen(args, stdout=stdout, stderr=stderr,
                                     env=os.environ.copy(), creationflags=0x00000004)  # CREATE_SUSPENDED
            try:
                job.assign_and_resume(child)
            except Exception:
                child.kill()
                child.wait(timeout=10)
                raise
            try:
                code = child.wait(timeout=remaining)
                return dict(name=name, argv=args, returncode=code, timeout=False, seconds=time.monotonic()-started)
            except subprocess.TimeoutExpired:
                job.terminate_and_reap(child, 10)
                return dict(name=name, argv=args, returncode=None, termination_returncode=child.returncode, timeout=True, seconds=time.monotonic()-started)
        finally:
            job.close()

