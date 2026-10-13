"""CPU policy fixture: verbatim pure functions/constants from frozen inference veda_cuda_guard.py.
Production guardian imports actual SHA-bound inference module; this fixture never authorizes GPU.
Original file SHA256 301f0d0dd18b55aa9eaf5f3ffebac8122004f7bde59aaac4e7c5103e3451bf3f
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
