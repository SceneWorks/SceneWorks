"""Fixed physical-GPU-1 adapter for the explicit NVFP4 proof, never ordinary CI.

Reuses inference@91ff's UUID byte-lock/idle protocol and actual WindowsJob API.
The Actions per-run key preserves requests; it is NOT a physical GPU mutex.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

PIN = "91ffb4a819aa2990c652db2d881ba0b9b10126ce"
GUARD_SHA = "301f0d0dd18b55aa9eaf5f3ffebac8122004f7bde59aaac4e7c5103e3451bf3f"
WINDOWS_SHA = "b50647bb1566df498f8cf4b57fa9288278e8ee1301de37afec3da478783123dd"
# Same ceiling as the preserved 120-minute workflow job, not a new resource cap.
JOB_SECONDS = 120 * 60


_shared_spec = importlib.util.spec_from_file_location("nvfp4_windows_execution", Path(__file__).with_name("windows_cuda_execution.py"))
shared = importlib.util.module_from_spec(_shared_spec)
_shared_spec.loader.exec_module(shared)
component, unlock, execute_owned = shared.component, shared.unlock, shared.execute_owned


def claim_fixed(guard, out, deadline, public, locker,
                clock=time.monotonic, sleep=time.sleep):
    if clock() >= deadline:
        raise TimeoutError("fixed GPU1 admission deadline exhausted")
    devices, _ = guard.inventory(out, "fixed-admission", deadline)
    selected = [device for device in devices if device["index"] == 1]
    if len(selected) != 1:
        raise RuntimeError("fixed physical GPU1 is unavailable; no device fallback")
    identifier = selected[0]["uuid"]
    checked, handles, baselines, waited = shared.claim_uuids(
        [identifier], guard, out, deadline, public, locker,
        required_indices={identifier: 1}, phase_names={identifier: "claimed-fixed-device"},
        clock=clock, sleep=sleep)
    return checked[0], handles[0], baselines[identifier], waited










def publish(output: Path, receipt: dict):
    """Add only controller-written resource metadata, not raw inventory/cache files."""
    artifact = output / "artifact"
    artifact.mkdir(parents=True, exist_ok=True)
    if artifact.is_symlink() or not artifact.is_dir():
        raise RuntimeError("unsafe child artifact directory")
    evidence_file = artifact / "evidence.json"
    if evidence_file.exists():
        if (evidence_file.is_symlink() or not evidence_file.is_file()
                or evidence_file.stat().st_nlink != 1
                or evidence_file.stat().st_size > 2 * 1024 * 1024):
            raise RuntimeError("unsafe child evidence manifest")
        evidence = json.loads(evidence_file.read_text(encoding="utf-8"))
    else:
        evidence = {"schema_version": 1, "accepted": False,
                    "source": receipt.get("app_source"),
                    "error": "physical admission or owned-child proof did not complete"}
    receipt["accepted"] = receipt.get("accepted") is True and evidence.get("body_proof_accepted") is True
    evidence["accepted"] = receipt["accepted"]
    evidence["resource_verification_pending"] = False
    evidence["physical_device"] = receipt
    body = (json.dumps(receipt, indent=2) + "\n").encode()
    physical = artifact / "physical-device.json"
    with physical.open("xb") as file:
        file.write(body)
    evidence["resource_evidence"] = {"relative_path": physical.name, "bytes": len(body),
                                     "sha256": hashlib.sha256(body).hexdigest()}
    temporary = artifact / "evidence.json.tmp"
    temporary.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    temporary.replace(evidence_file)
    # Keep the direct collector path consistent with the uploaded final verdict.
    root_evidence = output / "evidence.json"
    if root_evidence.exists() and (root_evidence.is_symlink() or not root_evidence.is_file()
                                   or root_evidence.stat().st_nlink != 1):
        raise RuntimeError("unsafe direct child evidence manifest")
    root_temporary = output / "evidence.json.guard-final.tmp"
    root_temporary.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    root_temporary.replace(root_evidence)
    return receipt["accepted"]


def main():
    if os.name != "nt" or len(sys.argv) != 1:
        raise RuntimeError("fixed NVFP4 physical guard requires Windows; no command/selector arguments")
    shared.require_outermost(os.environ)
    import msvcrt
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1" or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise RuntimeError("preserved physical-GPU1 environment is required")
    output = Path(os.environ["MODEL_PROOF_OUTPUT_DIR"])
    out = Path(os.environ["MODEL_PROOF_RESOURCE_DIR"])
    if not output.is_absolute() or output.exists() or not out.is_absolute():
        raise RuntimeError("proof paths must be fresh absolute owned directories")
    out.mkdir(parents=True, exist_ok=False)
    public = Path(os.environ["PUBLIC"])
    guard, windows = shared.prepared_components()
    receipt = {"schema_version": 1, "accepted": False,
               "guard_source_repository": "SceneWorks/inference", "guard_source_revision": PIN,
               "guard_sha256": GUARD_SHA, "windows_job_sha256": WINDOWS_SHA,
               "physical_index": 1, "lock_protocol": "Public/SceneWorks/gpu-locks/<UUID>.lock:byte0:msvcrt",
               "actions_group_is_physical_mutex": False}
    handle = None
    try:
        receipt["app_source"] = {
            "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "tree": subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], text=True).strip(),
        }
        deadline = time.monotonic() + JOB_SECONDS
        device, handle, baseline, waited = claim_fixed(guard, out, deadline, public, msvcrt)
        receipt.update(uuid=device["uuid"], lock_wait_seconds=waited,
                       reserved_baseline_mib=baseline["reserved_baseline_mib"])
        node = shutil.which("node")
        if not node:
            raise RuntimeError("prepared Node runtime is unavailable")
        child_env = os.environ.copy()
        # Logical device0 maps to the EXACT locked physical GPU1 UUID, even if
        # driver ordinals change. This is not selection of another idle device.
        child_env["CUDA_VISIBLE_DEVICES"] = device["uuid"]
        receipt["child_cuda_visible_devices"] = device["uuid"]
        result = execute_owned([node, str(root / "scripts/ci/imported-nvfp4-proof.mjs")],
                               out, deadline, windows.WindowsJob, child_env)
        receipt["child"] = result
        guard.verify_idle(device, out, "after-owned-proof", baseline)
        receipt["post_child_idle_verified"] = True
        receipt["accepted"] = result["returncode"] == 0 and not result["timeout"]
    except BaseException as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
    finally:
        # The owned Job has drained/reaped before returning; the physical lock
        # remains held through that drain and the post-child driver recheck.
        if handle is not None:
            unlock(handle, msvcrt)
    accepted = publish(output, receipt)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a", encoding="utf-8") as summary:
            summary.write(f"\n## Final physical CUDA proof verdict\n\nAccepted: **{str(accepted).lower()}**\n\n"
                          f"Physical GPU1 UUID: `{receipt.get('uuid', 'unverified')}`\n\n"
                          "This final verdict includes owned-child drain and the post-child idle recheck.\n")
    print(json.dumps({"accepted": accepted, "physical_uuid": receipt.get("uuid"),
                      "resource_evidence": str(output / "artifact/physical-device.json")}))
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
