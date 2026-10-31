"""One dispatched, shared-device quant kernel check; no model/VRAM proof or lease."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import time

_spec = importlib.util.spec_from_file_location("cuda_kernel_owned", Path(__file__).with_name("windows_cuda_execution.py"))
owned = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(owned)
COMMAND = ("cargo", "test", "--locked", "-p", "sceneworks-worker", "--features", "backend-candle",
           "--test", "cuda_quant_smoke", "cuda_qmatmul_matches_cpu", "--", "--ignored", "--exact",
           "--nocapture", "--test-threads=1")


DeviceIdentity = owned.DeviceIdentity

def require_dispatch(environment):
    owned.require_outermost(environment)
    if (environment.get("GITHUB_EVENT_NAME") != "workflow_dispatch"
            or not environment.get("GITHUB_WORKFLOW_REF", "").startswith(
                "SceneWorks/SceneWorks/.github/workflows/windows-candle.yml@")):
        raise RuntimeError("fixed kernel validation requires its dedicated dispatch profile")
    inputs = json.loads(environment.get("KERNEL_DISPATCH_INPUTS", "{}"))
    if (not isinstance(inputs, dict) or inputs.get("quant_kernel_only") is not True
            or inputs.get("quant_kernel_physical_index") not in ("0", "1")
            or any(value is True for key, value in inputs.items() if key != "quant_kernel_only")):
        raise RuntimeError("kernel-only dispatch refuses missing, malformed or mixed model profiles")
    return inputs


def validate(physical_index, environment, out, deadline, identity, job_type,
             resolve=owned.resolve_binding, execute=owned.execute_owned):
    if physical_index not in ("0", "1"):
        raise RuntimeError("only an explicit physical index 0 or 1 is supported")
    physical = resolve(identity, out, deadline, environment, physical_indices=physical_index)
    selected = physical["devices"]
    child_env = dict(environment, CUDA_VISIBLE_DEVICES=selected[0]["uuid"], CUDA_DEVICE_ORDER="PCI_BUS_ID",
                     SCENEWORKS_CUDA_KERNEL_REQUIRED="1")
    def binding():
        current = resolve(identity, out, deadline, child_env, cuda_ordinals="0")
        if current["devices"] != selected:
            raise RuntimeError("selected physical index/UUID differs from actual CUDA device 0")
        return current
    before = binding()
    child = execute(list(COMMAND), out, deadline, job_type, child_env)
    after = binding()
    if before["key"] != after["key"]:
        raise RuntimeError("CUDA mapping changed during the kernel check")
    if child.get("returncode") != 0 or child.get("timeout") is not False or child.get("owned_tree_empty") is not True:
        raise RuntimeError("owned kernel child failed, timed out or retained descendants")
    stdout = (out / "proof.stdout.log").read_text(encoding="utf-8", errors="replace")
    if (not re.search(r"^test cuda_qmatmul_matches_cpu \.\.\. ok$", stdout, re.M)
            or not re.search(r"^test result: ok\. 1 passed; 0 failed; 0 ignored;", stdout, re.M)):
        raise RuntimeError("exactly one required kernel test must pass; zero matches never pass")
    return {"physical_index": int(physical_index), "uuid": selected[0]["uuid"], "binding": after, "child": child}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-index", choices=("0", "1"))
    parser.add_argument("--check-dispatch", action="store_true")
    args = parser.parse_args()
    inputs = require_dispatch(os.environ)
    if args.check_dispatch:
        if args.physical_index is not None:
            parser.error("dispatch-only check cannot execute a device")
        return
    if args.physical_index != inputs["quant_kernel_physical_index"]:
        parser.error("physical index must match the exact dispatch selection")
    if os.name != "nt":
        raise RuntimeError("native Windows execution required")
    run_id, attempt = os.environ["GITHUB_RUN_ID"], os.environ["GITHUB_RUN_ATTEMPT"]
    if not run_id.isdecimal() or not attempt.isdecimal():
        raise RuntimeError("invalid run identity")
    out = Path(os.environ["RUNNER_TEMP"]) / f"cuda-kernel-{run_id}-{attempt}"
    if not out.is_absolute():
        raise RuntimeError("absolute runner temp required")
    out.mkdir(exist_ok=False)
    receipt = {"accepted": False, "shared_device": True, "isolated_device": False,
               "model_inference_proof": False, "vram_measurement": False, "command": list(COMMAND)}
    try:
        windows = owned.component("media_campaign_windows.py", owned.WINDOWS_SHA)
        receipt.update(validate(args.physical_index, os.environ, out, time.monotonic() + 60 * 60,
                                DeviceIdentity(windows.trusted_nvidia_smi()), windows.WindowsJob))
        receipt["accepted"] = True
    except BaseException as error:
        receipt["error"] = type(error).__name__
        raise
    finally:
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
