"""Own the complete manual terminal campaign under the engine's physical GPU lease.

No lock implementation lives here: the accepted inference checkout supplies both
the existing PUBLIC UUID lock and suspended-child Windows Job primitives.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace

CAMPAIGN_SECONDS = 590 * 60  # Leave ten minutes of the workflow for final upload.
CLEANUP_SECONDS = 60
GPU_UUIDS = ("GPU-b1a31911-c7b4-2901-3d8b-9a62e228bfc0", "GPU-e4b79931-7be6-f216-460a-f5405cfafffe")
ORIGINAL_RUNTIME = {"sceneworks": "c8815c7cd46feba2f82562a315d1a44a6c88a9e8",
                    "inference": "2ff0353820e370d004305ae99a035ab38b93de30"}


def validated_plan(env):
    plan = json.loads(env["SW_F329_PLAN"])
    mode = plan.get("mode")
    if mode not in (None, "matched-resident-control"):
        raise ValueError("unknown terminal mode")
    matched = mode == "matched-resident-control"
    index = plan.get("physicalGpuIndex", 1)
    if type(index) is not int or index not in (0, 1):
        raise ValueError("physicalGpuIndex must be the exact integer 0 or 1")
    if index == 0 or "physicalGpuUuid" in plan:
        if plan.get("physicalGpuUuid") != GPU_UUIDS[index]:
            raise ValueError("canonical selected physical GPU UUID required")
    if index == 0 and "retainedWorkstation" in plan:
        raise ValueError("GPU0 requires all three anchors fresh; retained GPU1 control refused")
    if (plan.get("stationary_pair_authorized") is not True
            or (not matched and plan.get("additionalSparse24") is not True)
            or env.get("GITHUB_ACTIONS") != "true"
            or env.get("GITHUB_EVENT_NAME") != "workflow_dispatch"
            or env.get("RUNNER_NAME") != plan.get("expectedRunner")
            or env.get("RUNNER_NAME") not in {"cuda-windows", "cuda-windows-2"}
            or env.get("GITHUB_SHA") != plan.get("sceneworks")
            or env.get("SCENEWORKS_GPU_ID") != str(index)
            or env.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"):
        raise ValueError("explicit stationary manual source pair and selected physical GPU required")
    uuid = env.get("CUDA_VISIBLE_DEVICES", "")
    if not re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", uuid):
        raise ValueError("exact physical GPU UUID required; no ordinal fallback")
    if "physicalGpuUuid" in plan and uuid != plan["physicalGpuUuid"]:
        raise ValueError("visibility differs from selected physical UUID")
    for key in ["poolBytes", "externalReserveBytes"]:
        if type(plan.get(key)) is not int or plan[key] <= 0:
            raise ValueError("positive integer pool and reserve required")
    if matched:
        if (index != 0 or plan.get("matchedResidentControlAuthorized") is not True
                or not re.fullmatch(r"[a-f0-9]{40}", str(plan.get("inference", "")))
                or env.get("H3_CONTROLLER_AUTHORIZED_PAIR") != f"{plan['sceneworks']}:{plan['inference']}"
                or env.get("H3_TERMINAL_AUTHORIZED_PAIR") != ":".join(ORIGINAL_RUNTIME[key] for key in ["sceneworks", "inference"])
                or type(plan.get("timeoutSeconds")) is not int
                or not 60 <= plan["timeoutSeconds"] <= 7200
                or plan["poolBytes"] != 92 * 1024**3
                or plan["externalReserveBytes"] != 3 * 1024**3
                or "additionalSparse24" in plan):
            raise ValueError("single matched Resident control requires current controller, immutable CGPU/M4 runtime, exact GPU0/92+3 policy")
    elif plan["poolBytes"] + plan["externalReserveBytes"] != 24 * 1024**3 or plan["externalReserveBytes"] < 2 * 1024**3:
        raise ValueError("plan pool and external reserve must preserve the 24GiB envelope")
    return plan


class ReapedJob:
    """Add verified descendant reaping before execute_owned closes its Job handle."""
    def __init__(self, job_type):
        self.job = job_type()
        self.child = None
        self.reaped = False

    def assign_and_resume(self, child):
        self.child = child
        self.job.assign_and_resume(child)

    def terminate_and_reap(self, child, grace):
        self.job.terminate_and_reap(child, grace)
        self.reaped = True

    def close(self):
        try:
            if self.child is not None and not self.reaped:
                self.terminate_and_reap(self.child, 10)
        finally:
            self.job.close()


def supervise(sw_root, inference_root, output, env, *, claim, execute, job_type, clock=time.monotonic):
    plan = validated_plan(env)
    index = plan.get("physicalGpuIndex", 1)
    seconds = plan['timeoutSeconds'] + 1800 if plan.get('mode') else CAMPAIGN_SECONDS
    deadline = clock() + seconds
    lease_output = Path(str(output) + "-lease")
    lease_output.mkdir()  # Never overwrite a prior run's evidence.
    policy = SimpleNamespace(cuda_device_uuid=env["CUDA_VISIBLE_DEVICES"],
                             deadline_seconds=seconds, poll_millis=500,
                             child_gpu_cap_bytes=plan["poolBytes"],
                             gpu_free_reserve_bytes=plan["externalReserveBytes"])
    receipt = {"passed": False, "physicalGpuIndex": index, "physicalGpuUuid": policy.cuda_device_uuid,
               "scope": plan.get("mode", "three-anchor campaign"),
               "sourcePair": {key: plan[key] for key in ["sceneworks", "inference"]},
               "deadlineSeconds": seconds, "cleanupReserveSeconds": CLEANUP_SECONDS,
               "minimumClaimFreeBytes": policy.child_gpu_cap_bytes + policy.gpu_free_reserve_bytes,
               "minimumClaimScope": "matched Resident95 control" if plan.get("mode") else "capped pair minimum; workstation provider admission remains authoritative"}
    if plan.get("mode"):
        receipt["controllerSourcePair"] = receipt.pop("sourcePair")
        receipt["runtimeSourcePair"] = dict(ORIGINAL_RUNTIME)
        receipt["leaseImplementationSource"] = {"repository": "SceneWorks/inference", "revision": plan["inference"],
                                                 "meaning": "current controller lease/owned cleanup, not the compiled model runtime"}
    try:
        print(f"Waiting for existing physical GPU{index} UUID lease; original deadline {seconds}s", flush=True)
        with claim("candle-cuda", policy, lease_output, env,
                   constraints={"required_cuda_physical_index": index,
                                "required_cuda_device_uuid": policy.cuda_device_uuid},
                   original_deadline_monotonic=deadline) as record:
            receipt["claim"] = record
            print(f"Claimed physical GPU{index}; starting owned {plan.get('mode', 'three-anchor campaign')}", flush=True)
            result = execute("campaign", ["node", str(sw_root / "scripts" / "h3-cuda-memory-terminal.mjs"),
                                         "run", str(sw_root), str(inference_root), str(output)],
                             lease_output, deadline - CLEANUP_SECONDS, lambda: ReapedJob(job_type))
            receipt["child"] = result
            if result["timeout"] or result["returncode"] != 0:
                raise RuntimeError("owned terminal campaign failed or exhausted its original deadline")
        if not record.get("post_execution_verified"):
            raise RuntimeError("physical GPU post-execution verification missing")
        receipt["passed"] = True
    except BaseException as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (lease_output / "campaign-lease.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main():
    if os.name != "nt" or len(sys.argv) != 4:
        raise ValueError("expected Windows manual launcher: sceneworks inference D-output")
    sw_root, inference_root, output = (Path(value).resolve() for value in sys.argv[1:])
    if not str(output).lower().startswith("d:\\codex\\work\\"):
        raise ValueError("generated output must belong on D:")
    validated_plan(os.environ)
    sys.path.insert(0, str(inference_root))
    from scripts.ci.model_proof_resources import execution_claim
    from scripts.release.veda_cuda_guard import execute_owned
    from scripts.media_campaign_windows import WindowsJob
    supervise(sw_root, inference_root, output, os.environ, claim=execution_claim,
              execute=execute_owned, job_type=WindowsJob)


if __name__ == "__main__":
    main()
