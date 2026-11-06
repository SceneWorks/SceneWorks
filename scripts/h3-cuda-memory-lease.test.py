"""Pure launch/ownership tests: no driver, device, subprocess or model execution."""
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("h3_memory_lease", Path(__file__).with_name("h3-cuda-memory-lease.py"))
lease = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lease)
GIB = 1024**3
UUID = "GPU-12345678-1234-1234-1234-123456789abc"


def environment():
    return {"SW_F329_PLAN": json.dumps({"stationary_pair_authorized": True, "additionalSparse24": True,
                                      "expectedRunner": "cuda-windows-2", "sceneworks": "a" * 40,
                                      "inference": "b" * 40, "poolBytes": 21 * GIB,
                                      "externalReserveBytes": 3 * GIB}),
            "GITHUB_ACTIONS": "true", "GITHUB_EVENT_NAME": "workflow_dispatch", "GITHUB_SHA": "a" * 40,
            "RUNNER_NAME": "cuda-windows-2", "SCENEWORKS_GPU_ID": "1", "CUDA_VISIBLE_DEVICES": UUID,
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}


class LaunchTests(unittest.TestCase):
    def campaign(self, output, *, status=0, timeout=False, release=True, wait=False, exception=None, env=None):
        env = environment() if env is None else env
        index = json.loads(env["SW_F329_PLAN"]).get("physicalGpuIndex", 1)
        events = []
        class Job:
            def assign_and_resume(self, child): events.append("resume-owned")
            def terminate_and_reap(self, child, grace): events.append("reap-owned")
            def close(self): events.append("close-job")
        @contextmanager
        def claim(backend, policy, directory, env, **options):
            self.assertEqual(backend, "candle-cuda")
            self.assertEqual(options["constraints"], {"required_cuda_physical_index": index, "required_cuda_device_uuid": env["CUDA_VISIBLE_DEVICES"]})
            current_plan = json.loads(env['SW_F329_PLAN'])
            seconds = current_plan['timeoutSeconds'] + 1800 if current_plan.get('mode') else lease.CAMPAIGN_SECONDS
            self.assertEqual(options["original_deadline_monotonic"], 100 + seconds)
            self.assertEqual(policy.child_gpu_cap_bytes, json.loads(env['SW_F329_PLAN'])['poolBytes'])
            self.assertEqual(policy.gpu_free_reserve_bytes, 3 * GIB)
            self.assertEqual(policy.poll_millis, 500)
            if wait:
                events.extend(["busy-queue", "busy-queue"])
                self.assertNotIn("resume-owned", events)
            events.append("claim")
            record = {"post_execution_verified": False}
            try: yield record
            finally:
                self.assertNotIn("resume-owned", events[events.index("close-job") + 1:])
                events.append("release-check")
                record["post_execution_verified"] = release
        def execute(name, argv, directory, deadline, job_type):
            current_plan = json.loads(env['SW_F329_PLAN'])
            seconds = current_plan['timeoutSeconds'] + 1800 if current_plan.get('mode') else lease.CAMPAIGN_SECONDS
            self.assertEqual(deadline, 100 + seconds - lease.CLEANUP_SECONDS)
            self.assertEqual(argv, ["node", str(Path("SW") / "scripts" / "h3-cuda-memory-terminal.mjs"), "run", "SW", "ENGINE", str(output)])
            self.assertEqual(events[-1], "claim")
            job = job_type()
            try:
                job.assign_and_resume("child")
                if exception: raise exception
                if timeout: job.terminate_and_reap("child", 10)
                return {"returncode": None if timeout else status, "timeout": timeout}
            finally: job.close()
        try:
            return lease.supervise(Path("SW"), Path("ENGINE"), output, env, claim=claim,
                                   execute=execute, job_type=Job, clock=lambda: 100), events
        finally:
            self.assertEqual(events[-3:], ["reap-owned", "close-job", "release-check"])

    def test_success_waits_then_owns_entire_campaign_and_reaps_before_release(self):
        with tempfile.TemporaryDirectory() as temp:
            receipt, events = self.campaign(Path(temp) / "out", wait=True)
            self.assertTrue(receipt["passed"])
            self.assertEqual(events[:3], ["busy-queue", "busy-queue", "claim"])
            self.assertEqual(receipt["minimumClaimFreeBytes"], 24 * GIB)

    def test_nonzero_and_timeout_are_failed_receipts_after_cleanup(self):
        for args in [{"status": 9}, {"timeout": True}]:
            with self.subTest(args=args), tempfile.TemporaryDirectory() as temp:
                output = Path(temp) / "out"
                with self.assertRaises(RuntimeError): self.campaign(output, **args)
                self.assertFalse(json.loads(Path(str(output) + "-lease/campaign-lease.json").read_text(encoding="utf-8"))["passed"])

    def test_interrupt_reaps_then_propagates_and_seals_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "out"
            with self.assertRaises(KeyboardInterrupt): self.campaign(output, exception=KeyboardInterrupt())
            receipt = json.loads(Path(str(output) + "-lease/campaign-lease.json").read_text(encoding="utf-8"))
            self.assertFalse(receipt["passed"])
            self.assertIn("KeyboardInterrupt", receipt["error"])

    def test_unverified_release_cannot_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(RuntimeError, "post-execution"):
                self.campaign(Path(temp) / "out", release=False)

    def test_bad_context_and_visibility_refuse_before_any_claim_or_spawn(self):
        for change in [{"CUDA_VISIBLE_DEVICES": "1"}, {"SCENEWORKS_GPU_ID": "0"}, {"GITHUB_ACTIONS": "false"},
                       {"GITHUB_EVENT_NAME": "push"}, {"RUNNER_NAME": "cuda-windows-4"}, {"GITHUB_SHA": "c" * 40}]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                lease.validated_plan({**environment(), **change})

    def test_plan_preserves_exact_pool_and_external_reserve(self):
        for change in [{"poolBytes": 24 * GIB}, {"externalReserveBytes": GIB}, {"poolBytes": True},
                       {"stationary_pair_authorized": False}, {"additionalSparse24": False}]:
            env = environment()
            env["SW_F329_PLAN"] = json.dumps({**json.loads(env["SW_F329_PLAN"]), **change})
            with self.subTest(change=change), self.assertRaises(ValueError): lease.validated_plan(env)

    def test_failed_reap_closes_job_but_never_marks_reaped(self):
        events = []
        class Job:
            def assign_and_resume(self, child): pass
            def terminate_and_reap(self, child, grace): raise RuntimeError("unverified owned cleanup")
            def close(self): events.append("closed")
        job = lease.ReapedJob(Job)
        job.assign_and_resume("child")
        with self.assertRaisesRegex(RuntimeError, "unverified"): job.close()
        self.assertFalse(job.reaped)
        self.assertEqual(events, ["closed"])

    def test_explicit_zero_owns_exact_uuid_and_preserves_budgets_and_cleanup(self):
        env = environment()
        env.update(SCENEWORKS_GPU_ID="0", CUDA_VISIBLE_DEVICES=lease.GPU_UUIDS[0])
        env["SW_F329_PLAN"] = json.dumps({**json.loads(env["SW_F329_PLAN"]),
                                        "physicalGpuIndex": 0, "physicalGpuUuid": lease.GPU_UUIDS[0]})
        with tempfile.TemporaryDirectory() as temp:
            receipt, events = self.campaign(Path(temp) / "out", env=env)
            self.assertEqual(receipt["physicalGpuIndex"], 0)
            self.assertEqual(receipt["physicalGpuUuid"], lease.GPU_UUIDS[0])
            self.assertEqual(receipt["minimumClaimFreeBytes"], 24 * GIB)
            self.assertEqual(events[-3:], ["reap-owned", "close-job", "release-check"])

    def test_selection_type_uuid_visibility_and_retained_control_refuse_before_claim(self):
        valid = {"physicalGpuIndex": 0, "physicalGpuUuid": lease.GPU_UUIDS[0]}
        mutations = [{"physicalGpuIndex": x} for x in [-1, 2, True, False, 0.0, "0", None]]
        mutations += [{"physicalGpuUuid": x} for x in [None, "0", UUID, lease.GPU_UUIDS[1]]]
        mutations += [{"retainedWorkstation": {}}, {"physicalGpuIndex": 1}]
        for change in mutations:
            env = environment()
            env.update(SCENEWORKS_GPU_ID="0", CUDA_VISIBLE_DEVICES=lease.GPU_UUIDS[0])
            env["SW_F329_PLAN"] = json.dumps({**json.loads(env["SW_F329_PLAN"]), **valid, **change})
            with self.subTest(change=change), self.assertRaises(ValueError): lease.validated_plan(env)
        for change in [{"CUDA_VISIBLE_DEVICES": lease.GPU_UUIDS[1]}, {"SCENEWORKS_GPU_ID": "1"},
                       {"CUDA_DEVICE_ORDER": "FASTEST_FIRST"}]:
            env = environment()
            env.update(SCENEWORKS_GPU_ID="0", CUDA_VISIBLE_DEVICES=lease.GPU_UUIDS[0])
            env["SW_F329_PLAN"] = json.dumps({**json.loads(env["SW_F329_PLAN"]), **valid})
            with self.subTest(change=change), self.assertRaises(ValueError): lease.validated_plan({**env, **change})

    def test_single_matched_control_uses_exact_92_plus_3_claim_and_same_owned_cleanup(self):
        env = environment()
        plan = json.loads(env['SW_F329_PLAN'])
        plan.pop('additionalSparse24')
        plan.update(mode='matched-resident-control', matchedResidentControlAuthorized=True,
                    timeoutSeconds=7200,
                    inference='2ff0353820e370d004305ae99a035ab38b93de30',
                    physicalGpuIndex=0, physicalGpuUuid=lease.GPU_UUIDS[0],
                    poolBytes=92 * GIB, externalReserveBytes=3 * GIB)
        env.update(SCENEWORKS_GPU_ID='0', CUDA_VISIBLE_DEVICES=lease.GPU_UUIDS[0])
        env['SW_F329_PLAN'] = json.dumps(plan)
        with tempfile.TemporaryDirectory() as temp:
            receipt, events = self.campaign(Path(temp) / 'out', env=env)
            self.assertEqual(receipt['minimumClaimFreeBytes'], 95 * GIB)
            self.assertEqual(receipt['scope'], 'matched-resident-control')
            self.assertEqual(events[-3:], ['reap-owned', 'close-job', 'release-check'])
        for change in [{'poolBytes': 21 * GIB}, {'externalReserveBytes': 2 * GIB},
                       {'mode': 'unknown'}, {'matchedResidentControlAuthorized': False},
                       {'inference': 'b' * 40}, {'additionalSparse24': True}]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                lease.validated_plan({**env, 'SW_F329_PLAN': json.dumps({**plan, **change})})


if __name__ == "__main__":
    unittest.main()
