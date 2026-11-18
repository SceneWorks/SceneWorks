"""CPU mocks only: no driver query, device creation, model, or real child process."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("fixed_nvfp4_guard", HERE / "imported-nvfp4-cuda-guard.py")
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class Locker:
    LK_NBLCK = 2
    LK_UNLCK = 0

    def __init__(self, busy=0):
        self.busy = busy
        self.calls = []

    def locking(self, fd, mode, count):
        import os
        assert os.lseek(fd, 0, os.SEEK_CUR) == 0
        assert os.fstat(fd).st_size >= 1
        self.calls.append((mode, count))
        if mode == self.LK_NBLCK and self.busy:
            self.busy -= 1
            raise OSError("fixture lock contention")


class Clock:
    def __init__(self):
        self.value = 0

    def now(self):
        return self.value

    def sleep(self, amount):
        self.value += amount


class FixedCudaOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="resource-unit-", dir=HERE)
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        self.device = dict(index=1, uuid="GPU-one", total_mib=97887, used_mib=100, utilization_percent=0)
        self.owner = dict(index=0, uuid="GPU-owner", total_mib=97887, used_mib=100, utilization_percent=0)
        self.guard = Mock()
        self.guard.inventory.return_value = ([self.owner, self.device], [])
        self.guard.idle.side_effect = AssertionError("admission never asks for an idle card")
        self.guard.sample_baseline.side_effect = AssertionError("admission never uses the strict baseline")
        self.clock = Clock()
        self.window = (adapter.shared.SHARED_WINDOW_SAMPLES - 1) * adapter.shared.SHARED_WINDOW_INTERVAL_SECONDS

    def claim(self, locker, deadline=60):
        return adapter.claim_fixed(self.guard, self.out, deadline, self.out, locker,
                                   self.clock.now, self.clock.sleep)

    def test_exact_pinned_components_import_without_calling_windows_or_driver(self):
        guard = adapter.component("veda_cuda_guard.py", adapter.GUARD_SHA)
        windows = adapter.component("media_campaign_windows.py", adapter.WINDOWS_SHA)
        self.assertTrue(callable(guard.sample_baseline))
        self.assertTrue(callable(windows.WindowsJob))
        with self.assertRaisesRegex(RuntimeError, "component changed"):
            adapter.component("veda_cuda_guard.py", "0" * 64)

    def test_shared_uuid_byte0_lock_is_held_during_post_claim_reinventory(self):
        locker = Locker()
        device, handle, baseline, _ = self.claim(locker)
        self.assertEqual(device, self.device)
        self.assertFalse(handle.closed)
        self.assertEqual(locker.calls, [(locker.LK_NBLCK, 1)])
        self.assertEqual((self.out / "SceneWorks/gpu-locks/GPU-one.lock").read_bytes(), b"0")
        self.assertFalse((self.out / "SceneWorks/gpu-locks/GPU-owner.lock").exists())
        self.assertEqual(baseline["baseline_used_mib"], 100)
        self.assertTrue(json.loads((self.out / "claimed-fixed-device-shared-window.json").read_text())["admitted"])
        adapter.unlock(handle, locker)
        self.assertTrue(handle.closed)
        self.assertEqual(locker.calls[-1], (locker.LK_UNLCK, 1))

    def test_busy_fixed_gpu_never_falls_back_to_idle_owner_gpu0(self):
        self.device.update(utilization_percent=100)
        with self.assertRaises(TimeoutError):
            self.claim(Locker(), deadline=30)
        self.assertFalse((self.out / "SceneWorks/gpu-locks/GPU-owner.lock").exists())
        window = json.loads((self.out / "claimed-fixed-device-shared-window.json").read_text())
        self.assertFalse(window["admitted"]); self.assertEqual(window["summary"]["index"], 1)

    def test_lock_contention_waits_for_same_uuid_without_switching_device(self):
        locker = Locker(busy=2)
        device, handle, _, waited = self.claim(locker)
        self.assertEqual(device["uuid"], "GPU-one")
        self.assertEqual(waited, 2 + self.window)
        self.assertEqual(locker.calls, [(locker.LK_NBLCK, 1)] * 3)
        adapter.unlock(handle, locker)

    def test_identity_change_is_refused_without_new_device_selection(self):
        changed = dict(self.device, uuid="GPU-replaced")
        self.guard.inventory.side_effect = [([self.owner, self.device], []), ([self.owner, changed], [])]
        with self.assertRaisesRegex(RuntimeError, "UUID changed"):
            self.claim(Locker())

    def test_refused_window_releases_then_rechecks_same_uuid(self):
        busy = dict(self.device, utilization_percent=100)
        self.guard.inventory.side_effect = lambda *_: ([self.owner, busy if self.clock.value < 5 else self.device], [])
        locker = Locker()
        _, handle, _, _ = self.claim(locker)
        self.assertEqual(locker.calls[:3], [(2, 1), (0, 1), (2, 1)])
        adapter.unlock(handle, locker)

    def test_root_exit_does_not_release_job_while_descendants_remain(self):
        job = Mock()
        job.members.side_effect = [{22}, {23}, set()]
        child = Mock()
        child.wait.return_value = 0
        popen = Mock(return_value=child)
        result = adapter.execute_owned(["fixed-node", "fixed-proof.mjs"], self.out, 10,
                                       lambda: job, {}, popen, self.clock.now, self.clock.sleep)
        self.assertEqual(result["returncode"], 0)
        self.assertTrue(result["owned_tree_empty"])
        self.assertEqual(popen.call_args.kwargs["creationflags"], 4)
        job.assign_and_resume.assert_called_once_with(child)
        self.assertEqual(job.members.call_count, 3)
        job.close.assert_called_once()

    def test_timeout_terminates_and_reaps_only_the_owned_job(self):
        job = Mock()
        child = Mock()
        child.wait.side_effect = subprocess.TimeoutExpired("fixed-proof", 1)
        result = adapter.execute_owned(["fixed-proof"], self.out, 10, lambda: job, {},
                                       Mock(return_value=child), self.clock.now, self.clock.sleep)
        self.assertTrue(result["timeout"])
        self.assertIsNone(result["returncode"])
        job.terminate_and_reap.assert_called_once_with(child, 10)
        job.close.assert_called_once()

    def test_assignment_failure_never_resumes_unowned_child(self):
        job = Mock()
        job.assign_and_resume.side_effect = RuntimeError("incompatible enclosing job")
        child = Mock()
        with self.assertRaisesRegex(RuntimeError, "incompatible"):
            adapter.execute_owned(["fixed-proof"], self.out, 10, lambda: job, {},
                                  Mock(return_value=child), self.clock.now, self.clock.sleep)
        child.kill.assert_called_once()
        job.close.assert_called_once()

    def test_missing_child_evidence_cannot_count_as_accepted(self):
        self.assertFalse(adapter.publish(self.out, {"accepted": True}))
        evidence = json.loads((self.out / "artifact/evidence.json").read_text())
        self.assertFalse(evidence["accepted"])

    def test_success_requires_both_owned_child_and_physical_resource_proof(self):
        artifact = self.out / "artifact"
        artifact.mkdir()
        (artifact / "evidence.json").write_text(json.dumps({"schema_version": 1, "accepted": False, "body_proof_accepted": True,
                                                               "resource_verification_pending": True}))
        self.assertTrue(adapter.publish(self.out, {"accepted": True, "uuid": "GPU-one"}))
        evidence = json.loads((artifact / "evidence.json").read_text())
        self.assertTrue(evidence["accepted"])
        self.assertFalse(evidence["resource_verification_pending"])
        self.assertEqual(json.loads((self.out / "evidence.json").read_text()), evidence)
        self.assertEqual(evidence["resource_evidence"]["relative_path"], "physical-device.json")

    def test_failed_resource_proof_cannot_seal_completed_child_body(self):
        artifact = self.out / "artifact"
        artifact.mkdir()
        (artifact / "evidence.json").write_text(json.dumps({"schema_version": 1, "accepted": False, "body_proof_accepted": True,
                                                               "resource_verification_pending": True}))
        self.assertFalse(adapter.publish(self.out, {"accepted": False, "error": "post-child idle unverified"}))
        self.assertFalse(json.loads((artifact / "evidence.json").read_text())["accepted"])
        self.assertEqual(sorted(path.name for path in artifact.iterdir()), ["evidence.json", "physical-device.json"])


if __name__ == "__main__":
    unittest.main()
