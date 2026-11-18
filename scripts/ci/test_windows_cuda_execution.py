"""CPU-only tests of exact CUDA mapping ABI, multi-UUID contention and no retargeting."""
import ctypes
import importlib.util
import inspect
import io
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent

def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

shared = load("windows_cuda_execution")
metadata = load("windows_cuda_metadata")


class FakeDriver:
    def __init__(self, count=2):
        self.cuInit = Mock(return_value=0)
        def get_count(pointer):
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int)).contents.value = count
            return 0
        def get_device(pointer, ordinal):
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int)).contents.value = ordinal + 10
            return 0
        def get_uuid(pointer, device):
            ctypes.memmove(pointer, bytes(range((device - 10) * 16, (device - 10) * 16 + 16)), 16)
            return 0
        self.cuDeviceGetCount = Mock(side_effect=get_count)
        self.cuDeviceGet = Mock(side_effect=get_device)
        self.cuDeviceGetUuid = Mock(side_effect=get_uuid)


class CudaExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="execution-unit-", dir=HERE)
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        self.one = dict(index=0, uuid="GPU-one", total_mib=97887, used_mib=100, utilization_percent=0)
        self.two = dict(index=1, uuid="GPU-two", total_mib=97887, used_mib=100, utilization_percent=0)
        self.guard = Mock()
        self.guard.inventory.return_value = ([self.one, self.two], [])
        self.guard.idle.return_value = True
        self.guard.sample_baseline.side_effect = lambda device, out, phase: (device, {"uuid": device["uuid"]})

    def test_metadata_abi_uses_only_init_enumeration_and_uuid_without_nul_truncation(self):
        driver = FakeDriver()
        packet = metadata.visible_devices(driver)
        self.assertFalse(packet["context_created"])
        self.assertEqual(ctypes.sizeof(metadata.CUuuid), 16)
        self.assertEqual(metadata.CUuuid.bytes.offset, 0)
        self.assertEqual(packet["devices"][0]["uuid"], "GPU-00010203-0405-0607-0809-0a0b0c0d0e0f")
        self.assertEqual(set(vars(driver)), {"cuInit", "cuDeviceGetCount", "cuDeviceGet", "cuDeviceGetUuid"})
        self.assertEqual(driver.cuInit.call_args.args, (0,))
        self.assertEqual(driver.cuDeviceGetUuid.restype, ctypes.c_uint32)

    def test_no_visible_device_and_driver_failure_are_explicit_failures(self):
        with self.assertRaisesRegex(RuntimeError, "no CUDA-visible"):
            metadata.visible_devices(FakeDriver(count=0))
        driver = FakeDriver(); driver.cuInit.return_value = 35
        with self.assertRaisesRegex(RuntimeError, "CUDA status 35"):
            metadata.visible_devices(driver)
        driver.cuDeviceGetCount.assert_not_called()

    def probe(self, rows):
        packet = {"context_created": False,
                  "abi_source": {"checksum": metadata.ABI_CRATE_CHECKSUM}, "devices": rows}
        return Mock(return_value=SimpleNamespace(stdout=json.dumps(packet)))

    def test_cuda_ordinal_zero_is_not_assumed_to_be_nvml_index_zero(self):
        run = self.probe([{"ordinal": 0, "uuid": "GPU-two"}])
        environment = {"CUDA_VISIBLE_DEVICES": "1", "CUDA_DEVICE_ORDER": "FASTEST_FIRST"}
        binding = shared.resolve_binding(self.guard, self.out, time.monotonic() + 30,
                                         environment, probe_visible="0", run=run)
        self.assertEqual(binding["devices"][0]["index"], 1)
        self.assertEqual(binding["mapping"], [{"cuda_ordinal": 0, "uuid": "GPU-two", "nvml_index": 1}])
        self.assertEqual(run.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(run.call_args.kwargs["env"]["CUDA_DEVICE_ORDER"], "FASTEST_FIRST")
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "1", "original actor environment is never retargeted")

    def test_explicit_actor_nvml_to_uuid_binding_needs_no_ordinal_guess(self):
        run = Mock(side_effect=AssertionError("driver probe not applicable to actor-forced UUID"))
        binding = shared.resolve_binding(self.guard, self.out, time.monotonic()+30, {},
                                         physical_indices="0", run=run)
        self.assertEqual(binding["devices"], [self.one])
        self.assertEqual(binding["kind"], "actor-forced-nvml-uuid")
        run.assert_not_called()

    def test_unmatched_or_duplicate_actual_cuda_uuid_mapping_refuses(self):
        for rows in [[{"ordinal": 0, "uuid": "GPU-missing"}],
                     [{"ordinal": 0, "uuid": "GPU-two"}, {"ordinal": 1, "uuid": "GPU-two"}]]:
            with self.assertRaises(RuntimeError):
                shared.resolve_binding(self.guard, self.out, time.monotonic()+30, {},
                                       cuda_ordinals="all", run=self.probe(rows))

    def test_recheck_retains_ordered_actual_ordinal_uuid_identity(self):
        first = shared.resolve_binding(self.guard, self.out, time.monotonic()+30, {},
            cuda_ordinals="all", run=self.probe([{"ordinal": 0, "uuid": "GPU-one"}, {"ordinal": 1, "uuid": "GPU-two"}]))
        second = shared.resolve_binding(self.guard, self.out, time.monotonic()+30, {},
            cuda_ordinals="all", run=self.probe([{"ordinal": 0, "uuid": "GPU-two"}, {"ordinal": 1, "uuid": "GPU-one"}]))
        self.assertNotEqual(first["key"], second["key"], "same set with swapped ordinal roles is still a binding change")

    def test_two_gpu_partial_contention_releases_all_then_reacquires_sorted_exact_set(self):
        class Locks:
            LK_NBLCK, LK_UNLCK = 2, 0
            def __init__(self):
                self.acquires = 0; self.held = set(); self.calls = []
            def locking(self, fd, mode, count):
                self.calls.append((mode, count))
                if mode == self.LK_NBLCK:
                    self.acquires += 1
                    if self.acquires == 2: raise OSError("second UUID is already leased")
                    self.held.add(fd)
                else:
                    self.held.remove(fd)
        locks = Locks(); now = [0]; held_while_waiting = []
        def sleep(amount):
            held_while_waiting.append(len(locks.held));now[0] += amount
        devices, handles, _, _ = shared.claim_uuids(["GPU-two", "GPU-one"], self.guard,
            self.out, 10, self.out, locks, clock=lambda:now[0], sleep=sleep)
        self.assertEqual([row["uuid"] for row in devices], ["GPU-one", "GPU-two"])
        # The contention wait holds nothing; only the admission window then sleeps with both held.
        self.assertEqual(held_while_waiting[0], 0)
        self.assertEqual(set(held_while_waiting[1:]), {2})
        self.assertEqual(len(locks.held), 2)
        shared.release_all(handles, locks)
        self.assertEqual(locks.held, set())


ENVIRONMENT = {"GITHUB_EVENT_NAME": "workflow_dispatch", "GITHUB_WORKFLOW_REF":
    "SceneWorks/SceneWorks/.github/workflows/memory-catalog-campaign.yml@refs/heads/feature/test",
    "SCENEWORKS_GPU_ID": "1", "CUDA_VISIBLE_DEVICES": "1", "NVIDIA_VISIBLE_DEVICES": "1",
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}


class Locker:
    LK_NBLCK, LK_UNLCK = 2, 0
    def locking(self, *_): pass


class AdmissionRuleTests(unittest.TestCase):
    """One admission rule for every actor: no allow-list, no idle card, an optional GPU restriction."""
    setUp = CudaExecutionTests.setUp
    environment = ENVIRONMENT

    def test_the_wrapper_has_no_actor_allow_list_or_alternative_policy(self):
        for name in ("STRICT_POLICY", "FUNCTIONAL_POLICY", "FUNCTIONAL_ACTORS", "SHARED_DISPLAY_POLICY",
                     "SHARED_DISPLAY_ACTORS", "require_admission_policy", "require_free_memory"):
            self.assertFalse(hasattr(shared, name), name)
        self.assertEqual(shared.resource_receipt("any-actor", ["x"])["admission_policy"], shared.SHARED_POLICY)
        self.assertNotIn("--admission-policy", inspect.getsource(shared.main))
        self.assertNotIn("idle(", inspect.getsource(shared.claim_uuids))
        self.assertNotIn("sample_baseline", inspect.getsource(shared))
        self.assertNotIn("verify_idle", inspect.getsource(shared))

    def test_every_actor_is_admitted_on_a_shared_card_with_a_loaded_foreign_model(self):
        # The desktop app holds a 30 GiB model in a compute context on the display card.
        self.two.update(used_mib=31000, utilization_percent=2)
        self.guard.inventory.return_value = ([self.one, self.two], [["GPU-two", "57052", "SceneWorks.exe", "C"],
                                                                    ["GPU-two", "768", "dwm.exe", "C+G"]])
        for actor in ("memory-catalog-candle", "krea-anchor", "iris-two-card", "anything-new"):
            receipt = shared.resource_receipt(actor, ["x"])
            now = [0]
            selected, handles, baselines, _ = shared.claim_uuids(["GPU-two"], self.guard, self.out, 60, self.out,
                Locker(), clock=lambda: now[0], sleep=lambda seconds: now.__setitem__(0, now[0] + seconds))
            shared.release_all(handles, Locker())
            self.assertEqual(selected, [self.two])
            self.assertEqual(baselines["GPU-two"]["free_mib"], 97887 - 31000)
            self.assertIn(("57052", "SceneWorks.exe", "C"), baselines["GPU-two"]["co_tenants"])
            self.assertEqual(receipt["admission_policy"], shared.SHARED_POLICY)
        self.guard.idle.assert_not_called(); self.guard.sample_baseline.assert_not_called()

    def test_the_default_free_memory_requirement_applies_when_the_job_names_none(self):
        self.two.update(used_mib=97887 - shared.DEFAULT_MIN_FREE_MIB + 1)
        now = [0]
        with self.assertRaises(TimeoutError):
            shared.claim_uuids(["GPU-two"], self.guard, self.out, 30, self.out, Locker(),
                clock=lambda: now[0], sleep=lambda seconds: now.__setitem__(0, now[0] + seconds))
        for bad in (0, -1, None, 1 << 20):
            with self.assertRaises(RuntimeError):
                shared.claim_uuids(["GPU-two"], self.guard, self.out, 30, self.out, Locker(), min_free_mib=bad)

    def test_required_physical_index_refuses_any_other_bound_card(self):
        def binding(*indices):
            return {"devices": [{"index": index, "uuid": f"GPU-{index}"} for index in indices]}
        shared.require_physical_index(binding(1), 1)
        shared.require_physical_index(binding(0, 1), None)
        for bad in (binding(0), binding(0, 1), binding()):
            with self.assertRaisesRegex(RuntimeError, "not the required GPU 1"):
                shared.require_physical_index(bad, 1)
        # A probe mask that differs from the child's own mask would guard the wrong card.
        shared.require_physical_index(binding(1), 1, {"CUDA_VISIBLE_DEVICES": "1"}, "1")
        for environment in ({"CUDA_VISIBLE_DEVICES": "0"}, {}):
            with self.assertRaisesRegex(RuntimeError, "differs from the job's CUDA_VISIBLE_DEVICES"):
                shared.require_physical_index(binding(1), 1, environment, "1")

    def test_a_transient_inventory_error_is_logged_and_retried_not_fatal(self):
        import io
        calls = {"n": 0}
        def inventory(*_):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("driver XML and CSV GPU identities differ")
            return [self.one, self.two], []
        self.guard.inventory.side_effect = inventory
        stream = io.StringIO(); now = [0]
        selected, handles, _, _ = shared.claim_uuids(["GPU-two"], self.guard, self.out, 60, self.out, Locker(),
            clock=lambda: now[0], sleep=lambda seconds: now.__setitem__(0, now[0] + seconds),
            reporter=shared.WaitReporter(stream))
        shared.release_all(handles, Locker())
        self.assertEqual(selected, [self.two])
        self.assertIn("driver XML and CSV GPU identities differ", stream.getvalue())

    def test_uuid_contention_budget_and_index_recheck_remain_fail_closed(self):
        class Busy:
            LK_NBLCK, LK_UNLCK = 2, 0
            def __init__(self, busy): self.busy = busy; self.held = set()
            def locking(self, fd, mode, _):
                if mode == self.LK_NBLCK:
                    if self.busy: self.busy -= 1; raise OSError("same UUID is leased")
                    self.held.add(fd)
                else: self.held.remove(fd)
        for busy, deadline, succeeds in [(2, 30, True), (40, 20, False)]:
            locker = Busy(busy); now = [0]
            def claim():
                return shared.claim_uuids(["GPU-two"], self.guard, self.out, deadline,
                    self.out, locker, required_indices={"GPU-two": 1}, clock=lambda: now[0],
                    sleep=lambda seconds: now.__setitem__(0, now[0]+seconds))
            if succeeds:
                _, handles, _, waited = claim()
                self.assertEqual(waited, 2 + (shared.SHARED_WINDOW_SAMPLES - 1) * shared.SHARED_WINDOW_INTERVAL_SECONDS)
                self.assertEqual(len(locker.held), 1); shared.release_all(handles, locker)
            else:
                with self.assertRaises(TimeoutError): claim()
                self.assertEqual(now[0], deadline)
            self.assertEqual(locker.held, set())
        self.guard.inventory.return_value = ([dict(self.two, index=0)], [])
        with self.assertRaisesRegex(RuntimeError, "index/UUID binding changed"):
            shared.claim_uuids(["GPU-two"], self.guard, self.out, 10, self.out, Busy(0),
                required_indices={"GPU-two": 1}, clock=lambda: 0)

    def test_completion_requires_success_no_timeout_empty_owned_tree_and_never_claims_idle(self):
        baselines = {"GPU-two": {"baseline_used_mib": 100}}
        for result, accepted in [({"returncode": 0, "timeout": False, "owned_tree_empty": True}, True),
                ({"returncode": 3, "timeout": False, "owned_tree_empty": True}, False),
                ({"returncode": None, "timeout": True, "owned_tree_empty": True}, False),
                ({"returncode": 0, "timeout": False, "owned_tree_empty": False}, False)]:
            receipt = shared.resource_receipt("load-progress-acceptance", ["original-actor"])
            shared.complete_resource(receipt, result, [self.two], self.guard, self.out, baselines)
            self.assertEqual(receipt["resource_completed"], accepted)
            for key in ("post_child_idle_verified", "isolated_device", "model_inference_proof"):
                self.assertIs(receipt[key], False)
            self.assertTrue(receipt["shared_device"])
            self.assertEqual(receipt["post_child"]["devices"][0]["drift_mib"], 0)
        # The post-child sample is a diagnostic: a failed inventory never changes the verdict.
        self.guard.inventory.side_effect = RuntimeError("driver XML and CSV GPU identities differ")
        receipt = shared.resource_receipt("x", ["x"])
        shared.complete_resource(receipt, {"returncode": 0, "timeout": False, "owned_tree_empty": True},
                                 [self.two], self.guard, self.out, baselines)
        self.assertTrue(receipt["resource_completed"]); self.assertIn("error", receipt["post_child"])
        self.guard.verify_idle.assert_not_called()

    def test_main_validates_the_required_gpu_before_and_the_binding_after_the_lock(self):
        def binding(index=1, identifier="GPU-two"):
            return {"kind": "context-free-cuda-ordinal-uuid", "devices": [{"index": index, "uuid": identifier}],
                    "mapping": [{"cuda_ordinal": 0, "uuid": identifier, "nvml_index": index}], "key": [(0, identifier)]}
        valid = binding()
        cases = [(valid, valid, True), (binding(index=0), valid, False),
                 (valid, binding(identifier="GPU-remapped"), False)]
        for attempt, (before, after, accepted) in enumerate(cases, 1):
            env = dict(self.environment, GITHUB_RUN_ID=str(attempt), GITHUB_RUN_ATTEMPT="1",
                       RUNNER_TEMP=str(self.out), PUBLIC=str(self.out))
            executor = Mock(return_value={"returncode": 0, "timeout": False, "owned_tree_empty": True})
            claim = Mock(return_value=([self.two], [], {"GPU-two": {"baseline_used_mib": 100}}, 0))
            with patch.object(shared, "os", SimpleNamespace(name="nt", environ=env)), \
                 patch.object(shared.sys, "argv", ["wrapper", "--actor", "load-progress-acceptance",
                     "--budget-minutes", "1", "--cuda-ordinals", "0", "--probe-visible", "1",
                     "--require-physical-index", "1", "--min-free-mib", "28000", "--", "original-actor"]), \
                 patch.dict(shared.sys.modules, {"msvcrt": SimpleNamespace()}), \
                 patch.object(shared, "prepared_components", return_value=(self.guard, SimpleNamespace(WindowsJob=object))), \
                 patch.object(shared, "resolve_binding", side_effect=[before, after]), \
                 patch.object(shared, "claim_uuids", claim), \
                 patch.object(shared, "execute_owned", executor):
                self.assertEqual(shared.main(), 0 if accepted else 1)
            receipt = json.loads((self.out / f"sceneworks-cuda-execution-{attempt}-1-load-progress-acceptance/resource-receipt.json").read_text())
            self.assertEqual(receipt["resource_completed"], accepted)
            self.assertEqual(executor.call_count, int(accepted))
            self.assertIs(receipt["post_child_idle_verified"], False)
            if before["devices"][0]["index"] != 1:
                claim.assert_not_called()  # refused before taking the lock
            else:
                self.assertEqual(claim.call_args.kwargs["min_free_mib"], 28000)
                self.assertEqual(claim.call_args.kwargs["required_indices"], {"GPU-two": 1})
        self.guard.verify_idle.assert_not_called()

    def test_main_prints_its_own_failure_reason_and_child_exit_into_the_job_log(self):
        import contextlib, io
        cases = [(Mock(side_effect=TimeoutError("exact GPU UUID lock/idle wait exhausted the existing execution budget")),
                  {"error": "TimeoutError: exact GPU UUID lock/idle wait exhausted the existing execution budget"}),
                 (Mock(return_value=([self.two], [], {"GPU-two": {}}, 0)),
                  {"child": {"returncode": 2, "timeout": False, "owned_tree_empty": True}})]
        for attempt, (claim, expected) in enumerate(cases, 1):
            env = dict(self.environment, GITHUB_RUN_ID=str(40 + attempt), GITHUB_RUN_ATTEMPT="1",
                       RUNNER_TEMP=str(self.out), PUBLIC=str(self.out))
            binding = {"kind": "context-free-cuda-ordinal-uuid", "devices": [{"index": 1, "uuid": "GPU-two"}], "key": [(0, "GPU-two")]}
            printed = io.StringIO()
            with patch.object(shared, "os", SimpleNamespace(name="nt", environ=env)), \
                 patch.object(shared.sys, "argv", ["wrapper", "--actor", "memory-catalog-candle", "--budget-minutes", "1",
                     "--cuda-ordinals", "0", "--", "bash", "scripts/ci/memory-catalog/run.sh"]), \
                 patch.dict(shared.sys.modules, {"msvcrt": SimpleNamespace()}), \
                 patch.object(shared, "prepared_components", return_value=(self.guard, SimpleNamespace(WindowsJob=object))), \
                 patch.object(shared, "resolve_binding", return_value=binding), \
                 patch.object(shared, "claim_uuids", claim), \
                 patch.object(shared, "release_transaction"), \
                 patch.object(shared, "execute_owned", Mock(return_value=expected.get("child"))), \
                 contextlib.redirect_stdout(printed):
                self.assertNotEqual(shared.main(), 0)
            line = json.loads(printed.getvalue().strip().splitlines()[-1])
            self.assertIs(line["resource_completed"], False)
            for key, value in expected.items():
                self.assertEqual(line[key], value)

    def test_owned_timeout_terminates_only_the_assigned_job_and_waits_for_empty_tree(self):
        child = Mock(); child.wait.side_effect = subprocess.TimeoutExpired("owned", 1)
        job = Mock()
        result = shared.execute_owned(["owned"], self.out, time.monotonic()+10,
            lambda: job, {}, popen=Mock(return_value=child))
        job.assign_and_resume.assert_called_once_with(child)
        job.terminate_and_reap.assert_called_once_with(child, 10)
        job.close.assert_called_once()
        child.kill.assert_not_called()
        self.assertTrue(result["timeout"]); self.assertTrue(result["owned_tree_empty"])


def fake_smi_xml(gpu1_used, processes1, gpu0_used=50220, gpu0_processes=(("40052", "C", "N/A"),)):
    """Shape of `nvidia-smi -q -x` on the Windows host: GPU0 at bus 21, display GPU1 at bus C1."""
    def gpu(bus, uuid, used, attached, processes):
        rows = "".join(f"<process_info><pid>{pid}</pid><type>{kind}</type><process_name>C:\\x\\p{pid}.exe</process_name>"
                       f"<used_memory>{memory}</used_memory></process_info>" for pid, kind, memory in processes)
        return (f'<gpu id="{bus}"><product_name>NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition</product_name>'
                f"<display_attached>{attached}</display_attached><display_active>Disabled</display_active>"
                f"<driver_model><current_dm>WDDM</current_dm></driver_model><uuid>{uuid}</uuid>"
                f"<fb_memory_usage><total>97887 MiB</total><used>{used} MiB</used><free>{97887 - used} MiB</free></fb_memory_usage>"
                f"<utilization><gpu_util>12 %</gpu_util></utilization><processes>{rows}</processes></gpu>")
    return ("<nvidia_smi_log>" + gpu("00000000:21:00.0", "GPU-one", gpu0_used, "No", gpu0_processes)
            + gpu("00000000:C1:00.0", "GPU-two", gpu1_used, "Yes", processes1) + "</nvidia_smi_log>")


DISPLAY = [(str(pid), "C+G", "N/A") for pid in (768, 13284, 28124, 39752)]


class SharedWindowAdmissionTests(unittest.TestCase):
    """Fake nvidia-smi output only: the one free-memory/stability window, for every actor."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="shared-window-", dir=HERE)
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)

    def guard_for(self, used_series, processes1, utilization=12):
        """A guard whose inventory replays one fake nvidia-smi snapshot per call (CSV rows + XML file)."""
        calls = {"n": 0}
        guard = Mock()
        def inventory(out, label, deadline):
            used = used_series(calls["n"]) if callable(used_series) else used_series[min(calls["n"], len(used_series) - 1)]
            calls["n"] += 1
            (out / f"{label}-nvidia-smi.xml").write_text(fake_smi_xml(used, processes1), encoding="utf-8")
            devices = [dict(index=0, uuid="GPU-one", total_mib=97887, used_mib=50220, utilization_percent=99),
                       dict(index=1, uuid="GPU-two", total_mib=97887, used_mib=used, utilization_percent=utilization)]
            processes = [["GPU-one", "40052", "p40052.exe", "C"]] + [["GPU-two", pid, f"p{pid}.exe", kind] for pid, kind, _ in processes1]
            return devices, processes
        guard.inventory.side_effect = inventory
        guard.idle.side_effect = AssertionError("admission never asks for an idle card")
        guard.sample_baseline.side_effect = AssertionError("admission never uses the strict benchmark baseline")
        return guard

    def claim(self, guard, deadline=60, min_free_mib=75776):
        now = [0]
        result = shared.claim_uuids(["GPU-two"], guard, self.out, deadline, self.out, Locker(),
            required_indices={"GPU-two": 1}, clock=lambda: now[0],
            sleep=lambda seconds: now.__setitem__(0, now[0] + seconds), min_free_mib=min_free_mib)
        shared.release_all(result[1], Locker())
        return result

    def window(self):
        return json.loads((self.out / "claimed-device-1-shared-window.json").read_text())

    def test_display_baseline_with_a_context_holder_is_admitted(self):
        # Run 43287950598 at lease start: 5847 MiB used, 12% util, display processes and one
        # compute context (WDDM per-process memory N/A).
        guard = self.guard_for([5847] * 10, DISPLAY + [("40068", "C", "N/A")])
        selected, _, baselines, _ = self.claim(guard)
        self.assertEqual(selected[0]["uuid"], "GPU-two")
        summary = baselines["GPU-two"]
        self.assertEqual(summary["baseline_used_mib"], 5847); self.assertEqual(summary["free_mib"], 97887 - 5847)
        self.assertTrue(self.window()["admitted"])

    def test_a_foreign_model_held_by_a_compute_process_is_admitted_when_the_job_fits(self):
        # Run 43287950598 at cancel: 21131 MiB used with a foreign compute process present. The old
        # shared-display rule refused this (> 8 GiB with a compute process); the job only needs free memory.
        guard = self.guard_for([21131] * 10, DISPLAY + [("40068", "C", "N/A")])
        _, _, baselines, _ = self.claim(guard)
        self.assertEqual(baselines["GPU-two"]["free_mib"], 97887 - 21131)
        self.assertIn(("40068", "p40068.exe", "C"), baselines["GPU-two"]["co_tenants"])
        # Even where the driver reports a large per-process footprint, only free memory decides.
        guard = self.guard_for([40000] * 10, DISPLAY + [("777", "C", "34000 MiB")])
        self.claim(guard, min_free_mib=50000)
        self.assertEqual(self.window()["reasons"], [])

    def test_growing_or_unstable_usage_busy_device_and_too_little_free_memory_are_refused(self):
        # A foreign job that keeps growing or oscillating never settles, so every window refuses.
        for series, needle in [(lambda n: 6000 + 20 * n, "growing"),
                               (lambda n: 6000 + 600 * (n % 2), "unstable"),
                               ([30000] * 10, "free 67887 MiB < required 75776 MiB")]:
            guard = self.guard_for(series, DISPLAY)
            with self.assertRaisesRegex(TimeoutError, "no admission within the budget"):
                self.claim(guard, deadline=30)
            self.assertTrue(any(needle in reason for reason in self.window()["reasons"]), (needle, self.window()))
        guard = self.guard_for([5847] * 10, DISPLAY, utilization=45)
        with self.assertRaises(TimeoutError): self.claim(guard, deadline=30)
        self.assertTrue(any("utilization 45%" in reason for reason in self.window()["reasons"]))

    def test_receipt_identities_name_both_gpus_with_display_and_process_types(self):
        rows = shared.gpu_identities(fake_smi_xml(5847, DISPLAY + [("40068", "C", "N/A")]))
        self.assertEqual([(row["pci_bus_id"], row["uuid"], row["display_attached"]) for row in rows],
                         [("00000000:21:00.0", "GPU-one", "No"), ("00000000:C1:00.0", "GPU-two", "Yes")])
        self.assertEqual([p["type"] for p in rows[1]["processes"]].count("C+G"), 4)
        receipt = shared.resource_receipt("tier2-ab-cuda", ["node"])
        self.assertTrue(receipt["shared_device"]); self.assertIn("baseline", receipt["observation_note"])
        guard = self.guard_for([5847], DISPLAY)
        shared.complete_resource(receipt, {"returncode": 0, "timeout": False, "owned_tree_empty": True},
                                 [{"index": 1, "uuid": "GPU-two"}], guard, self.out,
                                 {"GPU-two": {"baseline_used_mib": 5800}})
        self.assertTrue(receipt["resource_completed"]); self.assertIs(receipt["post_child_idle_verified"], False)
        self.assertEqual(receipt["post_child"]["devices"][0]["drift_mib"], 47)


class HostServiceTransactionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="service-unit-", dir=HERE)
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        outer = self
        class Locker:
            LK_NBLCK, LK_UNLCK = 2, 0
            def __init__(self):
                self.held = {}; self.events = []
            def locking(self, fd, mode, count):
                outer.assertEqual(os.lseek(fd, 0, os.SEEK_CUR), 0)
                outer.assertEqual(count, 1)
                inode = os.fstat(fd).st_ino
                name = next(p.name for p in outer.out.rglob("*.lock") if p.stat().st_ino == inode)
                if mode == self.LK_NBLCK:
                    if inode in self.held: raise OSError("another host transaction owns this byte")
                    self.held[inode] = fd
                    self.events.append((name, "acquire"))
                else:
                    outer.assertEqual(self.held.pop(inode), fd)
                    self.events.append((name, "release"))
        self.locker = Locker()
        self.now = 0

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds

    def claim(self, url, deadline=10):
        return shared.claim_service(url, self.out, deadline, self.locker, self.clock, self.sleep)

    def test_same_port_excludes_different_gpu_transactions_and_loopback_aliases(self):
        first, _ = self.claim("http://127.0.0.1:17821")
        second_gpu = Mock()
        def contender_on_another_gpu():
            handle, _ = self.claim("http://localhost:17821", deadline=2)
            try: second_gpu("GPU-different")
            finally: shared.unlock(handle, self.locker)
        with self.assertRaises(TimeoutError):
            contender_on_another_gpu()
        second_gpu.assert_not_called()
        self.assertFalse(first.closed)
        shared.unlock(first, self.locker)

    def test_different_custom_ports_do_not_conflict_or_substitute_defaults(self):
        first, one = self.claim("http://127.0.0.1:17821")
        second, two = self.claim("http://[::1]:19444/custom")
        self.assertEqual((one["port"], two["port"]), (17821, 19444))
        self.assertEqual(self.now, 0)
        shared.unlock(second, self.locker); shared.unlock(first, self.locker)

    def test_service_before_gpu_and_reverse_release_order(self):
        service, _ = self.claim("http://127.0.0.1:17821")
        guard = Mock()
        rows = [dict(index=0, uuid="GPU-a", total_mib=97887, used_mib=100, utilization_percent=0),
                dict(index=1, uuid="GPU-b", total_mib=97887, used_mib=100, utilization_percent=0)]
        guard.inventory.return_value = (rows, [])
        _, handles, _, _ = shared.claim_uuids(["GPU-b", "GPU-a"], guard, self.out,
            10, self.out, self.locker, clock=self.clock, sleep=self.sleep)
        shared.release_transaction(handles, service, self.locker)
        self.assertEqual(self.locker.events, [
            ("loopback-port-17821.lock", "acquire"), ("GPU-a.lock", "acquire"), ("GPU-b.lock", "acquire"),
            ("GPU-b.lock", "release"), ("GPU-a.lock", "release"), ("loopback-port-17821.lock", "release")])
        source = inspect.getsource(shared.main)
        self.assertLess(source.index("claim_service("), source.index("claim_uuids("))
        self.assertLess(source.index("execute_owned("), source.index("release_transaction("))

    def test_nested_boundary_refuses_without_reacquiring_or_trusting_marker(self):
        shared.require_outermost({})
        for value in ["parent-controller", "", "not-an-authority-token"]:
            with self.assertRaisesRegex(RuntimeError, "nested"):
                shared.require_outermost({shared.ACTIVE_KEY: value})
        job = Mock(); job.members.return_value = set()
        child = Mock(); child.wait.return_value = 0
        popen = Mock(return_value=child)
        original = {"CUDA_VISIBLE_DEVICES": "0,1"}
        shared.execute_owned(["owned-command"], self.out, time.monotonic()+10,
                             lambda:job, original, popen=popen, capture=False)
        inherited = popen.call_args.kwargs["env"]
        self.assertEqual(inherited["CUDA_VISIBLE_DEVICES"], "0,1")
        self.assertNotIn(shared.ACTIVE_KEY, original)
        with self.assertRaisesRegex(RuntimeError, "nested"):
            shared.require_outermost(inherited)
        self.assertEqual(self.locker.events, [])

    def test_non_loopback_credentials_and_implicit_ports_are_refused(self):
        for url in ["http://localhost", "http://127.0.0.1:0", "http://example.test:17821",
                    "http://user:secret@localhost:17821", "http://localhost:17821?token=secret"]:
            with self.subTest(url=url), self.assertRaises((RuntimeError, ValueError)):
                shared.service_identity(url)


class SharedKernelTests(unittest.TestCase):
    def setUp(self):
        self.kernel = load("cuda_quant_kernel_validation")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        self.good_output = "test cuda_qmatmul_matches_cpu ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out;\n"
        self.device = {"index": 1, "uuid": "GPU-two"}
        self.binding = {"devices": [self.device], "key": [(0, "GPU-two")]}
        self.resolve = Mock(return_value=self.binding)
        self.execute = Mock(side_effect=self.child)
        self.result = {"returncode": 0, "timeout": False, "owned_tree_empty": True}

    def child(self, args, out, deadline, job_type, env):
        (out / "proof.stdout.log").write_text(self.good_output, encoding="utf-8")
        self.assertEqual(tuple(args), self.kernel.COMMAND)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "GPU-two")
        self.assertEqual(env["SCENEWORKS_CUDA_KERNEL_REQUIRED"], "1")
        return self.result

    def validate(self, index="1"):
        return self.kernel.validate(index, {}, self.out, time.monotonic()+30, object(), object(),
                                    resolve=self.resolve, execute=self.execute)

    def test_shared_identity_queries_no_memory_utilization_or_processes(self):
        run = Mock(return_value=SimpleNamespace(stdout="0, GPU-one\n1, GPU-two\n"))
        identity = self.kernel.DeviceIdentity("trusted-smi", run)
        devices, processes = identity.inventory(self.out, "binding", time.monotonic()+30)
        self.assertEqual(devices, [{"index": 0, "uuid": "GPU-one"}, self.device])
        self.assertEqual(processes, [])
        self.assertEqual(run.call_args.args[0], ["trusted-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"])
        for invalid in ["0, GPU-one\n0, GPU-two", "0, GPU-one\n1, GPU-one", "1, invalid", ""]:
            run.return_value.stdout = invalid
            with self.assertRaises(RuntimeError):
                identity.inventory(self.out, "binding", time.monotonic()+30)

    def test_fixed_body_selected_physical_zero_or_one_and_owned_cleanup_required(self):
        for index in ("0", "1"):
            self.validate(index)
            self.assertEqual(self.resolve.call_args_list[-3].kwargs, {"physical_indices": index})
        self.assertIn("--ignored", self.kernel.COMMAND)
        self.assertIn("--exact", self.kernel.COMMAND)
        for result in [{"returncode": 3, "timeout": False, "owned_tree_empty": True},
                       {"returncode": 0, "timeout": True, "owned_tree_empty": True},
                       {"returncode": 0, "timeout": False, "owned_tree_empty": False}]:
            self.result = result
            with self.assertRaisesRegex(RuntimeError, "owned kernel child"):
                self.validate()

    def test_no_wrong_device_fallback_or_zero_match_false_green(self):
        self.resolve.side_effect = [self.binding, {"devices": [{"index": 0, "uuid": "GPU-one"}], "key": [(0, "GPU-one")]}]
        with self.assertRaisesRegex(RuntimeError, "physical index/UUID"):
            self.validate()
        self.execute.assert_not_called()
        self.resolve.side_effect = [self.binding, self.binding,
                                   {"devices": [self.device], "key": [(0, "GPU-remapped")]}]
        with self.assertRaisesRegex(RuntimeError, "mapping changed"):
            self.validate()
        self.resolve.side_effect = None
        self.good_output = "test result: ok. 0 passed; 0 failed; 0 ignored; 1 filtered out;\n"
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            self.validate()
        with self.assertRaisesRegex(RuntimeError, "explicit physical"):
            self.validate("2")

    def test_only_dedicated_dispatch_can_enter_shared_route(self):
        good = {"GITHUB_EVENT_NAME": "workflow_dispatch", "GITHUB_WORKFLOW_REF":
                "SceneWorks/SceneWorks/.github/workflows/windows-candle.yml@refs/heads/main",
                "KERNEL_DISPATCH_INPUTS": json.dumps({"quant_kernel_only": True, "quant_kernel_physical_index": "1"})}
        self.kernel.require_dispatch(good)
        for wrong in [dict(good, KERNEL_DISPATCH_INPUTS=json.dumps({"quant_kernel_only": True, "quant_kernel_physical_index": "1", "run_iris_worker_smoke": True})),
                      dict(good, KERNEL_DISPATCH_INPUTS="{}"), dict(good, GITHUB_EVENT_NAME="push"), dict(good, GITHUB_WORKFLOW_REF="other"),
                      dict(good, **{shared.ACTIVE_KEY: "1"})]:
            with self.assertRaises(RuntimeError):
                self.kernel.require_dispatch(wrong)


class WaitReportingTests(unittest.TestCase):
    """Every admission wait is visible in the job log; reporting never changes admission."""
    setUp = CudaExecutionTests.setUp

    def lines(self, stream):
        prefix = "cuda-execution: "
        return [json.loads(line[len(prefix):]) for line in stream.getvalue().splitlines() if line.startswith(prefix)]

    def test_an_admission_wait_prints_the_refusal_reason_every_interval_then_reports_the_end(self):
        import io
        busy = dict(self.two, used_mib=65000, utilization_percent=100)
        foreign = [["GPU-two", "57052", "python.exe", "C"]]
        now = [0.0]
        # A busy card for 150 s, then the foreign job exits and the device is quiet.
        self.guard.inventory.side_effect = lambda *_: (
            ([self.one, busy], foreign) if now[0] < 150 else ([self.one, self.two], []))
        stream = io.StringIO()
        devices, handles, _, waited = shared.claim_uuids(["GPU-two"], self.guard, self.out, 3600, self.out,
            Locker(), clock=lambda: now[0], sleep=lambda seconds: now.__setitem__(0, now[0] + seconds),
            reporter=shared.WaitReporter(stream))
        shared.release_all(handles, Locker())
        self.assertEqual(devices, [self.two]); self.assertEqual(waited, 159)
        lines = self.lines(stream)
        self.assertEqual([line["event"] for line in lines], ["wait-start", "waiting", "waiting", "wait-end"])
        self.assertEqual([line.get("elapsed_seconds") for line in lines[:3]], [9, 69, 129])
        for line in lines[:3]:
            self.assertEqual(line["wait"], "admission")
            self.assertEqual(line["devices"], [{"index": 1, "uuid": "GPU-two"}])
            refused = line["refused"]
            self.assertEqual((refused["index"], refused["uuid"]), (1, "GPU-two"))
            self.assertIn("utilization 100% > 30%", refused["reasons"])
            self.assertEqual(refused["window"]["free_mib"], 97887 - 65000)
        self.assertEqual(lines[3], {"event": "wait-end", "scope": "gpu", "wait_seconds": 159,
                                    "devices": [{"index": 1, "uuid": "GPU-two"}]})

    def test_uuid_lock_and_service_lock_waits_print_their_lines(self):
        import io
        class Locker:
            LK_NBLCK, LK_UNLCK = 2, 0
            def __init__(self, busy): self.busy = busy
            def locking(self, fd, mode, _):
                if mode == self.LK_NBLCK and self.busy:
                    self.busy -= 1; raise OSError("leased elsewhere")
        now = [0.0]
        clock, sleep = (lambda: now[0]), (lambda seconds: now.__setitem__(0, now[0] + seconds))
        stream = io.StringIO()
        reporter = shared.WaitReporter(stream)
        service, info = shared.claim_service("http://127.0.0.1:17821", self.out, 600, Locker(70), clock, sleep,
                                             reporter=reporter)
        _, handles, _, waited = shared.claim_uuids(["GPU-two"], self.guard, self.out, 600, self.out, Locker(61),
            clock=clock, sleep=sleep, reporter=reporter)
        shared.release_transaction(handles, service, Locker(0))
        window = (shared.SHARED_WINDOW_SAMPLES - 1) * shared.SHARED_WINDOW_INTERVAL_SECONDS
        self.assertEqual((info["wait_seconds"], waited), (70, 61 + window))
        lines = self.lines(stream)
        self.assertEqual([(line["event"], line.get("wait") or line.get("scope")) for line in lines],
            [("wait-start", "service"), ("waiting", "service"), ("wait-end", "service"),
             ("wait-start", "lock"), ("waiting", "lock"), ("wait-end", "gpu")])
        self.assertEqual(lines[0]["port"], 17821)
        self.assertTrue(lines[0]["lock_path"].endswith("loopback-port-17821.lock"))
        self.assertEqual(lines[1]["elapsed_seconds"], 60)
        self.assertEqual(lines[3]["held_lock"]["uuid"], "GPU-two")
        self.assertEqual(lines[3]["devices"], [{"index": 1, "uuid": "GPU-two"}])
        self.assertEqual(lines[4]["elapsed_seconds"], 60)

    def test_main_prints_the_launch_line_and_a_cancel_prints_the_wait_and_partial_receipt(self):
        import contextlib, io
        env = {"GITHUB_RUN_ID": "77", "GITHUB_RUN_ATTEMPT": "1", "RUNNER_TEMP": str(self.out), "PUBLIC": str(self.out)}
        binding = {"kind": "context-free-cuda-ordinal-uuid", "devices": [{"index": 1, "uuid": "GPU-two"}], "key": [(0, "GPU-two")]}
        printed, logged = io.StringIO(), io.StringIO()
        executor = Mock(return_value={"returncode": 0, "timeout": False, "owned_tree_empty": True})
        installed = []
        with patch.object(shared, "os", SimpleNamespace(name="nt", environ=env)), \
             patch.object(shared.sys, "argv", ["wrapper", "--actor", "memory-catalog-candle", "--budget-minutes", "1",
                 "--cuda-ordinals", "0", "--", "bash", "run.sh"]), \
             patch.dict(shared.sys.modules, {"msvcrt": SimpleNamespace()}), \
             patch.object(shared, "prepared_components", return_value=(self.guard, SimpleNamespace(WindowsJob=object))), \
             patch.object(shared, "resolve_binding", return_value=binding), \
             patch.object(shared, "claim_uuids", return_value=([self.two], [], {"GPU-two": {}}, 12.34)), \
             patch.object(shared, "complete_resource", return_value=0), \
             patch.object(shared, "release_transaction"), \
             patch.object(shared, "install_cancel_handler", side_effect=lambda handler: installed.append(handler) or {}), \
             patch.object(shared, "execute_owned", executor), \
             contextlib.redirect_stdout(printed), contextlib.redirect_stderr(logged):
            shared.main()
        launch = self.lines(logged)[-1]
        self.assertEqual(launch["event"], "launch")
        self.assertEqual(launch["lock_wait_seconds"], 12.3)
        self.assertEqual(launch["devices"], [{"index": 1, "uuid": "GPU-two"}])
        self.assertEqual(executor.call_count, 1)

        # A cancel mid-wait prints where it was and the receipt so far, then unwinds once.
        stream = io.StringIO()
        reporter = shared.WaitReporter(stream)
        reporter.waiting("admission", 90, 0, 600, [self.two], lambda: {"refused": {}})
        receipt = shared.resource_receipt("memory-catalog-candle", ["bash", "run.sh"])
        receipt["binding"] = binding
        handler = shared.cancel_handler(receipt, reporter)
        with self.assertRaises(KeyboardInterrupt):
            handler(shared.signal.SIGTERM, None)
        handler(shared.signal.SIGINT, None)  # a second signal only prints; cleanup is not interrupted
        cancel, again = self.lines(stream)[1:]
        self.assertEqual((cancel["event"], cancel["signal"]), ("cancel", "SIGTERM"))
        self.assertEqual(cancel["wait"]["wait"], "admission")
        self.assertEqual(cancel["wait"]["elapsed_seconds"], 90)
        self.assertEqual(cancel["partial_receipt"]["binding"]["devices"], [{"index": 1, "uuid": "GPU-two"}])
        self.assertNotIn("argv", cancel["partial_receipt"])
        self.assertEqual(again["signal"], "SIGINT")

    def run_main(self, run_id, execute, release, actor="memory-catalog-candle"):
        """Drive main with the real cancel handler captured; return (code, stdout, stderr, receipt)."""
        import contextlib, io
        env = {"GITHUB_RUN_ID": str(run_id), "GITHUB_RUN_ATTEMPT": "1", "RUNNER_TEMP": str(self.out), "PUBLIC": str(self.out)}
        binding = {"kind": "context-free-cuda-ordinal-uuid", "devices": [{"index": 1, "uuid": "GPU-two"}], "key": [(0, "GPU-two")]}
        printed, logged = io.StringIO(), io.StringIO()
        self.installed, self.handles = [], [object()]
        with patch.object(shared, "os", SimpleNamespace(name="nt", environ=env)), \
             patch.object(shared.sys, "argv", ["wrapper", "--actor", actor, "--budget-minutes", "1",
                 "--cuda-ordinals", "0", "--", "bash", "run.sh"]), \
             patch.dict(shared.sys.modules, {"msvcrt": SimpleNamespace()}), \
             patch.object(shared, "prepared_components", return_value=(self.guard, SimpleNamespace(WindowsJob=object))), \
             patch.object(shared, "resolve_binding", return_value=binding), \
             patch.object(shared, "claim_uuids", return_value=([self.two], self.handles, {"GPU-two": {}}, 0)), \
             patch.object(shared, "complete_resource", return_value=0), \
             patch.object(shared, "release_transaction", release), \
             patch.object(shared, "install_cancel_handler", side_effect=lambda handler: self.installed.append(handler) or {}), \
             patch.object(shared, "execute_owned", execute), \
             contextlib.redirect_stdout(printed), contextlib.redirect_stderr(logged):
            code = shared.main()
        receipt = json.loads((self.out / f"sceneworks-cuda-execution-{run_id}-1-{actor}/resource-receipt.json").read_text())
        return code, printed.getvalue(), logged.getvalue(), receipt

    def test_sdxl_proof_keeps_private_receipt_and_raw_error_out_of_job_logs(self):
        secret = str(self.out / "private-plan.json")
        code, printed, logged, receipt = self.run_main(83, Mock(side_effect=RuntimeError(secret)),
                                                        Mock(), actor="sdxl-exact-replay")
        self.assertEqual(code, 1)
        self.assertIn(secret, receipt["error"])
        self.assertNotIn(secret, printed + logged)
        self.assertNotIn("resource_receipt", printed)
        self.assertEqual(json.loads(printed.strip().splitlines()[-1])["error_class"], "RuntimeError")

        stream = io.StringIO()
        reporter = shared.WaitReporter(stream, private=True)
        reporter.state = {"wait": "admission", "refused": {"reasons": [secret]}}
        reporter.emit("wait-start", reporter.state)
        receipt["binding"] = {"private_path": secret}
        handler = shared.cancel_handler(receipt, reporter)
        with self.assertRaises(KeyboardInterrupt):
            handler(shared.signal.SIGTERM, None)
        self.assertNotIn(secret, stream.getvalue())
        self.assertNotIn("partial_receipt", stream.getvalue())

    def test_a_cancel_during_the_child_releases_once_writes_the_receipt_and_ends_with_the_summary(self):
        release = Mock()
        execute = Mock(side_effect=lambda *_, **__: self.installed[0](shared.signal.SIGTERM, None))
        code, printed, logged, receipt = self.run_main(81, execute, release)
        self.assertEqual(code, 1)
        release.assert_called_once()
        self.assertIs(release.call_args.args[0], self.handles)
        self.assertEqual(receipt["error"], "KeyboardInterrupt: cancelled by signal %d" % shared.signal.SIGTERM)
        summary = json.loads(printed.strip().splitlines()[-1])
        self.assertIs(summary["resource_completed"], False)
        self.assertTrue(summary["error"].startswith("KeyboardInterrupt"))
        self.assertIn('"event": "cancel"', logged)

    def test_a_first_signal_during_cleanup_only_prints_and_never_interrupts_release_or_receipt(self):
        released = []
        def release(handles, service, locker):
            self.installed[0](shared.signal.SIGTERM, None)  # Actions cancel lands mid-release
            released.append(handles)
        execute = Mock(return_value={"returncode": 0, "timeout": False, "owned_tree_empty": True})
        code, printed, logged, receipt = self.run_main(82, execute, release)
        self.assertEqual(released, [self.handles])
        self.assertNotIn("error", receipt)
        self.assertEqual(json.loads(printed.strip().splitlines()[-1])["resource_receipt"].split("/")[-1], "resource-receipt.json")
        self.assertIn('"event": "cancel"', logged)

    def test_install_skips_missing_signals_tolerates_off_main_thread_and_restores(self):
        calls = []
        def fake_signal(number, handler):
            calls.append((number, handler)); return f"previous-{number}"
        posix = SimpleNamespace(SIGTERM=15, SIGINT=2, signal=fake_signal)  # no SIGBREAK off Windows
        handler = object()
        previous = shared.install_cancel_handler(handler, posix)
        self.assertEqual(calls, [(15, handler), (2, handler)])
        self.assertEqual(previous, {15: "previous-15", 2: "previous-2"})
        calls.clear()
        shared.restore_signal_handlers(previous, posix)
        self.assertEqual(calls, [(15, "previous-15"), (2, "previous-2")])
        def off_main_thread(number, handler):
            raise ValueError("signal only works in main thread")
        windows = SimpleNamespace(SIGTERM=15, SIGINT=2, SIGBREAK=21, signal=off_main_thread)
        self.assertEqual(shared.install_cancel_handler(handler, windows), {})
        shared.restore_signal_handlers({15: None}, windows)  # never raises


if __name__ == "__main__":
    unittest.main()
