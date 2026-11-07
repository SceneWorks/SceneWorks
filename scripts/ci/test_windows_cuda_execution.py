"""CPU-only tests of exact CUDA mapping ABI, multi-UUID contention and no retargeting."""
import ctypes
import importlib.util
import inspect
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
        self.one = dict(index=0, uuid="GPU-one", used_mib=100, utilization_percent=0)
        self.two = dict(index=1, uuid="GPU-two", used_mib=100, utilization_percent=0)
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
        self.assertEqual(held_while_waiting, [0])
        self.assertEqual(len(locks.held), 2)
        shared.release_all(handles, locks)
        self.assertEqual(locks.held, set())


class FunctionalAdmissionTests(unittest.TestCase):
    setUp = CudaExecutionTests.setUp
    probe = CudaExecutionTests.probe
    environment = {"GITHUB_EVENT_NAME": "workflow_dispatch", "GITHUB_WORKFLOW_REF":
        "SceneWorks/SceneWorks/.github/workflows/memory-catalog-campaign.yml@refs/heads/feature/test",
        "SCENEWORKS_GPU_ID": "1", "CUDA_VISIBLE_DEVICES": "1", "NVIDIA_VISIBLE_DEVICES": "1",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}

    def require_policy(self, environment=None, actor="load-progress-acceptance", **selection):
        shared.require_admission_policy(shared.FUNCTIONAL_POLICY, actor,
            self.environment if environment is None else environment,
            selection.get("physical_indices"), selection.get("cuda_ordinals", "0"), selection.get("probe_visible", "1"))

    def test_functional_policy_is_explicit_actor_manual_workflow_and_fixed_gpu1_only(self):
        self.require_policy()
        direct = dict(self.environment, GITHUB_WORKFLOW_REF=
            "SceneWorks/SceneWorks/.github/workflows/load-progress-acceptance.yml@refs/heads/main")
        self.require_policy(direct)
        for environment in [dict(self.environment, GITHUB_EVENT_NAME="push"),
                dict(self.environment, GITHUB_WORKFLOW_REF="other"),
                dict(self.environment, CUDA_DEVICE_ORDER="FASTEST_FIRST")]:
            with self.assertRaises(RuntimeError): self.require_policy(environment)
        for key in ("SCENEWORKS_GPU_ID", "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"):
            for value in ("0", "", "0,1"):
                with self.assertRaises(RuntimeError): self.require_policy(dict(self.environment, **{key: value}))
        with self.assertRaises(RuntimeError): self.require_policy(actor="another-actor")
        for selection in ({"physical_indices": "1"}, {"cuda_ordinals": "1"}, {"probe_visible": "0"}):
            with self.assertRaises(RuntimeError): self.require_policy(**selection)
        # Existing callers keep their strict policy and device semantics.
        shared.require_admission_policy(shared.STRICT_POLICY, "legacy-actor", {}, "0", "all", None)

    def test_functional_device_only_probe_and_exact_binding_refuse_gpu0_missing_or_wrong_uuid(self):
        devices = Mock(return_value=SimpleNamespace(stdout="0, GPU-one\n1, GPU-two\n"))
        identity = shared.DeviceIdentity("trusted-smi", devices)
        for rows, accepted in [([{"ordinal": 0, "uuid": "GPU-two"}], True),
                ([{"ordinal": 0, "uuid": "GPU-one"}], False),
                ([{"ordinal": 0, "uuid": "GPU-missing"}], False), ([], False)]:
            launch = Mock()
            try:
                binding = shared.resolve_binding(identity, self.out, time.monotonic()+30,
                    self.environment, probe_visible="1", run=self.probe(rows))
                shared.require_functional_binding(binding)
            except RuntimeError:
                self.assertFalse(accepted)
            else:
                self.assertTrue(accepted)
                launch()
            self.assertEqual(launch.call_count, int(accepted))
        for value in ("0, GPU-one\n1, \n", "0, GPU-one\n1, GPU-one\n", ""):
            devices.return_value.stdout = value
            with self.assertRaises(RuntimeError): identity.inventory(self.out, "invalid", time.monotonic()+30)
        self.assertTrue(all(call.args[0] == ["trusted-smi", "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits"] for call in devices.call_args_list))

    def test_foreign_compute_permits_functional_but_default_strict_rejects_and_no_baseline_is_sampled(self):
        vendor = shared.component("veda_cuda_guard.py", shared.GUARD_SHA)
        self.two.update(used_mib=65000, utilization_percent=100)
        self.guard.inventory.return_value = ([self.one, self.two], [["GPU-two", "57052", "foreign", "C"]])
        self.guard.idle.side_effect = vendor.idle
        self.guard.sample_baseline.side_effect = AssertionError("functional must not require benchmark baseline")
        class Locker:
            LK_NBLCK, LK_UNLCK = 2, 0
            def locking(self, *_): pass
        now = [0]
        with self.assertRaises(TimeoutError):
            shared.claim_uuids(["GPU-two"], self.guard, self.out, 2, self.out, Locker(),
                clock=lambda: now[0], sleep=lambda seconds: now.__setitem__(0, now[0]+seconds))
        self.guard.idle.reset_mock()
        selected, handles, baseline, _ = shared.claim_uuids(["GPU-two"], self.guard,
            self.out, 10, self.out, Locker(), required_indices={"GPU-two": 1},
            admission_policy=shared.FUNCTIONAL_POLICY, clock=lambda: 0)
        self.assertEqual(selected, [self.two]); self.assertEqual(baseline, {})
        self.guard.idle.assert_not_called(); self.guard.sample_baseline.assert_not_called()
        self.assertFalse((self.out / "SceneWorks/gpu-locks/GPU-one.lock").exists())
        shared.release_all(handles, Locker())

    def test_functional_uuid_contention_budget_and_physical_recheck_remain_fail_closed(self):
        class Locker:
            LK_NBLCK, LK_UNLCK = 2, 0
            def __init__(self, busy): self.busy = busy; self.held = set()
            def locking(self, fd, mode, _):
                if mode == self.LK_NBLCK:
                    if self.busy: self.busy -= 1; raise OSError("same UUID is leased")
                    self.held.add(fd)
                else: self.held.remove(fd)
        for busy, deadline, succeeds in [(2, 5, True), (20, 2, False)]:
            locker = Locker(busy); now = [0]
            def claim():
                return shared.claim_uuids(["GPU-two"], self.guard, self.out, deadline,
                    self.out, locker, required_indices={"GPU-two": 1},
                    admission_policy=shared.FUNCTIONAL_POLICY, clock=lambda: now[0],
                    sleep=lambda seconds: now.__setitem__(0, now[0]+seconds))
            if succeeds:
                _, handles, _, waited = claim(); self.assertEqual(waited, 2)
                self.assertEqual(len(locker.held), 1); shared.release_all(handles, locker)
            else:
                with self.assertRaises(TimeoutError): claim()
                self.assertEqual(now[0], deadline)
            self.assertEqual(locker.held, set())
        self.guard.inventory.return_value = ([dict(self.two, index=0)], [])
        with self.assertRaisesRegex(RuntimeError, "index/UUID binding changed"):
            shared.claim_uuids(["GPU-two"], self.guard, self.out, 10, self.out, Locker(0),
                required_indices={"GPU-two": 1}, admission_policy=shared.FUNCTIONAL_POLICY, clock=lambda: 0)

    def test_functional_completion_requires_success_no_timeout_empty_owned_tree_and_never_claims_idle(self):
        self.guard.verify_idle.side_effect = RuntimeError("foreign baseline changed")
        for result, accepted in [({"returncode": 0, "timeout": False, "owned_tree_empty": True}, True),
                ({"returncode": 3, "timeout": False, "owned_tree_empty": True}, False),
                ({"returncode": None, "timeout": True, "owned_tree_empty": True}, False),
                ({"returncode": 0, "timeout": False, "owned_tree_empty": False}, False)]:
            receipt = shared.resource_receipt("load-progress-acceptance", ["original-actor"], shared.FUNCTIONAL_POLICY)
            shared.complete_resource(receipt, result, [self.two], self.guard, self.out, {}, shared.FUNCTIONAL_POLICY)
            self.assertEqual(receipt["resource_completed"], accepted)
            for key in ("post_child_idle_verified", "isolated_device", "vram_measurement", "performance_measurement", "model_inference_proof"):
                self.assertIs(receipt[key], False)
            self.assertTrue(receipt["shared_device"])
        self.guard.verify_idle.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, "foreign baseline changed"):
            shared.complete_resource({}, {"returncode": 0, "timeout": False}, [self.two],
                self.guard, self.out, {"GPU-two": {}}, shared.STRICT_POLICY)

    def test_main_validates_physical_one_before_and_after_lock_and_reports_only_owned_completion(self):
        def binding(index=1, identifier="GPU-two"):
            return {"kind": "context-free-cuda-ordinal-uuid", "devices": [{"index": index, "uuid": identifier}],
                    "mapping": [{"cuda_ordinal": 0, "uuid": identifier, "nvml_index": index}], "key": [(0, identifier)]}
        valid = binding()
        cases = [(valid, valid, True), (binding(index=0), valid, False),
                 (binding(identifier=None), valid, False), (valid, binding(index=0), False),
                 (valid, binding(identifier="GPU-remapped"), False)]
        for attempt, (before, after, accepted) in enumerate(cases, 1):
            env = dict(self.environment, GITHUB_RUN_ID=str(attempt), GITHUB_RUN_ATTEMPT="1",
                       RUNNER_TEMP=str(self.out), PUBLIC=str(self.out))
            executor = Mock(return_value={"returncode": 0, "timeout": False, "owned_tree_empty": True})
            with patch.object(shared, "os", SimpleNamespace(name="nt", environ=env)), \
                 patch.object(shared.sys, "argv", ["wrapper", "--actor", "load-progress-acceptance",
                     "--admission-policy", "functional-coexistence", "--budget-minutes", "1",
                     "--cuda-ordinals", "0", "--probe-visible", "1", "--", "original-actor"]), \
                 patch.dict(shared.sys.modules, {"msvcrt": SimpleNamespace()}), \
                 patch.object(shared, "prepared_components", return_value=(self.guard, SimpleNamespace(WindowsJob=object))), \
                 patch.object(shared, "resolve_binding", side_effect=[before, after]), \
                 patch.object(shared, "claim_uuids", return_value=([self.two], [], {}, 0)), \
                 patch.object(shared, "execute_owned", executor):
                self.assertEqual(shared.main(), 0 if accepted else 1)
            receipt = json.loads((self.out / f"sceneworks-cuda-execution-{attempt}-1-load-progress-acceptance/resource-receipt.json").read_text())
            self.assertEqual(receipt["resource_completed"], accepted)
            self.assertEqual(executor.call_count, int(accepted))
            self.assertIs(receipt["post_child_idle_verified"], False)
            self.assertIs(receipt["performance_measurement"], False)
            if accepted:
                self.assertEqual(receipt["binding_rechecked"]["devices"][0]["index"], 1)
        self.guard.verify_idle.assert_not_called()

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
        rows = [dict(index=0, uuid="GPU-a"), dict(index=1, uuid="GPU-b")]
        guard.inventory.return_value = (rows, [])
        guard.idle.return_value = True
        guard.sample_baseline.side_effect = lambda device, out, phase: (device, {})
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


if __name__ == "__main__":
    unittest.main()
