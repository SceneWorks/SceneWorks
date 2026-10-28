"""CPU-only tests of exact CUDA mapping ABI, multi-UUID contention and no retargeting."""
import ctypes
import importlib.util
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest
from unittest.mock import Mock

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
