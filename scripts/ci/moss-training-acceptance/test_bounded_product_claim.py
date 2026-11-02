"""Mock-only bounded claim integration against an authenticated inference checkout.

Run: python test_bounded_product_claim.py --inference-root /path/to/inference
No driver command, CUDA metadata subprocess, physical lease, or model is executed.
"""
import argparse
import ctypes
from contextlib import ExitStack, redirect_stdout
from datetime import datetime, timedelta, timezone
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('bounded_claim', HERE / 'bounded_product_claim.py')
claim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(claim)


class BoundedClaimTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pin = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ENGINE, text=True).strip()
        cls.modules = claim.source_modules(ENGINE, cls.pin)

    def invoke(self, *, bad_phase=None, response=None, probe_uuid=None, remaining=60,
               native=False, wait_results=(0, 0, 0), release_response=None,
               expire_release=False, unlock_failure=False, close_handle=True):
        guard, resources, primitives, seals = self.modules
        inventory = guard.inventory
        resolve_binding = primitives.resolve_binding
        verify_idle = guard.verify_idle
        state = {'phase': None, 'expired': False}
        trace = []
        good = f'1, {claim.UUID}, MOCK ONLY, 596.36, 97887, 5247, 92640, [N/A]'
        with tempfile.TemporaryDirectory(prefix='bounded-claim-metadata-only-') as temporary:
            root = Path(temporary)
            public = root / 'public'
            public.mkdir()
            output = root / 'evidence'
            environment = {
                'PUBLIC': str(public), 'MOSS_ACCEPTANCE_PHASE': 'bounded-product',
                'MODEL_INFERENCE_CASE_FQN': claim.SELECTOR, 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
                'CUDA_VISIBLE_DEVICES': '1', 'NVIDIA_VISIBLE_DEVICES': '1',
                'MOSS_ACCEPTANCE_RESOLVED_CAP_BYTES': str(claim.CAP),
                'MOSS_ACCEPTANCE_WORKFLOW_SOURCE': 'a' * 40, 'GITHUB_SHA': 'a' * 40,
                'GITHUB_EVENT_NAME': 'workflow_dispatch', 'GITHUB_RUN_ID': 'MOCK',
                'GITHUB_RUN_ATTEMPT': '1', 'MOSS_ACCEPTANCE_INFERENCE_PIN': self.pin,
            }
            args = SimpleNamespace(selector=claim.SELECTOR, output=str(output),
                inference_root=str(ENGINE),
                deadline=(datetime.fromtimestamp(1_700_000_000, timezone.utc) +
                          timedelta(seconds=remaining)).isoformat())
            def locking(handle, mode, count):
                trace.append(('lock', mode))
                if mode == 2 and unlock_failure:
                    raise OSError('MOCK explicit lease unlock failed')
            locker = SimpleNamespace(LK_NBLCK=1, LK_UNLCK=2, locking=Mock(side_effect=locking))
            statuses = iter(wait_results)
            def wait(handle, timeout):
                status = next(statuses, 0)
                trace.append(('wait', timeout, status))
                return status
            kernel = SimpleNamespace(OpenProcess=Mock(return_value=4242),
                WaitForSingleObject=Mock(side_effect=wait),
                CloseHandle=Mock(return_value=close_handle))
            native_root = {'scope': 'native-api-worker-root', 'pid': 4242,
                'executable': str(root / 'MOCK-NOT-EXECUTED.exe'), 'binarySha256': 'f' * 64}
            input_lines = (json.dumps({'root': native_root}) + '\n' if native else '') + '{"release":true}\n'
            packet = {'context_created': False,
                'abi_source': {'checksum': '1cea5f10a99e025c1b44ae2354c2d8326b25ddbd0baf76bde8e55cfd4018a2cc'},
                'devices': [{'ordinal': 0, 'uuid': probe_uuid or claim.UUID}]}
            probe = Mock(return_value=SimpleNamespace(stdout=json.dumps(packet)))

            def selected_snapshot(out=None, label='snapshot', deadline=None, *, selected_uuid=None):
                self.assertEqual(selected_uuid, claim.UUID, 'every call must use source-owned UUID')
                self.assertIsNotNone(deadline)
                state['phase'] = label
                return inventory(out, label, deadline, selected_uuid=selected_uuid)

            def driver(arguments, deadline):
                self.assertEqual(arguments, ['nvidia-smi',
                    '--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu',
                    '--format=csv,noheader,nounits', '--id=' + claim.UUID])
                self.assertGreater(deadline, time.monotonic())
                if state['phase'] == bad_phase:
                    if isinstance(response, BaseException):
                        raise response
                    return response
                if state['phase'].startswith('bounded-release') and release_response is not None:
                    return release_response
                return good

            def bind(reader, out, deadline, env, **kwargs):
                self.assertEqual(kwargs, {'cuda_ordinals': '0'})
                self.assertEqual(env, environment)
                return resolve_binding(reader, out, deadline, env, run=probe, **kwargs)

            def verify_release(*args, **kwargs):
                trace.append(('release-check',))
                if expire_release:
                    state['expired'] = True
                return verify_idle(*args, **kwargs)

            error = None
            with ExitStack() as stack:
                stack.enter_context(patch.object(claim, 'source_modules', return_value=self.modules))
                # Mock the consumer's OS seam without changing pathlib or other modules' OS.
                stack.enter_context(patch.object(claim, 'os', SimpleNamespace(
                    name='nt', environ=environment, fstat=os.fstat)))
                stack.enter_context(patch.dict(sys.modules, {'msvcrt': locker}))
                stack.enter_context(patch.object(ctypes, 'WinDLL', return_value=kernel, create=True))
                stack.enter_context(patch.object(claim.sys, 'stdin', io.StringIO(input_lines)))
                stack.enter_context(patch.object(claim.threading, 'Thread',
                    side_effect=lambda target, daemon: SimpleNamespace(start=target)))
                stack.enter_context(patch.object(guard.time, 'sleep'))
                stack.enter_context(patch.object(guard.time, 'time', return_value=1_700_000_000))
                stack.enter_context(patch.object(guard.time, 'monotonic', side_effect=lambda:
                    100.0 + remaining + 1 if state['expired'] else 100.0))
                commands = stack.enter_context(patch.object(guard, 'command', side_effect=driver))
                snapshots = stack.enter_context(patch.object(guard, 'inventory', side_effect=selected_snapshot))
                baseline = stack.enter_context(patch.object(guard, 'sample_baseline', wraps=guard.sample_baseline))
                release = stack.enter_context(patch.object(guard, 'verify_idle', side_effect=verify_release))
                binding = stack.enter_context(patch.object(primitives, 'resolve_binding', side_effect=bind))
                stack.enter_context(redirect_stdout(io.StringIO()))
                try:
                    claim.main(args)
                except (ValueError, RuntimeError, TimeoutError, subprocess.TimeoutExpired, OSError) as caught:
                    error = caught
            record = json.loads((output / 'physical-claim.json').read_text(encoding='utf-8'))
            samples = {p.name: json.loads(p.read_text(encoding='utf-8')) for p in output.glob('*-samples.json')}
            self.assertEqual(record['leaseReleased'], not unlock_failure and close_handle)
            self.assertEqual([call.args[1] for call in locker.locking.call_args_list],
                [1, 2] if close_handle else [1])
            return SimpleNamespace(error=error, record=record, samples=samples, snapshots=snapshots,
                commands=commands, baseline=baseline, release=release, binding=binding, probe=probe,
                kernel=kernel, trace=trace)

    def test_all_claim_phases_use_selected_primary_counters_without_census(self):
        observed = self.invoke(remaining=1)
        self.assertIsNone(observed.error)
        self.assertTrue(observed.record['postExecutionVerified'])
        self.assertFalse(observed.record['nativeProcessStarted'])
        diagnostic = observed.record['admissionDiagnostic']
        self.assertEqual(diagnostic['process_inventory_status'], 'unknown/not-collected-diagnostic-only')
        self.assertIsNone(diagnostic['selected_process_count'])
        self.assertIsNone(diagnostic['selected_process_type_counts'])
        labels = [call.args[1] for call in observed.snapshots.call_args_list]
        self.assertEqual(labels, ['bounded-claimed-device'] +
            [f'bounded-admission-sample-{i}' for i in range(1, 6)] + ['binding'] +
            [f'bounded-release-sample-{i}' for i in range(1, 6)])
        self.assertEqual(observed.commands.call_count, 12)
        original_deadline = observed.snapshots.call_args_list[0].args[2]
        self.assertTrue(all(call.args[2] <= original_deadline for call in observed.snapshots.call_args_list))
        callback = observed.baseline.call_args_list[0].kwargs['inventory_reader']
        self.assertIs(observed.binding.call_args.args[0].inventory, callback)
        self.assertIs(observed.release.call_args.kwargs['inventory_reader'], callback)
        for call in observed.baseline.call_args_list:
            self.assertIs(call.kwargs['inventory_reader'], callback)
            self.assertFalse(call.kwargs['require_stable_contexts'])
            self.assertTrue(call.kwargs['permit_bounded_growth'])
        observed.probe.assert_called_once()
        self.assertEqual(observed.record['ordinalBinding']['mapping'],
            [{'cuda_ordinal': 0, 'uuid': claim.UUID, 'nvml_index': 1}])
        self.assertEqual(observed.record['sourceSeals'], self.modules[3])
        for samples in observed.samples.values():
            self.assertEqual(len(samples), 5)
            self.assertTrue(all(row['devices'][0]['process_inventory_status'] ==
                'unknown/not-collected-diagnostic-only' for row in samples))

    def test_primary_timeout_refuses_initial_baseline_binding_and_release(self):
        for phase in ['bounded-claimed-device', 'bounded-admission-sample-3', 'binding', 'bounded-release-sample-3']:
            with self.subTest(phase=phase):
                observed = self.invoke(bad_phase=phase, response=TimeoutError('MOCK primary timeout'))
                self.assertIsInstance(observed.error, TimeoutError)
                self.assertFalse(observed.record['postExecutionVerified'])
                self.assertEqual(observed.record['state'], 'failed')

    def test_bad_identity_or_memory_refuses_every_phase_without_fallback(self):
        good = f'1, {claim.UUID}, MOCK ONLY, 596.36, 97887, 5247, 92640, 73'
        for phase in ['bounded-claimed-device', 'bounded-admission-sample-3', 'binding', 'bounded-release-sample-3']:
            for response in [good.replace(claim.UUID, 'GPU-00000000-0000-0000-0000-000000000000'),
                             good.replace('1, GPU-', '0, GPU-'), '', good + '\n' + good,
                             good.replace('92640', 'nan'), good.replace('97887', 'inf')]:
                with self.subTest(phase=phase, response=response):
                    observed = self.invoke(bad_phase=phase, response=response)
                    self.assertIsNotNone(observed.error)
                    self.assertFalse(observed.record['postExecutionVerified'])
        # New-work headroom is admission-only, not release or the metadata-only binding probe.
        for phase in ['bounded-claimed-device', 'bounded-admission-sample-3']:
            observed = self.invoke(bad_phase=phase, response=good.replace('5247, 92640', '20000, 77887'))
            self.assertIsNotNone(observed.error)
            self.assertFalse(observed.record['postExecutionVerified'])

    def test_original_low_free_busy_release_passes_only_after_owned_native_exit(self):
        # Original run 41195927630/1: admission free 95253 MiB; release free 72908,
        # util 100. These are mock scalar counters, never actual queries/model work.
        released = f'1, {claim.UUID}, MOCK ONLY, 596.36, 97887, 24531, 72908, 100'
        observed = self.invoke(native=True, release_response=released)
        self.assertIsNone(observed.error)
        self.assertTrue(observed.record['nativeProcessExited'])
        self.assertTrue(observed.record['postExecutionVerified'])
        self.assertEqual(observed.record['releasePolicy'], 'owned-native-exit-and-selected-primary-metadata')
        self.assertEqual(observed.record['admissionBaseline']['reserved_baseline_mib'], 5247,
            'release observations must not rewrite the admission baseline')
        self.assertLess(observed.trace.index(('wait', 0, 0)), observed.trace.index(('release-check',)))
        self.assertLess(observed.trace.index(('release-check',)), observed.trace.index(('lock', 2)))
        samples = observed.samples['bounded-release-samples.json']
        self.assertEqual(len(samples), 5)
        self.assertTrue(all(row['devices'][0]['free_mib'] == 72908 and
            row['devices'][0]['utilization_percent'] == 100 for row in samples))
        guard, resources, _, _ = self.modules
        policy = SimpleNamespace(child_gpu_cap_bytes=claim.CAP, gpu_free_reserve_bytes=claim.RESERVE)
        self.assertFalse(resources.proof_cuda_idle(samples[0]['devices'][0], [], policy,
            activity_profile=claim.PROFILE), 'the original admission predicate must still refuse new work')
        original_admission = {**samples[0]['devices'][0], 'used_mib': 2186,
            'free_mib': 95253, 'utilization_percent': 0}
        self.assertEqual(original_admission['total_mib'] - original_admission['used_mib'] -
            original_admission['free_mib'], 448)
        self.assertTrue(claim.selected_primary_metadata_valid(original_admission),
            'driver-reserved bytes must not require used+free==total')
        self.assertTrue(resources.proof_cuda_idle(original_admission, [], policy,
            activity_profile=claim.PROFILE))
        for rounding in (1, 2):
            self.assertTrue(claim.selected_primary_metadata_valid({**samples[0]['devices'][0],
                'free_mib': 97887 - 24531 + rounding}),
                'preserve the established +2 MiB driver-counter rounding tolerance')

    def test_pending_native_pid_cannot_enter_release_sampling_or_succeed(self):
        observed = self.invoke(native=True, wait_results=(258, 258, 0, 0))
        self.assertIn('native process still alive', str(observed.error))
        self.assertEqual(observed.record['state'], 'failed')
        self.assertFalse(observed.record['postExecutionVerified'])
        observed.release.assert_not_called()
        self.assertTrue(observed.record['nativeProcessExited'], 'cleanup waits for actual owned exit')
        self.assertLess(observed.trace.index(('wait', 0, 0)), observed.trace.index(('lock', 2)))

    def test_release_rejects_invalid_primary_metadata_even_after_owned_exit(self):
        good = f'1, {claim.UUID}, MOCK ONLY, 596.36, 97887, 24531, 72908, 100'
        for invalid in [good.replace(claim.UUID, 'GPU-00000000-0000-0000-0000-000000000000'),
                        good.replace('1, GPU-', '0, GPU-'), '', good + '\n' + good,
                        good.replace('72908', 'nan'), good.replace('97887', 'inf'),
                        good.replace('24531', '-1'), good.replace('72908', '97888'),
                        good.replace('72908', '73359'),  # Exceeds the +2 MiB rounding allowance.
                        good.replace(', 100', ', 101'), good.replace(', 100', ', -1')]:
            with self.subTest(invalid=invalid):
                observed = self.invoke(native=True, release_response=invalid)
                self.assertIsNotNone(observed.error)
                self.assertTrue(observed.record['nativeProcessExited'])
                self.assertFalse(observed.record['postExecutionVerified'])

    def test_release_keeps_original_deadline_and_lease_failures_fatal(self):
        observed = self.invoke(native=True, bad_phase='bounded-release-sample-3',
            response=TimeoutError('MOCK release primary timeout'))
        self.assertIsInstance(observed.error, TimeoutError)
        self.assertTrue(observed.record['nativeProcessExited'])
        self.assertFalse(observed.record['postExecutionVerified'])
        observed = self.invoke(native=True, expire_release=True)
        self.assertIn('deadline exhausted', str(observed.error))
        self.assertFalse(observed.record['postExecutionVerified'])
        self.assertTrue(observed.record['nativeProcessExited'])
        observed = self.invoke(native=True, unlock_failure=True)
        self.assertIsInstance(observed.error, OSError)
        self.assertEqual(observed.record['state'], 'failed')
        self.assertFalse(observed.record['leaseReleased'])
        observed = self.invoke(native=True, close_handle=False)
        self.assertIn('handle close failed', str(observed.error))
        self.assertEqual(observed.record['state'], 'failed')
        self.assertFalse(observed.record['leaseReleased'])

    def test_actual_ordinal_metadata_still_must_match_the_selected_uuid(self):
        observed = self.invoke(probe_uuid='GPU-00000000-0000-0000-0000-000000000000')
        self.assertIsInstance(observed.error, RuntimeError)
        self.assertFalse(observed.record['postExecutionVerified'])
        observed.probe.assert_called_once()
        observed.release.assert_not_called()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inference-root', required=True, type=Path)
    arguments, rest = parser.parse_known_args()
    ENGINE = arguments.inference_root.resolve()
    unittest.main(argv=[sys.argv[0], *rest])
