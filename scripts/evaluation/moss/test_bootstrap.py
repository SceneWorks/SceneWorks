"""CPU-only contracts for acquisition integrity, transport inputs and device lock.

These tests never import torch, contact a host, acquire real weights or render.
"""
import contextlib
import datetime as dt
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('bootstrap', Path(__file__).with_name('bootstrap.py'))
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


class BootstrapContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.deadline = (b.utc() + dt.timedelta(minutes=30)).isoformat()

    def tearDown(self):
        self.temp.cleanup()

    def request(self, **extra):
        return json.dumps(dict(campaign='sc-26322-contract', deadline_utc=self.deadline, **extra))

    def test_prepare_exact_fields_and_no_path_injection(self):
        value, root = b.control('prepare', self.request(), self.root)
        self.assertEqual(root, self.root / 'sc-26322-contract')
        for campaign in ('../escape', 'sc-26322-x/../../escape', 'sc-26322-$(whoami)', 'sc-26322-a\\b'):
            with self.assertRaises(ValueError):
                b.control('prepare', json.dumps(dict(campaign=campaign, deadline_utc=self.deadline)), self.root)
        with self.assertRaises(ValueError):
            b.control('prepare', self.request(command='calc.exe'), self.root)

    def test_expired_and_unbounded_deadlines_rejected(self):
        for deadline in (b.utc() - dt.timedelta(seconds=1), b.utc() + dt.timedelta(hours=7)):
            with self.assertRaises(ValueError):
                b.control('prepare', json.dumps(dict(campaign='sc-26322-a', deadline_utc=deadline.isoformat())), self.root)

    def test_calibration_needs_reviewed_protocol_and_complete_lease(self):
        lease = dict(id='lease-1', owner='chat', host='host', gpu_uuid=b.RUNNER_GPU_UUID,
                     expires_utc=self.deadline, external_owners_reconciled=True, desktop_baseline=[],
                     baseline_max_memory_mib=2048, baseline_max_utilization_percent=2)
        request = self.request(protocol_sha256='a' * 64, lease=lease)
        b.control('calibration', request, self.root)
        for key, invalid in [('protocol_sha256', 'main'), ('lease', {**lease, 'external_owners_reconciled': False}),
                             ('lease', {**lease, 'expires_utc': b.utc().isoformat()})]:
            value = json.loads(request)
            value[key] = invalid
            with self.assertRaises(ValueError):
                b.control('calibration', json.dumps(value), self.root)

    def test_freeze_requires_bounded_complete_ownership(self):
        ownership = dict(responsible_chat='chat', lease='lease', job='job', timeout_seconds=1800,
                         resume='exact command', next_action='review calibration')
        b.control('freeze', self.request(gpu_uuid=b.RUNNER_GPU_UUID, ownership=ownership), self.root)
        with self.assertRaises(ValueError):
            b.control('freeze', self.request(gpu_uuid='0', ownership=ownership), self.root)
        with self.assertRaises(ValueError):
            b.control('freeze', self.request(gpu_uuid=b.RUNNER_GPU_UUID, ownership={**ownership, 'timeout_seconds': 1801}), self.root)

    def test_upstream_path_guards_windows_aliases_and_traversal(self):
        for name in ('../weights', '/weights', 'a/../b', 'a//b', 'a\\b', 'x:stream', 'CON.json', 'x./file', './a'):
            with self.assertRaises(ValueError, msg=name):
                b.safe_name(name)
        self.assertEqual(str(b.safe_name('transformer/model-00001.safetensors')), 'transformer/model-00001.safetensors')

    def test_artifact_lfs_digest_rejects_same_size_tamper(self):
        path = self.root / 'weights'
        path.write_bytes(b'abc')
        item = dict(size=3, lfs={'sha256': hashlib.sha256(b'abc').hexdigest()})
        receipt = b.verify_blob(path, item)
        self.assertEqual(receipt['sha256'], item['lfs']['sha256'])
        path.write_bytes(b'xyz')
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            b.verify_blob(path, item)

    def test_small_upstream_files_verify_git_object_hash_not_raw_sha1(self):
        path = self.root / 'config.json'
        path.write_bytes(b'{}')
        item = dict(size=2, blobId=hashlib.sha1(b'blob 2\0{}').hexdigest())
        b.verify_blob(path, item)
        item['blobId'] = hashlib.sha1(b'{}').hexdigest()
        with self.assertRaisesRegex(ValueError, 'Git blob'):
            b.verify_blob(path, item)

    def test_stage_minimal_components_and_verified_regular_files(self):
        payload = b'CPU integrity fixture, not tensor data'
        metadata = dict(sha=b.REPOS['moss_base'][1], siblings=[dict(rfilename='transformer/a.safetensors',
            size=len(payload), lfs={'sha256': hashlib.sha256(payload).hexdigest()})])
        with patch.object(b, 'fetch_json', return_value=metadata), patch.object(b.urllib.request, 'urlopen', return_value=io.BytesIO(payload)):
            receipt = b.stage(self.root, 'moss_base')
        path = Path(receipt['files'][0]['path'])
        self.assertEqual(path.read_bytes(), payload)
        self.assertFalse(path.is_symlink())
        path.write_bytes(b'x' * len(payload))
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            b.stage(self.root, 'moss_base')

    def test_no_duplicate_encoder_or_combined_dmd_download(self):
        self.assertFalse(b.selected('moss_base', 'text_encoder/model.safetensors'))
        self.assertTrue(b.selected('moss_sft', 'text_encoder/model.safetensors'))
        self.assertFalse(b.selected('moss_dmd', 'generator.safetensors'))
        for name in ('transformer/model.safetensors', 'lora.safetensors', 'time_embedding.safetensors', 'dataspace.safetensors'):
            self.assertTrue(b.selected('moss_dmd', name))
        self.assertFalse(b.selected('qwen_image_2_1', 'q8/transformer/model.safetensors'))
        self.assertTrue(b.selected('qwen_image_2_1', 'q4/transformer/model.safetensors'))

    def test_device_lock_is_exclusive_never_steals_stale_file(self):
        uuid = 'GPU-12345678-12345678'
        with b.device_lock(self.root, uuid, {'lease_id': 'owned'}) as lock:
            self.assertEqual(b.read(lock)['lease_id'], 'owned')
            with self.assertRaises(FileExistsError):
                with b.device_lock(self.root, uuid, {'lease_id': 'other'}):
                    self.fail('second lease entered')
        self.assertFalse(lock.exists())
        lock.write_text('stale')
        with self.assertRaises(FileExistsError):
            with b.device_lock(self.root, uuid, {'lease_id': 'new'}):
                self.fail('stale lock was stolen')
        self.assertEqual(lock.read_text(), 'stale')

    def test_noncompute_gpu_cannot_be_mistaken_for_selected_device(self):
        with patch.object(b, 'run', return_value='GPU-selected, 123\nGPU-other, 456\n'):
            self.assertEqual(b.compute_pids('GPU-selected'), {123})
        with patch.object(b, 'run', return_value='GPU-selected, N/A\n'):
            with self.assertRaises(ValueError):
                b.compute_pids('GPU-selected')

    def test_reviewed_desktop_guards_reject_foreign_owner_and_ceiling_breach(self):
        lease = dict(gpu_uuid=b.RUNNER_GPU_UUID, desktop_baseline=[],
                     baseline_max_memory_mib=1000, baseline_max_utilization_percent=2)
        gpu = dict(index='1', uuid=b.RUNNER_GPU_UUID, used_mib='781', utilization_percent='0')
        with patch.object(b, 'desktop_identities', return_value=set()), patch.object(b, 'gpu_rows', return_value=[gpu]):
            with patch.object(b, 'compute_pids', return_value={999}):
                with self.assertRaisesRegex(ValueError, 'owners differ'):
                    b.idle_device(lease)
            with patch.object(b, 'compute_pids', return_value=set()):
                self.assertIn('limitation', b.idle_device(lease))
                gpu['used_mib'] = '1001'
                with self.assertRaisesRegex(ValueError, 'ceiling'):
                    b.idle_device(lease)
                gpu['used_mib'] = '781'
                gpu['index'] = '0'
                with self.assertRaisesRegex(ValueError, 'GPU1'):
                    b.idle_device(lease)

    def test_gpu0_lease_is_rejected_before_runtime_actions(self):
        lease = dict(id='lease-1', owner='chat', host='host', gpu_uuid='GPU-b1a31911-c7b4-2901-3d8b-9a62e228bfc0',
                     expires_utc=self.deadline, external_owners_reconciled=True, desktop_baseline=[],
                     baseline_max_memory_mib=0, baseline_max_utilization_percent=0)
        with self.assertRaisesRegex(ValueError, 'GPU1'):
            b.control('calibration', self.request(protocol_sha256='a' * 64, lease=lease), self.root)

    def test_foreign_uuid_and_malformed_controls_cannot_create_lock(self):
        with self.assertRaises(ValueError):
            with b.device_lock(self.root, '../../elsewhere', {}):
                pass
        self.assertFalse((self.root / 'device-locks').exists())


if __name__ == '__main__':
    unittest.main()
