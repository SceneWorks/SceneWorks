"""CPU-only contracts for acquisition integrity, transport inputs and device lock.

These tests never import torch, contact a host, acquire real weights or render.
"""
import contextlib
import datetime as dt
import hashlib
import importlib.util
import io
import json
import os
import socket
import sys
import types
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('bootstrap', Path(__file__).with_name('bootstrap.py'))
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
import ownership as o


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
        with patch.object(o, 'run', return_value='GPU-selected, 123\nGPU-other, 456\n'):
            self.assertEqual(b.compute_pids('GPU-selected'), {123})
        with patch.object(o, 'run', return_value='GPU-selected, N/A\n'):
            with self.assertRaises(ValueError):
                b.compute_pids('GPU-selected')

    def test_reviewed_desktop_guards_reject_foreign_owner_and_ceiling_breach(self):
        lease = dict(gpu_uuid=b.RUNNER_GPU_UUID, desktop_baseline=[],
                     baseline_max_memory_mib=1000, baseline_max_utilization_percent=2)
        gpu = dict(index='1', uuid=b.RUNNER_GPU_UUID, used_mib='781', utilization_percent='0')
        with patch.object(o, 'desktop_identities', return_value=set()), patch.object(o, 'gpu_rows', return_value=[gpu]):
            with patch.object(o, 'compute_pids', return_value={999}):
                with self.assertRaisesRegex(ValueError, 'owners differ'):
                    b.idle_device(lease)
            with patch.object(o, 'compute_pids', return_value=set()):
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

    def test_primary_dependency_conflict_is_fixed_in_exact_reviewed_closure(self):
        resolution = b.dependency_resolution()
        self.assertEqual(len(resolution['packages']), 43)
        versions = {p['name'].lower().replace('_','-'):p['version'] for p in resolution['packages']}
        self.assertEqual(versions['safetensors'], '0.8.0')
        self.assertEqual(versions['diffusers'], '0.38.0')
        self.assertEqual(resolution['binary_wheel_requests'], 0)
        old = tuple('safetensors==0.6.2' if p.startswith('safetensors==') else p for p in b.PACKAGES)
        with patch.object(b, 'PACKAGES', old):
            with self.assertRaisesRegex(ValueError, 'closure drift'):
                b.dependency_resolution()

    def test_scored_control_needs_accepted_digests_and_fixed_authorization(self):
        lease = dict(id='l',owner='chat',host='host',gpu_uuid=b.RUNNER_GPU_UUID,expires_utc=self.deadline,
            external_owners_reconciled=True,desktop_baseline=[],baseline_max_memory_mib=1000,baseline_max_utilization_percent=2)
        raw=json.loads(self.request(protocol_sha256='a'*64,lease=lease))
        with self.assertRaises(ValueError):b.control('scored',json.dumps(raw),self.root)
        raw.update(calibration_receipts_sha256='b'*64,calibration_estimate_sha256='c'*64,
            authorization=dict(id='accepted',coordinator='chat',accepted_calibration=True,max_scored_outputs=288,acknowledge_shape_uncertainty=True))
        b.control('scored',json.dumps(raw),self.root)
        for key,value in [('max_scored_outputs',289),('accepted_calibration',False),('acknowledge_shape_uncertainty',False),('coordinator','other')]:
            broken=json.loads(json.dumps(raw));broken['authorization'][key]=value
            with self.assertRaises(ValueError):b.control('scored',json.dumps(broken),self.root)

    def test_calibration_estimate_uses_actual_canvas_and_does_not_invent_full_cost(self):
        manifest=dict(corpus=dict(cases=[dict(width=1024,height=1024),dict(width=1536,height=1024)],seeds=[1,2]))
        results={key:dict(status='ok',cell=dict(config_id=key,width=1024,height=1024),metrics=dict(generation_seconds=10,load_seconds=2)) for key in b.REPOS}
        estimate=b.calibration_estimate(manifest,results)
        self.assertIsNone(estimate['whole_campaign_estimate_seconds'])
        self.assertEqual(estimate['calibration_outputs'],6)
        self.assertEqual(estimate['conditions']['moss_base']['same_canvas_scored_outputs'],2)
        self.assertEqual(estimate['conditions']['moss_base']['same_canvas_generation_estimate_seconds'],20)
        results['moss_base']['cell']['width']=512
        with self.assertRaisesRegex(ValueError,'actual1024'):b.calibration_estimate(manifest,results)

    def test_cleanup_handles_exited_parent_and_never_kills_reused_or_unrelated_pid(self):
        events=[]
        class Missing(Exception):pass
        class Child:
            def __init__(self,pid):self.pid=pid
            def create_time(self):return 10 if self.pid==222 else 11
            def terminate(self):events.append(('terminate',self.pid))
            def kill(self):events.append(('kill',self.pid))
            def is_running(self):return True
        def process(pid):
            if pid==111:raise Missing()
            return Child(pid)
        parent=types.SimpleNamespace(pid=111,poll=lambda:0)
        fake=types.SimpleNamespace(Process=process,NoSuchProcess=Missing)
        with patch.dict(sys.modules,psutil=fake):
            b.kill_owned(parent,{222:10,333:10})
        self.assertEqual(events,[('terminate',222),('kill',222)])

    def test_desktop_pid_reuse_or_name_change_never_inherits_allowance(self):
        process=types.SimpleNamespace(name=lambda:'chrome.exe',create_time=lambda:100)
        lease=dict(desktop_baseline=[dict(pid=55,name='chrome.exe',created_epoch=100)])
        with patch.dict(sys.modules,psutil=types.SimpleNamespace(Process=lambda pid:process)):
            self.assertEqual(o.desktop_identities(lease),{55})
            lease['desktop_baseline'][0]['created_epoch']=99
            with self.assertRaisesRegex(ValueError,'identity changed'):o.desktop_identities(lease)
            lease['desktop_baseline'][0]['created_epoch']=100;lease['desktop_baseline'][0]['name']='python.exe'
            with self.assertRaisesRegex(ValueError,'identity changed'):o.desktop_identities(lease)

    def test_accepted_scored_authorization_is_immutable_and_estimate_bound(self):
        b.write(self.root/'calibration-estimate.json',dict(measured=True))
        value=dict(protocol_sha256='a'*64,calibration_receipts_sha256='b'*64,
            calibration_estimate_sha256=b.digest(self.root/'calibration-estimate.json'),
            authorization=dict(id='accepted',coordinator='chat',accepted_calibration=True,max_scored_outputs=288,acknowledge_shape_uncertainty=True))
        with patch.object(b,'calibration_bundle',return_value='b'*64):
            first=b.accepted_scored_authorization(self.root,value)
            self.assertEqual(first,b.accepted_scored_authorization(self.root,value))
            changed=json.loads(json.dumps(value));changed['authorization']['id']='replacement'
            with self.assertRaisesRegex(ValueError,'authorization changed'):b.accepted_scored_authorization(self.root,changed)
            (self.root/'calibration-estimate.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'estimate digest'):b.accepted_scored_authorization(self.root,value)

    def test_foreign_uuid_and_malformed_controls_cannot_create_lock(self):
        with self.assertRaises(ValueError):
            with b.device_lock(self.root, '../../elsewhere', {}):
                pass
        self.assertFalse((self.root / 'device-locks').exists())


class SharedOwnershipContracts(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name).resolve()/'authority'/'campaign'
        self.root.mkdir(parents=True)
        staged={};configs=[]
        for key,(repo,revision) in o.REPOS.items():
            snapshot=self.root/'hub'/('models--'+repo.replace('/','--'))/'snapshots'/revision
            snapshot.mkdir(parents=True);file=snapshot/'tiny.safetensors';file.write_bytes(b'CPU file integrity fixture')
            refs=snapshot.parent.parent/'refs';refs.mkdir();(refs/'main').write_text(revision)
            meta=self.root/'metadata'/(key+'.json');meta.parent.mkdir(exist_ok=True);meta.write_text('{}')
            item=dict(repo=repo,revision=revision,snapshot=str(snapshot),metadata=str(meta),metadata_sha256=o.digest(meta),files=[dict(path=str(file),sha256=o.digest(file),bytes=file.stat().st_size)])
            staged[key]=item
            configs.append(dict(id=key,repo=repo,revision=revision,snapshot=str(snapshot),files=item['files'],settings=dict(hub_cache=str(self.root/'hub'))))
        b.write(self.root/'prepared.json',dict(staged=staged))
        self.manifest=dict(hardware=dict(host=socket.gethostname(),gpu_uuid=o.RUNNER_GPU_UUID),ownership=dict(lease='lease',responsible_chat='chat'),configs=configs,
            preparation=dict(path=str(self.root/'prepared.json'),sha256=o.digest(self.root/'prepared.json')))
        self.lease=dict(lease_id='lease',responsible_chat='chat',host=socket.gethostname(),gpu_uuid=o.RUNNER_GPU_UUID,
            protocol_sha256='a'*64,owner_pid=987,phase='calibration',deadline_epoch=5000,lease_expires_epoch=5000,
            prepared_root=str(self.root),prepared_sha=o.digest(self.root/'prepared.json'),external_owners_reconciled=True,desktop_baseline=[],baseline_max_memory_mib=1000,baseline_max_utilization_percent=2)
        self.path=self.root.parent/'device-locks'/(o.RUNNER_GPU_UUID+'.lock');b.write(self.path,self.lease)
        self.env=dict(MOSS_DEVICE_LOCK=str(self.path),MOSS_LEASE_ID='lease',MOSS_PROTOCOL_SHA256='a'*64)
        class Process:
            def __init__(self,pid=None):self.pid=pid or 123
            def create_time(self):return 1000
            def children(self,recursive=True):return[]
            def parents(self):return[types.SimpleNamespace(pid=987)]
        self.stack=contextlib.ExitStack()
        self.stack.enter_context(patch.dict(sys.modules,psutil=types.SimpleNamespace(Process=Process)))
        self.gpu=dict(index='1',uuid=o.RUNNER_GPU_UUID,used_mib='781',utilization_percent='0')
        self.stack.enter_context(patch.object(o,'gpu_rows',return_value=[self.gpu]))
        self.pids=self.stack.enter_context(patch.object(o,'compute_pids',return_value=set()))

    def tearDown(self):self.stack.close();self.temp.cleanup()

    def guard(self,**kwargs):return o.ExecutionGuard(self.manifest,'a'*64,environ=self.env,now=lambda:1001,**kwargs)

    def test_direct_runner_guard_cannot_accept_gpu0_or_nonancestor(self):
        self.guard()
        with self.assertRaisesRegex(ValueError,'ancestor'):self.guard(ancestor_pids={1})
        self.manifest['hardware']['gpu_uuid']='GPU-b1a31911-c7b4-2901-3d8b-9a62e228bfc0'
        with self.assertRaisesRegex(ValueError,'GPU1'):self.guard()

    def test_continuous_guard_rejects_deadline_foreign_owner_and_physical_remap(self):
        guard=self.guard();guard.check()
        guard.now=lambda:5001
        with self.assertRaisesRegex(ValueError,'expired'):guard.check()
        guard.now=lambda:1001;self.pids.return_value={888}
        with self.assertRaisesRegex(ValueError,'foreign'):guard.check()
        self.pids.return_value=set();self.gpu['index']='0'
        with self.assertRaisesRegex(ValueError,'GPU1'):guard.check()

    def test_ref_and_snapshot_resolver_bypass_rejected_continuously(self):
        guard=self.guard();item=self.manifest['configs'][3];snapshot=Path(item['snapshot']);ref=snapshot.parent.parent/'refs'/'main'
        ref.write_text('f'*40)
        with self.assertRaisesRegex(ValueError,'ref drift'):guard.check()
        ref.write_text(item['revision']);(snapshot.parent/('f'*40)).mkdir()
        with self.assertRaisesRegex(ValueError,'unexpected snapshot'):guard.check()

    def test_same_size_weight_change_is_detected_without_payload_hash_per_check(self):
        guard=self.guard();file=Path(self.manifest['configs'][0]['files'][0]['path'])
        file.write_bytes(b'x'*file.stat().st_size)
        with self.assertRaisesRegex(ValueError,'metadata changed'):guard.check()

    def test_shared_guard_rejects_unresolved_desktop_identity(self):
        self.lease['desktop_baseline']=[dict(pid=2356,name='[Insufficient Permissions]',created_epoch=100)]
        self.path.unlink();b.write(self.path,self.lease)
        with self.assertRaisesRegex(ValueError,'known process identity'):self.guard()

    def test_scored_digests_cannot_accept_failed_or_missing_calibration(self):
        value=dict(calibration_receipts_sha256='b'*64,calibration_estimate_sha256='c'*64)
        with patch.object(b,'calibration_bundle',side_effect=ValueError('six successful validated calibration receipts required')):
            with self.assertRaisesRegex(ValueError,'successful'):b.accepted_scored_authorization(self.root,value)
        self.assertFalse((self.root/'scored-authorization.json').exists())

    def test_direct_scored_guard_requires_accepted_authorization(self):
        self.lease['phase']='scored';self.path.unlink();b.write(self.path,self.lease)
        with self.assertRaises((ValueError,KeyError,TypeError)):self.guard(phase='scored')


if __name__ == '__main__':
    unittest.main()
