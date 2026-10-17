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
import warnings
import zipfile

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

    def test_reuse_control_binds_owned_sibling_and_completed_job_digests(self):
        reuse=dict(campaign='sc-26322-reference-v1',prepared_sha256='a'*64,job='123-1',job_receipt_sha256='b'*64)
        b.control('prepare',self.request(reuse=reuse),self.root)
        b.control('prepare',self.request(reuse={**reuse,'native_rebuild':True}),self.root)
        for invalid in (False,'true',1,None):
            with self.assertRaisesRegex(ValueError,'native_rebuild'):
                b.control('prepare',self.request(reuse={**reuse,'native_rebuild':invalid}),self.root)
        for update in (dict(campaign='../escape'),dict(campaign='sc-26322-contract'),dict(job='latest'),dict(prepared_sha256='HEAD'),dict(path='E:/production')):
            with self.assertRaises(ValueError):
                b.control('prepare',self.request(reuse={**reuse,**update}),self.root)

    def test_seed_verified_bytes_copy_only_binary_and_tamper_refusal(self):
        source=self.root/'immutable';source.write_bytes(b'owned fixture');sha=b.digest(source)
        target=self.root/'new'/'blob'
        receipt=b.seed_file(source,target,sha)
        self.assertEqual(target.read_bytes(),source.read_bytes())
        self.assertIn(receipt['method'],('hardlink','copy'))
        binary=self.root/'new'/'binary'
        self.assertEqual(b.seed_file(source,binary,sha,link=False)['method'],'copy')
        self.assertNotEqual(binary.stat().st_ino,source.stat().st_ino)
        binary.write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError,'destination bytes'):
            b.seed_file(source,binary,sha,link=False)
        with self.assertRaisesRegex(ValueError,'source bytes'):
            b.seed_file(source,self.root/'other','0'*64)

    def test_seed_copy_reserves_build_and_output_headroom(self):
        source=self.root/'source';source.write_bytes(b'fixture')
        with patch.object(b.shutil,'disk_usage',return_value=types.SimpleNamespace(free=48*1024**3)):
            with self.assertRaisesRegex(ValueError,'plus48GiB headroom'):
                b.seed_file(source,self.root/'destination',b.digest(source),link=False)
        self.assertFalse((self.root/'destination').exists())

    def test_reuse_requires_actual_completed_receipt_before_any_seeding(self):
        source=self.root/'sc-26322-old';source.mkdir();target=self.root/'sc-26322-new';target.mkdir()
        job=dict(mode='prepare',status='failed',job='123-1',root=str(source),sceneworks_source=b.SOURCE,inference=b.INFERENCE,accelerator_execution_requested=False)
        b.write(source/'prepared.json',{})
        receipt=source/'job-123-1-prepare.json';b.write(receipt,job)
        value=dict(reuse=dict(campaign=source.name,job='123-1',prepared_sha256=b.digest(source/'prepared.json'),job_receipt_sha256=b.digest(receipt)))
        with patch.object(b,'seed_file') as seed:
            with self.assertRaisesRegex(ValueError,'not complete'):b.reuse_inputs(target,value)
            seed.assert_not_called()
        value['reuse']['job_receipt_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'receipt changed'):b.reuse_inputs(target,value)

    def test_native_reuse_runs_same_offline_locked_build_then_copies_unchanged_binary(self):
        source=self.root/'old';native=source/'native-source';native.mkdir(parents=True)
        binary=source/'target'/'release'/'sceneworks-rust-api.exe';binary.parent.mkdir(parents=True);binary.write_bytes(b'PE fixture')
        (source/'cargo-home').mkdir();new=self.root/'new';new.mkdir()
        prep=dict(binary=str(binary),binary_sha256=b.digest(binary));record={}
        identity=dict(required_adjacent_dlls=[],adjacent_dll_files=[])
        with patch.object(b,'verify_native_source') as verify,patch.object(b,'native_identity',return_value=identity),patch.object(b,'run') as run:
            target,_=b.reuse_native(new,source,prep,record)
            self.assertEqual(verify.call_args_list,[unittest.mock.call(native),unittest.mock.call(native)])
            self.assertEqual(run.call_args.kwargs['cwd'],native)
            env=run.call_args.kwargs['env']
            self.assertEqual(env['CARGO_NET_OFFLINE'],'true')
            self.assertEqual(env['CARGO_HOME'],str(source/'cargo-home'))
            self.assertEqual(env['CARGO_TARGET_DIR'],str(source/'target'))
            self.assertIn('--locked',run.call_args.args[0])
            self.assertEqual(b.digest(target),prep['binary_sha256'])
            self.assertEqual(record['native']['binary']['method'],'copy')
            self.assertEqual(record['native']['identity']['build']['mode'],'exact_binary_reuse')
            self.assertNotEqual(target.stat().st_ino,binary.stat().st_ino)
        with patch.object(b,'verify_native_source'),patch.object(b,'run'),patch.object(b,'native_identity',return_value={**identity,'required_adjacent_dlls':['unreviewed.dll']}):
            with self.assertRaisesRegex(ValueError,'adjacent DLLs'):b.reuse_native(new,source,prep,{})
        def changed(*args,**kwargs):binary.write_bytes(b'changed build')
        with patch.object(b,'verify_native_source'),patch.object(b,'run',side_effect=changed):
            with self.assertRaisesRegex(ValueError,'changed source binary'):b.reuse_native(new,source,prep,{})

    def test_explicit_native_rebuild_recovers_drift_without_trusting_old_executable(self):
        source=self.root/'old';native=source/'native-source';native.mkdir(parents=True)
        binary=source/'target'/'release'/'sceneworks-rust-api.exe';binary.parent.mkdir(parents=True)
        binary.write_bytes(b'original prepared executable');original=b.digest(binary)
        prep=dict(binary=str(binary),binary_sha256=original)
        b.write(source/'prepared.json',prep);old_receipt=(source/'prepared.json').read_bytes()
        (native/'Cargo.lock').write_text('controlled lock');(source/'cargo-home').mkdir()
        binary.write_bytes(b'mutable target left by failed prior reuse')
        observed=b.digest(binary);commands=[];new=self.root/'new';new.mkdir()
        def cargo(args,**kwargs):
            commands.append(args)
            self.assertEqual(kwargs['env']['CARGO_NET_OFFLINE'],'true')
            if args[1]=='clean':binary.unlink()
            else:
                self.assertFalse(binary.exists())
                # A controlled build may embed different non-source inputs. This
                # synthetic fixture establishes recovery, not the PE drift cause.
                binary.write_bytes(('fresh build '+kwargs['env']['RUSTFLAGS']).encode())
        identity=dict(required_adjacent_dlls=[],adjacent_dll_files=[],rustc='actual compiler fixture')
        with patch.object(b,'verify_native_source'),patch.object(b,'run',side_effect=cargo),patch.object(b,'native_identity',return_value=identity),patch.dict(os.environ,{'RUSTFLAGS':'new controlled build flags'}):
            with self.assertRaisesRegex(ValueError,'native binary drift'):b.reuse_native(new,source,prep,{})
            self.assertFalse(commands)
            record={'native_rebuild':True};target,evidence=b.reuse_native(new,source,prep,record)
        self.assertEqual(commands[0],['cargo','clean','--release','-p','sceneworks-rust-api'])
        self.assertIn('--locked',commands[1])
        self.assertEqual((source/'prepared.json').read_bytes(),old_receipt)
        self.assertNotEqual(b.digest(target),original)
        self.assertFalse(os.path.samefile(binary,target))
        build=evidence['build']
        self.assertEqual(build['mode'],'controlled_rebuild')
        self.assertEqual(build['prior_prepared_binary_sha256'],original)
        self.assertEqual(build['observed_before_binary_sha256'],observed)
        self.assertEqual(build['binary_sha256'],b.digest(target))
        self.assertFalse(build['byte_identity_to_prior_prepared'])
        self.assertEqual(build['input_sha256']['Cargo.lock'],b.digest(native/'Cargo.lock'))
        self.assertEqual(build['environment']['RUSTFLAGS'],'new controlled build flags')
        binary.write_bytes(b'later cargo mutation')
        self.assertEqual(b.digest(target),build['binary_sha256'])

    def test_explicit_native_rebuild_requires_clean_to_remove_old_binary(self):
        source=self.root/'old';native=source/'native-source';native.mkdir(parents=True)
        binary=source/'target'/'release'/'sceneworks-rust-api.exe';binary.parent.mkdir(parents=True);binary.write_bytes(b'old')
        (source/'cargo-home').mkdir()
        with patch.object(b,'verify_native_source'),patch.object(b,'run') as run,patch.object(b,'native_identity') as identity:
            with self.assertRaisesRegex(ValueError,'clean retained'):
                b.reuse_native(self.root/'new',source,dict(binary=str(binary),binary_sha256=b.digest(binary)),{'native_rebuild':True})
        self.assertEqual(run.call_count,1);identity.assert_not_called()

    def test_native_source_pin_and_own_cache_physical_ancestors_required(self):
        native=self.root/'native';native.mkdir()
        with patch.object(b,'run',return_value='0'*40):
            with self.assertRaisesRegex(ValueError,'source mismatch'):b.verify_native_source(native)
        outside=self.root/'outside';outside.mkdir();source=self.root/'source';source.mkdir()
        (source/'cargo-home').symlink_to(outside,target_is_directory=True)
        with patch.object(b,'run') as run:
            with self.assertRaisesRegex(ValueError,'reparse'):b.reuse_native(self.root/'new',source,dict(binary=str(source/'target'/'x')), {})
            run.assert_not_called()

    def test_retained_reviewed_torch_local_version_basename_and_path_rejection(self):
        wheels=self.root/'wheels';wheels.mkdir();name='torch-2.9.1+cu128-cp312-cp312-win_amd64.whl'
        (wheels/name).write_bytes(b'controlled wheel');sha=b.digest(wheels/name)
        (self.root/'requirements.lock').write_text('fixture lock')
        manifest=dict(primary=list(b.PACKAGES),requirements_sha256=b.digest(self.root/'requirements.lock'),files=[dict(name=name,sha256=sha)])
        b.write(self.root/'wheels.json',manifest)
        resolution=dict(packages=[dict(url='https://download.pytorch.org/whl/'+name.replace('+','%2B'),sha256=sha)])
        with patch.object(b,'dependency_resolution',return_value=resolution),patch.object(b,'run') as run:
            b.lock_wheels(self.root);run.assert_not_called()
            for unsafe in ('../'+name,'/tmp/'+name,'nested/'+name,'C:\\'+name):
                manifest['files'][0]['name']=unsafe
                (self.root/'wheels.json').write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError,'filenames'):b.lock_wheels(self.root)

    def test_reuse_seeds_only_verified_inputs_with_fresh_refs_and_harness(self):
        source=self.root/'sc-26322-old';source.mkdir();target=self.root/'sc-26322-new';target.mkdir()
        staged={};payload=b'CPU fixture, no tensor data'
        for key,(_,revision) in b.REPOS.items():
            name=('transformer/' if key.startswith('moss_') else 'q4/')+'a.safetensors'
            siblings=[dict(rfilename=name,size=len(payload),lfs={'sha256':hashlib.sha256(payload).hexdigest()})]
            if key=='moss_sft':siblings.append(dict(rfilename='pipeline_moss_image.py',size=len(payload),lfs={'sha256':hashlib.sha256(payload).hexdigest()}))
            b.write(source/'metadata'/(key+'.json'),dict(sha=revision,siblings=siblings))
            with patch.object(b,'PIPELINE',hashlib.sha256(payload).hexdigest()),patch.object(b.urllib.request,'urlopen',side_effect=lambda *a,**k:io.BytesIO(payload)):staged[key]=b.stage(source,key)
        wheels=source/'wheels';wheels.mkdir();wheel=wheels/'fixture-1-py3-none-any.whl';wheel.write_bytes(b'wheel fixture')
        lock=source/'requirements.lock';lock.write_text('fixture==1 --hash=sha256:'+b.digest(wheel))
        b.write(source/'wheels.json',dict(primary=list(b.PACKAGES),requirements_sha256=b.digest(lock),files=[dict(name=wheel.name,sha256=b.digest(wheel))]))
        binary=source/'target'/'release'/'sceneworks-rust-api.exe';binary.parent.mkdir(parents=True);binary.write_bytes(b'exe fixture')
        (source/'harness').mkdir();(source/'harness'/'evaluate.py').write_text('OLD HARNESS, MUST NOT BE REUSED')
        prep=dict(sceneworks_source=b.SOURCE,inference=b.INFERENCE,binary=str(binary),binary_sha256=b.digest(binary),requirements_sha256=b.digest(lock),staged=staged,harness={'evaluate.py':b.digest(source/'harness'/'evaluate.py')})
        b.write(source/'prepared.json',prep)
        b.write(source/'job-123-1-prepare.json',dict(mode='prepare',status='completed',job='123-1',root=str(source),sceneworks_source=b.SOURCE,inference=b.INFERENCE,accelerator_execution_requested=False))
        value=dict(reuse=dict(campaign=source.name,job='123-1',prepared_sha256=b.digest(source/'prepared.json'),job_receipt_sha256=b.digest(source/'job-123-1-prepare.json')))
        old_prepared=(source/'prepared.json').read_bytes()
        resolution=dict(packages=[dict(url='https://example.org/'+wheel.name,sha256=b.digest(wheel))])
        with patch.object(b,'PIPELINE',hashlib.sha256(payload).hexdigest()),patch.object(b,'dependency_resolution',return_value=resolution),patch.object(b,'fetch_json',side_effect=AssertionError('network forbidden')),patch.object(b.urllib.request,'urlopen',side_effect=AssertionError('network forbidden')),patch.object(b,'run',side_effect=AssertionError('subprocess forbidden')):
            _,_,record=b.reuse_inputs(target,value)
            for key in b.REPOS:
                row=b.stage(target,key)
                old_ref=Path(staged[key]['snapshot']).parent.parent/'refs'/'main';new_ref=Path(row['snapshot']).parent.parent/'refs'/'main'
                self.assertEqual(new_ref.read_bytes(),old_ref.read_bytes())
                self.assertFalse(os.path.samefile(old_ref,new_ref))
        self.assertFalse((target/'harness').exists())
        self.assertFalse((target/'venv').exists())
        self.assertEqual((source/'prepared.json').read_bytes(),old_prepared)
        self.assertEqual((source/'harness'/'evaluate.py').read_text(),'OLD HARNESS, MUST NOT BE REUSED')
        self.assertEqual(record['downloaded_seed_bytes'],0)
        self.assertEqual(len(record['seeds']),1+2+6+7)
        binary.write_bytes(b'previous controlled build changed mutable target')
        value['reuse']['native_rebuild']=True
        other=self.root/'sc-26322-recovered';other.mkdir()
        with patch.object(b,'PIPELINE',hashlib.sha256(payload).hexdigest()),patch.object(b,'dependency_resolution',return_value=resolution),patch.object(b,'run',side_effect=AssertionError('subprocess forbidden')):
            with self.assertRaisesRegex(ValueError,'native binary drift'):b.prepared(source)
            _,_,recovery=b.reuse_inputs(other,value)
            self.assertTrue(recovery['native_rebuild'])
            self.assertFalse((other/'target').exists())
            self.assertEqual((source/'prepared.json').read_bytes(),old_prepared)
            (source/'harness'/'evaluate.py').write_text('tampered immutable harness')
            with self.assertRaisesRegex(ValueError,'harness drift'):b.reuse_inputs(other,value)

    def test_native_identity_records_compiler_and_unknown_dlls_without_executing_binary(self):
        binary=self.root/'app.exe';binary.write_bytes(b'fixture')
        outputs={'dumpbin':'  KERNEL32.dll\n  unknown_model.dll\n','rustc':'rustc actual compiler','cargo':'cargo actual version','nvcc':'native CUDA compiler actual version'}
        def run(args,**kwargs):return outputs[args[0]]
        with patch.object(b,'run',side_effect=run) as commands,patch.dict(os.environ,{'SystemRoot':str(self.root/'no-system'),'PATH':str(self.root/'no-path'),'CUDA_PATH':'native-cuda-path'}):
            result=b.native_identity(binary)
        self.assertEqual(result['rustc'],outputs['rustc'])
        self.assertEqual(result['nvcc'],outputs['nvcc'])
        self.assertEqual(result['cuda_path'],'native-cuda-path')
        self.assertEqual(result['dependency_names'],['KERNEL32.dll','unknown_model.dll'])
        self.assertTrue(all(c['candidate_path'] is None and c['loaded_proof'] is False for c in result['dependency_candidates']))
        self.assertTrue(all(str(binary)!=str(c.args[0][0]) for c in commands.call_args_list))

    def test_literal_dll_candidates_ignore_pathext_preserve_order_and_reject_exe_decoys(self):
        binary=self.root/'app.exe';binary.write_bytes(b'not executed')
        system=self.root/'windows'/'System32';system.mkdir(parents=True)
        first=self.root/'cuda-bin';first.mkdir();second=self.root/'other-bin';second.mkdir()
        (system/'shared.dll').write_bytes(b'system takes precedence')
        (first/'shared.dll').write_bytes(b'PATH duplicate')
        (first/'toolkit.dll').write_bytes(b'first literal PATH DLL')
        (second/'toolkit.dll').write_bytes(b'second PATH DLL')
        (first/'decoy.dll.exe').write_bytes(b'executable decoy must not match')
        outputs={'dumpbin':'  shared.dll\n  toolkit.dll\n  decoy.dll\n  api-ms-win-missing.dll\n','powershell':'1.2.3','rustc':'compiler','cargo':'cargo','nvcc':'nvcc'}
        env={'SystemRoot':str(system.parent),'PATH':os.pathsep.join((str(first),str(second))),'PATHEXT':'.COM;.EXE;.BAT'}
        with patch.object(b,'run',side_effect=lambda args,**kwargs:outputs[args[0]]) as commands,patch.object(b.shutil,'which',side_effect=AssertionError('executable lookup forbidden')),patch.dict(os.environ,env):
            result=b.native_identity(binary)
            self.assertEqual({key:os.environ[key] for key in env},env)
        rows={row['name']:row for row in result['dependency_candidates']}
        for name,path in (('shared.dll',system/'shared.dll'),('toolkit.dll',first/'toolkit.dll')):
            self.assertEqual(rows[name]['candidate_path'],str(path))
            self.assertEqual(rows[name]['sha256'],b.digest(path))
            self.assertEqual(rows[name]['file_version'],'1.2.3')
        self.assertIsNone(rows['decoy.dll']['candidate_path'])
        self.assertIsNone(rows['api-ms-win-missing.dll']['candidate_path'])
        self.assertTrue(all(row['loaded_proof'] is False for row in rows.values()))
        self.assertEqual(result['adjacent_dll_files'],[])
        self.assertTrue(all(str(binary)!=str(call.args[0][0]) for call in commands.call_args_list))

    def test_prepared_receipt_binds_reuse_ledger(self):
        binary=self.root/'binary';binary.write_bytes(b'fixture');lock=self.root/'requirements.lock';lock.write_text('locked')
        ledger=self.root/'reuse.json';b.write(ledger,dict(source_prepared_sha256='a'*64))
        value=dict(sceneworks_source=b.SOURCE,inference=b.INFERENCE,binary=str(binary),binary_sha256=b.digest(binary),requirements_sha256=b.digest(lock),staged={},harness={},reuse=dict(path=str(ledger),sha256=b.digest(ledger)))
        b.write(self.root/'prepared.json',value)
        with patch.object(b,'verify_staged'):
            b.prepared(self.root)
            ledger.write_text('changed provenance')
            with self.assertRaisesRegex(ValueError,'reuse provenance drift'):b.prepared(self.root)

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

    def wheel(self, filename, entries):
        path = self.root / filename
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)  # Deliberate duplicate ZIP entry fixture.
            with zipfile.ZipFile(path, 'w') as archive:
                for name, metadata in entries:
                    archive.writestr(name, metadata)
        return path

    def setuptools_entries(self):
        # Exact METADATA path shape of the reviewed setuptools84 wheel,
        # SHA256 51a52592b3b99e102b609654876bd65f19f999935166d1352678931132b0c670.
        vendors = ('autocommand-2.2.2', 'backports.tarfile-1.2.0', 'importlib_metadata-8.7.1',
                   'jaraco.text-4.0.0', 'jaraco_context-6.1.0', 'jaraco_functools-4.4.0',
                   'more_itertools-10.8.0', 'packaging-26.0', 'platformdirs-4.4.0',
                   'tomli-2.4.0', 'wheel-0.46.3', 'zipp-3.23.0')
        return [(f'setuptools/_vendor/{vendor}.dist-info/METADATA',
                 f'Name: {vendor.rsplit("-", 1)[0]}\nVersion: {vendor.rsplit("-", 1)[1]}\n')
                for vendor in vendors] + [('setuptools-84.0.0.dist-info/METADATA',
                                          'Metadata-Version: 2.4\r\nName: setuptools\r\nVersion: 84.0.0\r\n\r\n')]

    def test_wheel_root_identity_ignores_actual_setuptools_vendor_metadata_shape(self):
        wheel = self.wheel('setuptools-84.0.0-py3-none-any.whl', self.setuptools_entries())
        with zipfile.ZipFile(wheel) as archive:
            self.assertEqual(len([n for n in archive.namelist() if n.endswith('.dist-info/METADATA')]), 13)
        self.assertEqual(b.wheel_identity(wheel, dict(name='setuptools', version='84.0.0')), ('setuptools', '84.0.0'))

    def test_wheel_root_metadata_is_required_unique_and_matches_filename(self):
        entries = self.setuptools_entries()
        for invalid in (entries[:-1], entries + [entries[-1]],
                        entries + [('other-1.0.dist-info/METADATA', 'Name: other\nVersion: 1.0\n')],
                        [('other-1.0.dist-info/METADATA', 'Name: setuptools\nVersion: 84.0.0\n')]):
            with self.subTest(paths=[name for name, _ in invalid]):
                wheel = self.wheel('setuptools-84.0.0-py3-none-any.whl', invalid)
                with self.assertRaisesRegex(ValueError, 'root wheel metadata'):
                    b.wheel_identity(wheel, dict(name='setuptools', version='84.0.0'))

    def test_wheel_identity_must_match_reviewed_name_version_and_single_headers(self):
        for metadata, error in (
                ('Name: other\nVersion: 84.0.0\n', 'reviewed dependency'),
                ('Name: setuptools\nVersion: 83.0.0\n', 'reviewed dependency'),
                ('Name: setuptools\n', 'identity fields'),
                ('Name: setuptools\nName: other\nVersion: 84.0.0\n', 'identity fields'),
                ('Name: setuptools\nVersion: 84.0.0\nVersion: 83.0.0\n', 'identity fields')):
            with self.subTest(metadata=metadata):
                wheel = self.wheel('setuptools-84.0.0-py3-none-any.whl',
                                   [('setuptools-84.0.0.dist-info/METADATA', metadata)])
                with self.assertRaisesRegex(ValueError, error):
                    b.wheel_identity(wheel, dict(name='setuptools', version='84.0.0'))
        wheel = self.wheel('example_pkg-1.0-py3-none-any.whl',
                           [('example_pkg-1.0.dist-info/METADATA', 'Name: Example.Pkg\nVersion: 1.0\n')])
        self.assertEqual(b.wheel_identity(wheel, dict(name='example-pkg', version='1.0')), ('Example.Pkg', '1.0'))

    def test_retained_downloads_can_lock_then_resume_with_hash_and_identity_checks(self):
        wheels = self.root / 'wheels'
        wheels.mkdir()
        wheel = self.wheel('setuptools-84.0.0-py3-none-any.whl', self.setuptools_entries())
        wheel = wheel.rename(wheels / wheel.name)
        sha = b.digest(wheel)
        package = dict(name='setuptools', version='84.0.0', sha256=sha, url='https://example.org/' + wheel.name)
        resolution = dict(primary=list(b.PACKAGES), packages=[package])
        with patch.object(b, 'dependency_resolution', return_value=resolution), patch.object(b, 'run') as run:
            b.lock_wheels(self.root)
            self.assertEqual(run.call_args.args[0][-1], package['url'] + '#sha256=' + sha)
            self.assertIn('--no-deps', run.call_args.args[0])
            self.assertIn('--only-binary=:all:', run.call_args.args[0])
            self.assertEqual((self.root / 'requirements.lock').read_text(), f'setuptools==84.0.0 --hash=sha256:{sha}\n')
            run.reset_mock()
            b.lock_wheels(self.root)
            run.assert_not_called()
            wheel.write_bytes(b'tamper')
            with self.assertRaisesRegex(ValueError, 'wheel content drift'):
                b.lock_wheels(self.root)
        # Same content hash with a false reviewed identity must fail before a lock is published.
        (self.root / 'requirements.lock').unlink()
        wheel = self.wheel('setuptools-84.0.0-py3-none-any.whl', self.setuptools_entries())
        wheel.replace(wheels / wheel.name)
        with patch.object(b, 'dependency_resolution', return_value={**resolution, 'packages': [{**package, 'name': 'other'}]}), patch.object(b, 'run'):
            with self.assertRaisesRegex(ValueError, 'reviewed dependency'):
                b.lock_wheels(self.root)
        self.assertFalse((self.root / 'requirements.lock').exists())

    def test_scored_control_needs_accepted_digests_and_fixed_authorization(self):
        lease = dict(id='l',owner='chat',host='host',gpu_uuid=b.RUNNER_GPU_UUID,expires_utc=self.deadline,
            external_owners_reconciled=True,desktop_baseline=[],baseline_max_memory_mib=1000,baseline_max_utilization_percent=2)
        raw=json.loads(self.request(protocol_sha256='a'*64,lease=lease))
        with self.assertRaises(ValueError):b.control('scored',json.dumps(raw),self.root)
        raw.update(calibration_receipts_sha256='b'*64,calibration_estimate_sha256='c'*64,
            authorization=dict(id='accepted',coordinator='chat',accepted_calibration=True,max_scored_outputs=288,acknowledge_shape_uncertainty=True))
        b.control('scored',json.dumps(raw),self.root)
        b.control('scored',json.dumps({**raw,'unavailable_configs':{'moss_dmd':{'reviewed':'structure validated against actual receipts during authorization'}}}),self.root)
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
        with patch.object(b,'calibration_bundle',return_value='b'*64),patch.object(b,'availability_decisions',return_value={}):
            first=b.accepted_scored_authorization(self.root,value,manifest={})
            self.assertEqual(first,b.accepted_scored_authorization(self.root,value,manifest={}))
            changed=json.loads(json.dumps(value));changed['authorization']['id']='replacement'
            with self.assertRaisesRegex(ValueError,'authorization changed'):b.accepted_scored_authorization(self.root,changed,manifest={})
            (self.root/'calibration-estimate.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'estimate digest'):b.accepted_scored_authorization(self.root,value,manifest={})

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

    def test_json_roundtrip_closure_diagnostics_keep_bytecode_strict(self):
        staged=json.loads(json.dumps(b.read(self.root/'prepared.json')['staged']))
        o.verify_staged(self.root,staged,hash_weights=False)
        item=staged['moss_sft'];missing=Path(item['files'][0]['path']);missing.unlink()
        bytecode=Path(item['snapshot'])/'__pycache__'/'pipeline_moss_image.cpython-312.pyc'
        bytecode.parent.mkdir();bytecode.write_bytes(b'injected bytecode must invalidate')
        with self.assertRaisesRegex(ValueError,'snapshot input closure changed') as failure:
            o.verify_staged(self.root,staged,hash_weights=False)
        detail=json.loads(str(failure.exception).split(': ',1)[1])
        self.assertEqual(detail['config_id'],'moss_sft')
        self.assertEqual(detail['added'],[str(bytecode.resolve())])
        self.assertEqual(detail['removed'],[str(missing)])
        self.assertEqual((detail['added_count'],detail['removed_count']),(1,1))
        self.assertFalse(detail['paths_truncated'])

    def test_same_size_weight_change_is_detected_without_payload_hash_per_check(self):
        guard=self.guard();file=Path(self.manifest['configs'][0]['files'][0]['path'])
        file.write_bytes(b'x'*file.stat().st_size)
        with self.assertRaisesRegex(ValueError,'metadata changed'):guard.check()

    def test_mixed_scored_guard_monitors_original_calibration_evidence(self):
        import evaluate as ev
        run=self.root/'run';run.mkdir();(run/'logs').mkdir();(run/'receipts').mkdir()
        cells=[]
        for key in o.REPOS:
            cell=dict(id='calibration.'+key,config_id=key);cells.append(cell)
            log=run/'logs'/(key+'.log');log.write_text('CPU guard fixture')
            b.write(run/'receipts'/(cell['id']+'.json'),dict(log=dict(path='logs/'+log.name,sha256=b.digest(log))))
        estimate=self.root/'calibration-estimate.json';b.write(estimate,{'fixture':True})
        accepted=dict(protocol_sha256='a'*64,calibration_receipts_sha256='b'*64,calibration_estimate_sha256=b.digest(estimate),authorization=dict(id='fixture',coordinator='chat',accepted_calibration=True,max_scored_outputs=288,acknowledge_shape_uncertainty=True))
        authority=self.root/'scored-authorization.json';b.write(authority,accepted);b.write(run/'availability-authorization.json',accepted)
        self.lease.update(phase='scored',scored_authorization=dict(path=str(authority),sha256=b.digest(authority)))
        self.path.unlink();b.write(self.path,self.lease)
        with patch.object(o,'calibration_bundle',return_value='b'*64),patch.object(o,'availability_decisions',return_value={'moss_dmd':{'fixture':True}}),patch.object(ev,'cells',return_value=cells):
            guard=self.guard(phase='scored')
        self.assertEqual(set(guard.unavailable_configs),{'moss_dmd'})
        (run/'logs'/'moss_dmd.log').write_text('mutated evidence')
        with self.assertRaisesRegex(ValueError,'availability calibration evidence changed'):guard.check()

    def test_shared_guard_rejects_unresolved_desktop_identity(self):
        self.lease['desktop_baseline']=[dict(pid=2356,name='[Insufficient Permissions]',created_epoch=100)]
        self.path.unlink();b.write(self.path,self.lease)
        with self.assertRaisesRegex(ValueError,'known process identity'):self.guard()

    def test_scored_digests_cannot_accept_failed_or_missing_calibration(self):
        value=dict(calibration_receipts_sha256='b'*64,calibration_estimate_sha256='c'*64)
        with patch.object(b,'calibration_bundle',side_effect=ValueError('six successful validated calibration receipts required')):
            with self.assertRaisesRegex(ValueError,'successful'):b.accepted_scored_authorization(self.root,value,manifest={})
        self.assertFalse((self.root/'scored-authorization.json').exists())

    def test_direct_scored_guard_requires_accepted_authorization(self):
        self.lease['phase']='scored';self.path.unlink();b.write(self.path,self.lease)
        with self.assertRaises((ValueError,KeyError,TypeError)):self.guard(phase='scored')


if __name__ == '__main__':
    unittest.main()
