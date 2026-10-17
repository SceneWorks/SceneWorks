"""CPU-only frozen-device recovery; no device queries, rendering or downloads."""
from pathlib import Path
import sys
import io
import json
import hashlib
import zipfile
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]/'scripts/evaluation/moss'
sys.path.insert(0,str(ROOT))
import ownership as o
import bootstrap as b
import device_recovery as d
import evaluate as ev

GPU0='GPU-b1a31911-c7b4-2901-3d8b-9a62e228bfc0'  # source-bound expected identity; all queries below mocked


def test_explicit_zero_binding_and_wrong_index_uuid_fail_closed():
    expected=dict(physical_gpu_index=0,gpu_uuid=GPU0)
    assert o.device_identity(expected)==(0,GPU0)
    assert o.verify_device(expected,[dict(index='0',uuid=GPU0)])['index']=='0'
    for rows in ([dict(index='1',uuid=GPU0)],[],[dict(index='0',uuid=GPU0)]*2):
        with pytest.raises(ValueError):o.verify_device(expected,rows)
    for bad in (dict(gpu_uuid=GPU0),dict(physical_gpu_index=True,gpu_uuid=GPU0),dict(physical_gpu_index=2,gpu_uuid=GPU0),dict(physical_gpu_index=0,gpu_uuid='0')):
        with pytest.raises(ValueError):o.device_identity(bad)
    assert o.device_identity(dict(gpu_uuid=o.RUNNER_GPU_UUID))==(1,o.RUNNER_GPU_UUID)


def test_idle_gpu0_allows_only_reviewed_target_owners_not_gpu1(monkeypatch):
    lease=dict(physical_gpu_index=0,gpu_uuid=GPU0,desktop_baseline=[],baseline_max_memory_mib=1024,baseline_max_utilization_percent=2)
    monkeypatch.setattr(o,'desktop_identities',lambda lease:set())
    monkeypatch.setattr(o,'gpu_rows',lambda:[dict(index='0',uuid=GPU0,used_mib='20',utilization_percent='0'),dict(index='1',uuid=o.RUNNER_GPU_UUID,used_mib='5000',utilization_percent='99')])
    monkeypatch.setattr(o,'run',lambda argv:GPU0+', 999\n'+o.RUNNER_GPU_UUID+', 888\n')
    with pytest.raises(ValueError,match='owners differ'):o.idle_device(lease)
    monkeypatch.setattr(o,'run',lambda argv:o.RUNNER_GPU_UUID+', 888\n')
    assert o.idle_device(lease)['gpu']['index']=='0'


def test_live_guard_gpu0_keeps_lease_cache_deadline_and_foreign_owner_policy():
    from test_bootstrap import SharedOwnershipContracts
    fixture=SharedOwnershipContracts();fixture.setUp()
    try:
        fixture.manifest['hardware'].update(physical_gpu_index=0,gpu_uuid=GPU0)
        fixture.lease.update(physical_gpu_index=0,gpu_uuid=GPU0)
        fixture.path.unlink();fixture.path=fixture.root.parent/'device-locks'/(GPU0+'.lock');b.write(fixture.path,fixture.lease);fixture.env['MOSS_DEVICE_LOCK']=str(fixture.path)
        fixture.gpu.update(index='0',uuid=GPU0)
        guard=fixture.guard();guard.check()
        fixture.pids.return_value={777}
        with pytest.raises(ValueError,match='foreign'):guard.check()
        fixture.pids.return_value=set();fixture.gpu['index']='1'
        with pytest.raises(ValueError,match='GPU0'):guard.check()
        fixture.gpu['index']='0';guard.now=lambda:6000
        with pytest.raises(ValueError,match='expired'):guard.check()
    finally:fixture.tearDown()


def test_native_gpu0_environment_and_measurement_mapping(tmp_path,monkeypatch):
    import native_services as ns
    binary=tmp_path/'api.exe';binary.write_bytes(b'never executed');hub=tmp_path/'hub';hub.mkdir();calls=[]
    class Process:
        def __init__(self,args,**kw):self.pid=100+len(calls);self.stdin=io.BytesIO();self.code=None;calls.append(kw['env'])
        def poll(self):return self.code
        def terminate(self):self.code=-15
        def wait(self,**kw):return self.code
    monkeypatch.setattr(ns.subprocess,'Popen',Process)
    monkeypatch.setattr(ns,'physical_index',lambda uuid:'0')
    class Response(io.BytesIO):
        def __enter__(self):return self
        def __exit__(self,*a):self.close()
    monkeypatch.setattr(ns,'urlopen',lambda *a,**kw:Response(b'[{"id":"fixture","status":"idle"}]'))
    settings=dict(state_root=str(tmp_path/'state'),binary=str(binary),binary_sha256=ev.file_hash(binary),hub_cache=str(hub),device_id=GPU0,physical_gpu_index=0,port=17931,worker_id='fixture',runtime_versions={'fixture':'1'})
    with ns.services(settings) as context:
        assert context['physical_gpu_index']=='0' and context['gpu_uuid']==GPU0
        assert all(x['SCENEWORKS_GPU_ID']=='0' and x['CUDA_VISIBLE_DEVICES']==GPU0 and x['NVIDIA_VISIBLE_DEVICES']=='0' for x in calls)
    monkeypatch.setattr(ns,'physical_index',lambda uuid:'1')
    with pytest.raises(ValueError,match='index/UUID'):
        with ns.services(dict(settings,state_root=str(tmp_path/'other'))):pytest.fail('remapped device spawned')
    assert len(calls)==2


def test_worker_sets_uuid_logical_mapping_before_reference_import(tmp_path,monkeypatch):
    import run
    import queue
    import os
    captured={}
    class Reference:
        def __init__(self,config,mode):captured.update({k:os.environ[k] for k in ('CUDA_VISIBLE_DEVICES','MOSS_PHYSICAL_GPU_INDEX','PYTHONDONTWRITEBYTECODE')})
    monkeypatch.setitem(sys.modules,'reference',SimpleNamespace(Reference=Reference))
    for key in ('CUDA_VISIBLE_DEVICES','MOSS_PHYSICAL_GPU_INDEX','PYTHONDONTWRITEBYTECODE'):monkeypatch.setenv(key,'old')
    requests=queue.Queue();requests.put(None);replies=queue.Queue()
    run.worker(dict(id='moss_base',settings={'physical_gpu_index':0}),requests,replies,str(tmp_path/'worker.log'),GPU0)
    assert replies.get()[0]=='ready'
    assert captured==dict(CUDA_VISIBLE_DEVICES=GPU0,MOSS_PHYSICAL_GPU_INDEX='0',PYTHONDONTWRITEBYTECODE='1')


@pytest.fixture
def recovery(tmp_path,monkeypatch):
    root=tmp_path/'sc-26322-reference-v9';root.mkdir();old=tmp_path/'sc-26322-reference-v8';old.mkdir()
    contents={'protocol.lock.json':b'old raw protocol','run/receipts/scored.moss_base.x.json':b'old unaccepted row','run/outputs/scored.moss_base.x.png':b'failed-candidate image bytes','run/invalidations/foreign.json':b'foreign GPU1 owner'}
    manifest=tmp_path/'manifest.json';manifest.write_text('reviewed exact manifest fixture')
    monkeypatch.setattr(d,'MANIFEST',manifest);monkeypatch.setattr(d,'MANIFEST_SHA256',o.digest(manifest))
    data=io.BytesIO()
    with zipfile.ZipFile(data,'w') as z:
        for name,payload in contents.items():
            p=old/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(payload);z.writestr(name,payload)
    raw=data.getvalue();value=dict(files=[dict(path=n,bytes=len(v),sha256=hashlib.sha256(v).hexdigest()) for n,v in contents.items()],original_zip_sha256=hashlib.sha256(raw).hexdigest(),original_zip_bytes=len(raw))
    monkeypatch.setattr(d,'authority',lambda:value);monkeypatch.setattr(d,'download_zip',lambda value:raw)
    request=dict(authority='explicit coordinator device switch',responsible_chat='chat')
    return root,old,request


def test_fresh_recovery_preserves_raw_subset_no_imports_or_old_claim_writes(recovery):
    root,old,request=recovery
    before={p.relative_to(old).as_posix():p.read_bytes() for p in old.rglob('*') if p.is_file()}
    binding=d.install(root,request);d.verify(root,binding)
    assert binding['accounting']['current_successful_output_cap']==318 and binding['current_rows_imported']==0
    assert not (root/'run').exists()
    assert before=={p.relative_to(old).as_posix():p.read_bytes() for p in old.rglob('*') if p.is_file()}
    m=ev.template();assert len(list(ev.cells(m,'calibration')))==6 and len(list(ev.cells(m)))==288 and len(ev.parity_cells(m))==5
    for n,raw in before.items():assert (root/d.DIRECTORY/'files'/n).read_bytes()==raw
    with pytest.raises(ValueError):d.install(root,request)
    p=root/d.DIRECTORY/'files'/'protocol.lock.json';p.write_bytes(b'tampered')
    with pytest.raises(ValueError,match='preserved bytes'):d.verify(root,binding)


def test_recovery_refuses_tampered_prior_bytes_before_any_archive_write(recovery):
    root,old,request=recovery;(old/'protocol.lock.json').write_bytes(b'tampered')
    with pytest.raises(ValueError,match='selected v8 file changed'):d.install(root,request)
    assert not (root/d.DIRECTORY).exists()


def test_recovery_refuses_existing_current_run_or_reparse_source(recovery,tmp_path):
    root,old,request=recovery;(root/'run').mkdir()
    with pytest.raises(ValueError,match='already has evidence'):d.install(root,request)
    assert not (root/d.DIRECTORY).exists()
    (root/'run').rmdir();file=old/'protocol.lock.json';file.unlink();outside=tmp_path/'other';outside.write_bytes(b'old raw protocol');file.symlink_to(outside)
    with pytest.raises(ValueError,match='reparse'):d.install(root,request)
    assert not (root/d.DIRECTORY).exists()


def test_new_control_requires_exact_recovery_and_device_before_any_spawn(tmp_path):
    deadline=(b.utc()+b.dt.timedelta(minutes=20)).isoformat()
    copy=dict(kind='fresh_gpu0_v8',campaign='sc-26322-reference-v8',prepared_sha256='a'*64,job='123-1',job_receipt_sha256='b'*64,binary_sha256='c'*64)
    value=dict(campaign='sc-26322-reference-v9',deadline_utc=deadline,reuse=dict(campaign='sc-26322-reference-v1',prepared_sha256='d'*64,job='122-1',job_receipt_sha256='e'*64,native_copy=copy),device_recovery=dict(kind='fresh_gpu0_v8',manifest_sha256=d.MANIFEST_SHA256,authority='user explicitly switched to GPU0',responsible_chat='chat'))
    assert b.control('prepare',json.dumps(value),tmp_path)[1].name==value['campaign']
    from copy import deepcopy
    for mutate in (lambda x:x.pop('device_recovery'),lambda x:x['device_recovery'].update(manifest_sha256='f'*64),lambda x:x['reuse'].update(native_rebuild=True),lambda x:x['reuse']['native_copy'].update(campaign='sc-26322-reference-v7')):
        bad=deepcopy(value);mutate(bad)
        with pytest.raises(ValueError):b.control('prepare',json.dumps(bad),tmp_path)
    own=dict(responsible_chat='chat',lease='lease',job='job',timeout_seconds=1800,resume='reviewed command',next_action='review')
    freeze=dict(campaign=value['campaign'],deadline_utc=deadline,physical_gpu_index=0,gpu_uuid=GPU0,ownership=own)
    b.control('freeze',json.dumps(freeze),tmp_path)
    for change in ({'physical_gpu_index':1},{'physical_gpu_index':True},{'gpu_uuid':'0'}):
        with pytest.raises(ValueError):b.control('freeze',json.dumps({**freeze,**change}),tmp_path)


def test_exact_v8_native_copy_never_builds_and_checks_dll_identity(tmp_path,monkeypatch):
    source=tmp_path/'sc-26322-reference-v8';binary=source/'target/release/sceneworks-rust-api.exe';binary.parent.mkdir(parents=True);binary.write_bytes(b'exact frozen evaluated native bytes')
    native=tmp_path/'sc-26322-reference-v1/native-source';native.mkdir(parents=True);(native/'Cargo.lock').write_bytes(b'frozen lock')
    identity=dict(required_adjacent_dlls=[],adjacent_dll_files=[],dependency_candidates=[dict(name='fixture.dll',candidate_path='frozen path',sha256='a'*64,loaded_proof=False)])
    prep=dict(binary=str(binary),binary_sha256=b.digest(binary),native_build={**identity,'build':dict(cwd=str(native),input_sha256={'Cargo.lock':b.digest(native/'Cargo.lock')})})
    b.write(source/'prepared.json',prep);receipt=source/'job-123-1-prepare.json';b.write(receipt,dict(status='completed',mode='prepare',job='123-1',root=str(source),sceneworks_source=b.SOURCE,inference=b.INFERENCE,accelerator_execution_requested=False))
    request=dict(kind='fresh_gpu0_v8',campaign=source.name,prepared_sha256=b.digest(source/'prepared.json'),job='123-1',job_receipt_sha256=b.digest(receipt),binary_sha256=prep['binary_sha256']);root=tmp_path/'sc-26322-reference-v9';record={'native_copy':request}
    monkeypatch.setattr(b,'prepared',lambda path:prep);monkeypatch.setattr(b,'verify_native_source',lambda path:None);monkeypatch.setattr(b,'native_identity',lambda path:identity)
    monkeypatch.setattr(b,'run',lambda *a,**kw:pytest.fail('no Cargo/build/executable allowed'))
    target,_=b.copy_v6_native(root,record)
    assert target.read_bytes()==binary.read_bytes() and target.stat().st_ino!=binary.stat().st_ino
    assert record['native']['mode']=='exact_v8_binary_copy_no_build'
    monkeypatch.setattr(b,'native_identity',lambda path:{**identity,'dependency_candidates':[]})
    with pytest.raises(ValueError,match='DLL candidate identity changed'):b.copy_v6_native(root,record)


@pytest.mark.parametrize('config_id',['moss_base','qwen_image_2_1'])
def test_measurements_must_bind_actual_selected_device(config_id,tmp_path):
    from test_moss_reference_evaluation import png
    m=ev.template();m['hardware'].update(physical_gpu_index=0,gpu_uuid=GPU0)
    m['ownership']={k:'fixture' for k in m['ownership']};m['ownership']['timeout_seconds']=10
    cell=next(c for c in ev.cells(m,'calibration') if c['config_id']==config_id)
    image=tmp_path/'image.png';image.write_bytes(png(1024,1024));log=tmp_path/'log';log.write_text('actual fixture viewing not claimed')
    device=dict(physical_gpu_index=0,visible_uuid=GPU0,logical_index=0)
    metrics=dict(generation_seconds=.1,peak_gpu_bytes=42,peak_host_rss_bytes=42,download_bytes=0,latency_condition='cold_process',cold_seconds=.1,warm_seconds=None,gpu_memory_method='CPU test fixture',component_dtypes={'fixture':'float32'},sampler={'fixture':'test'},runtime_versions={'fixture':'1'},pixels_finite=True,device=device)
    if not config_id.startswith('moss_'):metrics['native_proof']=dict(runtime_context=dict(physical_gpu_index='0',gpu_uuid=GPU0))
    row=dict(lock_sha256='a'*64,cell=cell,status='ok',ownership=dict(m['ownership'],revision='fixture'),elapsed_seconds=.1,output=dict(path='image.png',sha256=ev.file_hash(image)),log=dict(path='log',sha256=ev.file_hash(log)),metrics=metrics)
    ev.validate_receipt(tmp_path,'a'*64,cell,ev.seal(row),manifest=m)
    for change in ({'physical_gpu_index':1},{'physical_gpu_index':False},{'visible_uuid':o.RUNNER_GPU_UUID}):
        metrics['device']={**device,**change}
        with pytest.raises(ValueError,match='physical GPU proof'):ev.validate_receipt(tmp_path,'a'*64,cell,ev.seal(row),manifest=m)


def test_fixed_actual_prior_manifest_zip_inventory_matches_no_historical_status_relabel():
    a=d.authority()
    assert len(a['files'])==270 and a['scored_ok_label_inventory_only']=={'moss_base':48,'moss_sft':3}
    assert a['current_rows_import_permitted']==0 and a['individually_invalidated_ok_label_rows']==3
    assert d.ACCOUNTING['prior_calibration_ok_labels']==6 and d.ACCOUNTING['prior_failed_calibration_pngs']==1
    assert d.ACCOUNTING['lifetime_ok_label_upper_bound']==375 and d.ACCOUNTING['lifetime_png_upper_bound_including_failed']==376
