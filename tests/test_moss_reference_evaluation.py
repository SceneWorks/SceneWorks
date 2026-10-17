"""Controlled CPU fixtures only: none of these files are campaign evidence."""
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path
import struct
import sys
import zlib

import pytest

ROOT=Path(__file__).resolve().parents[1]/'scripts/evaluation/moss'
sys.path.insert(0,str(ROOT))
import evaluate as ev
from native import request_body
from reference import dmd_loop, import_pipeline


def png(width,height):
    def chunk(kind,data):return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)
    return b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',width,height,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(bytes(height*(1+3*width))))+chunk(b'IEND',b'')


@pytest.fixture
def unavailable_run(tmp_path):
    manifest=ev.template();log=tmp_path/'controlled-fixture.log';log.write_text('CONTROLLED TEST FIXTURE: deliberate unavailable runtime\n')
    for category in ('runtime','hardware','ownership'):
        manifest[category]={k:(30 if k=='timeout_seconds' else 'CONTROLLED TEST FIXTURE') for k in manifest[category]}
    for c in manifest['configs']:
        c['unavailable']=dict(reason='CONTROLLED TEST FIXTURE',reproduce='deliberate fixture',evidence_path=str(log),evidence_sha256=ev.file_hash(log))
    source=tmp_path/'manifest.json';lock=tmp_path/'lock.json';ev.write(source,manifest);ev.freeze(source,lock)
    run=tmp_path/'run';(run/'receipts').mkdir(parents=True);(run/'log').write_bytes(log.read_bytes())
    for cell in ev.cells(manifest):
        row=dict(lock_sha256=ev.read(lock)['sha256'],cell=cell,status='unavailable',ownership=dict(manifest['ownership'],revision='FIXTURE'),elapsed_seconds=0,error='CONTROLLED TEST FIXTURE',reproduce='fixture',log=dict(path='log',sha256=ev.file_hash(run/'log')))
        ev.write(run/'receipts'/(cell['id']+'.json'),ev.seal(row))
    return manifest,lock,run


def test_matrix_exact_coverage_and_separate_calibration():
    m=ev.template();scored=list(ev.cells(m));calibration=list(ev.cells(m,'calibration'))
    assert len(scored)==288 and len({c['id'] for c in scored})==288
    assert len(calibration)==6 and {c['config_id'] for c in calibration}==set(ev.CONFIG_IDS)
    assert all(c['width']==c['height']==1024 for c in calibration)
    first=m['corpus']['cases'][0]
    assert all(c['width']==first['width'] and c['height']==first['height'] for c in calibration)
    assert all(c['phase']=='scored' for c in scored)
    assert {c['config_id'] for c in scored}==set(ev.CONFIG_IDS)


def test_model_prompt_forms_preserve_source_corpus_and_only_six_by_three_selected_inputs_change():
    m=ev.template();source=deepcopy(m['corpus']);source.pop('prompt_representation_provenance')
    narratives={c['id'] for c in source['cases'] if 'prompt_narrative_raw' in c}
    assert narratives=={'json_text-1','json_text-2','json_objects-1','json_objects-2','layout_revision-1','layout_revision-2'}
    for c in source['cases']:
        c.pop('prompt_narrative_raw',None);c.pop('prompt_narrative_enhanced',None)
    # Canonical corpus at8add90b163, before adding only the reviewed representations.
    assert ev.digest(source)=='25ccd7df5b7b30a104efcf45511059f4eb35923fd79c9beec85c01bd960e50a3'
    changed=set();rows=list(ev.cells(m));assert len(rows)==288
    for row in rows:
        case=next(c for c in m['corpus']['cases'] if c['id']==row['case_id'])
        expected_changed=row['config_id'] in ev.CONFIG_IDS[3:] and case['id'] in narratives
        assert (row['prompt']!=case['prompt_raw'])==expected_changed
        assert (row['prompt_representation']=='narrative_layout')==expected_changed
        assert row['prompt_sha256']==hashlib.sha256(row['prompt'].encode()).hexdigest()
        assert row['source_raw_prompt_sha256']==hashlib.sha256(case['prompt_raw'].encode()).hexdigest()
        assert (row['width'],row['height'])==(case['width'],case['height'])
        assert row['seed'] in m['corpus']['seeds']
        if expected_changed:changed.add((row['config_id'],row['case_id'],row['seed']))
    assert len(changed)==6*3*2
    for config in m['configs']:
        config['prompt_mode']='enhanced'
    for row in ev.cells(m):
        case=next(c for c in m['corpus']['cases'] if c['id']==row['case_id'])
        expected=case['prompt_narrative_enhanced'] if row['config_id'] in ev.CONFIG_IDS[3:] and case['id'] in narratives else case['prompt_enhanced']
        assert row['prompt']==expected and row['prompt_mode']=='enhanced'
    assert len(list(ev.cells(m,'calibration')))==6


def test_frozen_prompt_representation_rejects_altered_bytes_or_hash(unavailable_run):
    manifest,_,_=unavailable_run
    for field in ('raw','raw_sha256','source_enhanced_sha256','representation'):
        changed=deepcopy(manifest)
        changed['configs'][3]['prompt_contracts']['json_text-1'][field]='altered'
        with pytest.raises(ValueError,match='prompt representation or hash changed'):
            ev.validate_manifest(changed)


def test_reviewed_narrative_exact_bytes_literal_text_and_geometry_equivalent():
    reviewed={
      'json_text-1':'13ad6ddd8cd655adb9c0ce727b522964ba19ac2a814173ec85e7db5d916e95da',
      'json_text-2':'9537f25e5604fc9d4f39a1f020ad298480b6338935c59749343835a05cd2e924',
      'json_objects-1':'6cad2a9640e3b9503f7671b29fc9c039fcd0bc861bac23c460c15f8030c42380',
      'json_objects-2':'12ecc668dfbd16cfbdbd8969684ae86339fca21876b7128ecf96b1b11532a7c5',
      'layout_revision-1':'0b6e65d68307fe42fde465270d32a89fbafac336c320ece087c870420b1f5089',
      'layout_revision-2':'fa88e4c05fd8c3c83e3bf8ddc3b03fa98fb25534bdae66740264587a73f7258c'}
    for c in ev.corpus()['cases']:
        if c['id'] not in reviewed:continue
        prompt=c['prompt_narrative_raw']
        assert hashlib.sha256(prompt.encode()).hexdigest()==reviewed[c['id']]
        assert 'from the center of the canvas' in prompt and 'increasing downward' in prompt
        assert c['prompt_narrative_enhanced']==prompt+' Composition must make every requested element clearly visible; retain the exact requested wording, counts, colors and positions.'
        assert re.findall('“([^”]+)”',prompt)==c['expected_text']
        bounds=re.findall(r'horizontally from (-?\d+) to (-?\d+) and vertically from (-?\d+) to (-?\d+)',prompt)
        assert len(bounds)==len(c['layout_spec']['items'])==2
        for raw,item,box in zip(bounds,c['layout_spec']['items'],c['boxes']):
            x1,x2,y1,y2=map(int,raw);assert [y1,x1,y2,x2]==item['bbox']
            normalized=box['box'];pixels=[(normalized[0]-.5)*c['width'],(normalized[2]-.5)*c['width'],(normalized[1]-.5)*c['height'],(normalized[3]-.5)*c['height']]
            assert max(abs(a-b) for a,b in zip((x1,x2,y1,y2),pixels))<=.5
            if item['label']=='object':assert item['caption'].split('.')[0].lower() in prompt.lower()
        if c['id']=='json_objects-1':assert 'plain grey background' in prompt
        if c['id']=='json_objects-2':assert 'no extra objects' in prompt
        if c['scenario']=='layout_revision':assert 'fresh generation, not an edit of a source image' in prompt


def test_layout_uses_author_coordinates_and_has_independent_objective_boxes():
    for case in ev.corpus()['cases']:
        if case['boxes']:
            spec=case['layout_spec'];assert spec['bbox_coord_space']=='abs_centralized' and spec['bbox_format']=='yxyx'
            for item,box in zip(spec['items'],case['boxes']):
                x1,y1,x2,y2=box['box'];w,h=case['width'],case['height']
                assert item['bbox']==[round((y1-.5)*h),round((x1-.5)*w),round((y2-.5)*h),round((x2-.5)*w)]
    assert all(c['prior_layout_for_context_only'] for c in ev.corpus()['cases'] if c['scenario']=='layout_revision')


def test_unprepared_template_cannot_be_frozen(tmp_path):
    with pytest.raises(ValueError,match='incomplete runtime'):ev.validate_manifest(ev.template())


def test_failure_receipts_are_complete_explicit_and_not_quality_success(unavailable_run,tmp_path):
    _,lock,run=unavailable_run
    _,results=ev.receipts(lock,run);assert len(results)==288
    public=tmp_path/'pairs.json';ev.blind_pairs(lock,run,public)
    pairs=ev.unseal(ev.read(public))['pairs'];mapping=ev.unseal(ev.read(str(public)+'.private.json'))['mapping']
    assert len(pairs)==len(mapping)==432
    assert all(p['arms'][0]['image'] is None for p in pairs)
    assert all('.moss_' in p['A'] or '.moss_' in p['B'] for p in mapping)
    judgments=[dict(pair_id=p['id'],winner='unavailable',reviewer='FIXTURE',reviewer_kind='machine_assisted',judge_version='CONTROLLED FIXTURE',notes='No outputs',uncertainty='high') for p in pairs]
    ev.write(tmp_path/'judgments.json',judgments);ev.write(tmp_path/'annotations.json',{})
    report=ev.report(lock,run,public,tmp_path/'judgments.json',tmp_path/'annotations.json')
    assert report['recommendation_eligible'] is False and report['usable_blind_pairs']==0
    assert report['missing_outputs']==288 and report['dmd_adapter_prediction_parity_complete'] is False


def test_incomplete_receipts_and_altered_identity_rejected(unavailable_run):
    _,lock,run=unavailable_run
    path=next((run/'receipts').glob('*.json'));original=path.read_bytes();path.unlink()
    with pytest.raises(ValueError,match='missing cell'):ev.receipts(lock,run)
    path.write_bytes(original);row=ev.read(path);row['cell']['seed']+=1
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError,match='digest mismatch'):ev.receipts(lock,run)
    row=ev.seal({k:v for k,v in row.items() if k!='sha256'});path.write_text(json.dumps(row))
    with pytest.raises(ValueError,match='identity mismatch'):ev.receipts(lock,run)


def test_artifact_tampering_and_escape_rejected(unavailable_run):
    _,lock,run=unavailable_run
    (run/'log').write_text('tamper')
    with pytest.raises(ValueError,match='artifact hash mismatch'):ev.receipts(lock,run)
    with pytest.raises(ValueError,match='escapes'):ev.artifact(run,dict(path='../manifest.json',sha256='bad'))


def test_windows_relative_artifacts_resolve_on_posix_without_rewriting_sealed_receipts(unavailable_run):
    manifest,lock,run=unavailable_run
    (run/'logs').mkdir();(run/'log').rename(run/'logs'/'worker.log')
    before={}
    for path in (run/'receipts').glob('*.json'):
        row=ev.unseal(ev.read(path));row['log']['path']=r'logs\worker.log'
        path.write_text(json.dumps(ev.seal(row)));before[path]=path.read_bytes()
    _,results=ev.receipts(lock,run)
    assert len(results)==288 and all(r['log']['path']==r'logs\worker.log' for r in results.values())
    assert all(path.read_bytes()==body for path,body in before.items())
    cell=next(ev.cells(manifest));output=run/'outputs'/'sample.png';output.parent.mkdir();output.write_bytes(png(cell['width'],cell['height']))
    row=dict(lock_sha256=ev.read(lock)['sha256'],cell=cell,status='ok',ownership=dict(manifest['ownership'],revision='CPU FIXTURE'),elapsed_seconds=1,
        output=dict(path=r'outputs\sample.png',sha256=ev.file_hash(output)),log=dict(path=r'logs\worker.log',sha256=ev.file_hash(run/'logs'/'worker.log')),
        metrics=dict(generation_seconds=1,peak_gpu_bytes=1,peak_host_rss_bytes=1,download_bytes=0,latency_condition='cold_process',cold_seconds=1,gpu_memory_method='CPU FIXTURE ONLY',component_dtypes={'fixture':'none'},sampler={'fixture':True},runtime_versions={'fixture':True},pixels_finite=True))
    sealed=ev.seal(row);original=ev.canonical(sealed)
    assert ev.validate_receipt(run,ev.read(lock)['sha256'],cell,sealed)['output']['path']==r'outputs\sample.png'
    assert ev.canonical(sealed)==original
    output.write_bytes(b'tampered')
    with pytest.raises(ValueError,match='artifact hash mismatch'):ev.artifact(run,row['output'])


def test_portable_artifact_paths_reject_both_platforms_absolute_traversal_and_escape(tmp_path):
    (tmp_path/'safe').write_bytes(b'fixture')
    for relative in ('/etc/passwd',r'C:\outputs\safe',r'C:outputs\safe',r'\\server\share\safe',r'\outputs\safe','../safe',r'outputs\..\safe',r'outputs/..\safe','outputs/./safe','outputs//safe','safe:stream',''):
        with pytest.raises(ValueError):ev.artifact_path(tmp_path,relative)
    outside=tmp_path.parent/(tmp_path.name+'-outside');outside.write_bytes(b'fixture')
    try:
        (tmp_path/'escape').symlink_to(outside)
        with pytest.raises(ValueError,match='escapes'):ev.artifact_path(tmp_path,'escape')
    finally:outside.unlink()


def test_unicode_text_and_boxes_are_independently_scored():
    case=next(c for c in ev.corpus()['cases'] if c['id']=='json_text-1')
    annotation=dict(annotator='fixture',method='manual',tool_revision='v1',text=list(case['expected_text']),boxes={b['id']:b['box'] for b in case['boxes']})
    assert ev.objective_scores(case,annotation)==dict(text=[1,1],box=[1,1])
    annotation['text'][0]='wrong'
    assert ev.objective_scores(case,annotation)['text'][0]<.95
    annotation['boxes']['title']=None
    assert ev.objective_scores(case,annotation)['box'][0]==0
    assert ev.edit_similarity('中秋快乐','中秋快乐')==1 and ev.edit_similarity('中秋快乐','中秋快')<1
    assert ev.edit_similarity('BOOKS & MAPS','BOOKS AND MAPS')<1


def test_malformed_png_crc_pixels_and_canvas_rejected(tmp_path):
    p=tmp_path/'image.png';p.write_bytes(png(16,32));assert ev.image_dimensions(p)==(16,32)
    data=bytearray(p.read_bytes());data[-5]^=1;p.write_bytes(data)
    with pytest.raises(ValueError,match='CRC'):ev.image_dimensions(p)
    p.write_bytes(png(16,32)[:24])
    with pytest.raises(ValueError):ev.image_dimensions(p)


def test_dmd_fixed_noise_prefix_and_recurrence_discriminate_common_wrong_samplers():
    # z=2, Q=.25, velocity=.5*x+code. Fixed z gives -2 then -3.75.
    seen=[]
    result=dmd_loop(2.,lambda x:.25*x,[3.,5.,99.],lambda x,t:.5*x+t,steps=2,observer=lambda *x:seen.append(x))
    assert result==-3.75 and [s[1] for s in seen]==[2.,1.5]
    assert dmd_loop(2.,lambda x:.25*x,[3.,5.],lambda x,t:.5*x+t,steps=1)==-2
    with pytest.raises(ValueError):dmd_loop(2.,lambda x:x,[1.],lambda x,t:x,steps=2)
    with pytest.raises(ValueError):dmd_loop(2.,lambda x:x,[1.],lambda x,t:x,steps=0)


def test_unreviewed_pipeline_rejected_before_any_import(tmp_path):
    p=tmp_path/'evil.py';p.write_text('raise RuntimeError("must never execute")')
    with pytest.raises(ValueError,match='unreviewed'):import_pipeline(p)


def test_worker_pipeline_import_keeps_snapshot_bytecode_free(tmp_path,monkeypatch):
    import queue
    import reference
    import run
    source=tmp_path/'snapshot'/'pipeline_moss_image.py';source.parent.mkdir()
    source.write_text('FIXTURE_VALUE = 123\n')
    monkeypatch.setattr(reference,'PIPELINE_SHA256',ev.file_hash(source))
    monkeypatch.setattr(sys,'dont_write_bytecode',False)
    monkeypatch.setenv('PYTHONDONTWRITEBYTECODE','')
    class FixtureReference:
        def __init__(self,*args,**kwargs):
            assert sys.dont_write_bytecode and os.environ['PYTHONDONTWRITEBYTECODE']=='1'
            assert reference.import_pipeline(source).FIXTURE_VALUE==123
    monkeypatch.setattr(reference,'Reference',FixtureReference)
    requests=queue.Queue();requests.put(None);replies=queue.Queue()
    run.worker({'id':'moss_base'},requests,replies,str(tmp_path/'worker.log'),'CPU-FIXTURE')
    assert replies.get_nowait()[0]=='ready'
    assert sorted(p.relative_to(source.parent).as_posix() for p in source.parent.rglob('*'))==[source.name]


def test_native_advanced_request_and_precision_are_explicit():
    m=ev.template();config=m['configs'][3];cell=next(c for c in ev.cells(m) if c['config_id']==config['id'])
    body=request_body(config,cell,'fixture-project')
    assert body['count']==1 and body['model']=='qwen_image_2_1'
    assert 'steps' not in body and body['advanced']['steps']==40
    assert body['advanced']['mlxQuantize']==4 and body['advanced']['mlxQuantizeExplicit'] is True
    assert body['advanced']['enhancePrompt'] is False


def test_published_settings_cannot_silently_change(unavailable_run):
    manifest,_,_=unavailable_run
    manifest['configs'][2]['guidance']=3
    with pytest.raises(ValueError,match='published model settings'):ev.validate_manifest(manifest)


def test_failed_judgment_cannot_claim_winner(unavailable_run,tmp_path):
    _,lock,run=unavailable_run;public=tmp_path/'pairs.json';ev.blind_pairs(lock,run,public)
    pairs=ev.unseal(ev.read(public))['pairs']
    judgments=[dict(pair_id=p['id'],winner='unavailable',reviewer='FIXTURE',reviewer_kind='machine_assisted',judge_version='CONTROLLED FIXTURE',notes='No outputs',uncertainty='high') for p in pairs]
    judgments[0]['winner']='A';ev.write(tmp_path/'j.json',judgments);ev.write(tmp_path/'a.json',{})
    with pytest.raises(ValueError,match='must be unavailable'):ev.report(lock,run,public,tmp_path/'j.json',tmp_path/'a.json')


def test_blind_pair_substitution_rejected_even_if_resealed(unavailable_run,tmp_path):
    _,lock,run=unavailable_run;public=tmp_path/'pairs.json';ev.blind_pairs(lock,run,public)
    private=ev.unseal(ev.read(str(public)+'.private.json'));private['mapping'][0]['A']=private['mapping'][1]['A'];private['mapping'][0]['B']=private['mapping'][1]['B']
    Path(str(public)+'.private.json').write_text(json.dumps(ev.seal(private)))
    with pytest.raises(ValueError,match='missing or duplicated'):ev.report(lock,run,public,tmp_path/'nonexistent',tmp_path/'nonexistent')


def test_true_unmerged_fp32_adapter_and_both_modulation_replacements():
    torch=pytest.importorskip('torch',reason='actual adapter tensor test requires prepared reference environment')
    from torch import nn
    from reference import install_adapter, PROJECTIONS
    class Layer(nn.Module):
        def __init__(self):
            super().__init__();self.proj=nn.ModuleDict({name:nn.Linear(128,128,bias=False,dtype=torch.bfloat16) for name in PROJECTIONS})
    class Transformer(nn.Module):
        def __init__(self):
            super().__init__();self.layers=nn.ModuleList([Layer()]);self.time_embedding=nn.Sequential(nn.Linear(128,128),nn.SiLU(),nn.Linear(128,128));self.shared_block_modulation=nn.Linear(128,128);self.video_head_modulation=nn.Linear(128,128)
    model=Transformer();lora={}
    for name in PROJECTIONS:
        prefix='diffusion_model.layers.0.proj.'+name
        with torch.no_grad():model.layers[0].proj[name].weight.copy_(torch.eye(128))
        lora[prefix+'.lora_down.weight']=torch.eye(128)
        lora[prefix+'.lora_up.weight']=torch.eye(128)*.0001
        lora[prefix+'.alpha']=torch.tensor(128.)
    replacements={k:torch.ones_like(v,dtype=torch.float32)*.25 for k,v in model.named_parameters() if k.startswith('time_embedding.') or k in ('shared_block_modulation.bias','video_head_modulation.bias')}
    with pytest.raises(ValueError,match='both modulation'):
        install_adapter(model,lora,{k:v for k,v in replacements.items() if k!='video_head_modulation.bias'})
    install_adapter(model,lora,replacements)
    x=torch.ones(1,128,dtype=torch.bfloat16)
    actual=model.layers[0].proj['q'](x)
    assert actual.dtype==torch.float32 and torch.allclose(actual,torch.ones(1,128)*1.0001,rtol=0,atol=1e-7)
    assert torch.equal(model.layers[0].proj['q'].weight,torch.eye(128,dtype=torch.bfloat16))
    assert model.shared_block_modulation.bias.dtype==model.video_head_modulation.bias.dtype==torch.float32

    if os.environ.get('MOSS_CPU_FIXTURE_RECEIPT'):
        ev.write(os.environ['MOSS_CPU_FIXTURE_RECEIPT'],dict(kind='CPU numerical fixture; no model weights or image quality evidence',torch_version=torch.__version__,python=sys.version,device=str(actual.device),test_sha256=ev.file_hash(__file__),runtime_sha256=ev.file_hash(ROOT/'reference.py'),output_dtype=str(actual.dtype),observed_first_value=float(actual.detach()[0,0]),expected_first_value=1.0001,backbone_unchanged=bool(torch.equal(model.layers[0].proj['q'].weight,torch.eye(128,dtype=torch.bfloat16))),shared_modulation_dtype=str(model.shared_block_modulation.bias.dtype),head_modulation_dtype=str(model.video_head_modulation.bias.dtype)))


def test_native_initialization_failure_closes_owned_services(monkeypatch):
    import native
    import native_services
    from contextlib import contextmanager
    closed=[]
    @contextmanager
    def fake_services(settings):
        try:yield {'worker_pid':123}
        finally:closed.append(True)
    monkeypatch.setattr(native_services,'services',fake_services)
    def fail(self):raise ValueError('controlled connection failure')
    monkeypatch.setattr(native.Native,'_connect',fail)
    with pytest.raises(ValueError,match='controlled connection'):native.Native({'settings':{}})
    assert closed==[True]



def test_native_uuid_maps_physical_telemetry_without_changing_logical_cuda_zero():
    from native_services import physical_index
    assert physical_index('GPU-second','0, GPU-first\n1, GPU-second\n')=='1'
    with pytest.raises(ValueError):physical_index('GPU-missing','0, GPU-first\n1, GPU-second\n')
    with pytest.raises(ValueError):physical_index('GPU-second','0, GPU-second\n1, GPU-second\n')


def test_native_supervised_services_hold_stdin_pipe_and_capture_named_exit(tmp_path,monkeypatch):
    import io
    import native_services as ns
    import ownership
    from types import SimpleNamespace
    binary=tmp_path/'api.exe';binary.write_bytes(b'CPU fixture, never executed')
    hub=tmp_path/'hub';hub.mkdir();children=[]
    class Process:
        def __init__(self,args,**kw):
            assert kw['stdin']==ns.subprocess.PIPE
            self.stdin=io.BytesIO();self.pid=100+len(children);self.code=None;self.worker=kw['env'].get('SCENEWORKS_WORKER_ONLY')=='1'
            kw['stdout'].write(('worker startup fixture\n' if self.worker else 'API ready fixture\n').encode());kw['stdout'].flush();children.append(self)
        def poll(self):return 7 if self.worker else self.code
        def terminate(self):self.code=-15
        def wait(self,**kw):return self.poll()
    monkeypatch.setattr(ns.subprocess,'Popen',Process);monkeypatch.setattr(ns,'physical_index',lambda *a:'1')
    settings=dict(state_root=str(tmp_path/'state'),binary=str(binary),binary_sha256=ev.file_hash(binary),hub_cache=str(hub),device_id=ownership.RUNNER_GPU_UUID,port=17931,worker_id='fixture')
    with pytest.raises(RuntimeError,match='owned service exited:') as error:
        with ns.services(settings):pytest.fail('failed worker registered')
    evidence=json.loads(str(error.value).split('owned service exited: ',1)[1])
    assert [(r['service'],r['returncode']) for r in evidence['services']]==[('api',None),('worker',7)]
    assert evidence['services'][1]['returncode_hex']=='0x00000007'
    assert 'worker startup fixture' in evidence['services'][1]['tail_utf8']
    assert all(child.stdin.closed for child in children)


@pytest.fixture
def carry_fixture(tmp_path):
    """Synthetic startup failures and images; no native executable or tensor load."""
    import bootstrap as b
    import ownership as o
    import calibration_source as cs
    import shutil
    base=tmp_path/'owned';base.mkdir()
    runtime=dict(python='CPU fixture',packages={'fixture':'1'},os='CPU fixture',cuda='fixture')
    def make_campaign(name,old=False):
        root=base/name;root.mkdir();harness=root/'harness';harness.mkdir()
        for path in ROOT.iterdir():
            if path.suffix=='.py' or path.name in ('corpus.json','dependency-resolution.json'):shutil.copyfile(path,harness/path.name)
        if old:
            path=harness/'native_services.py';path.write_text(path.read_text().replace('stdin=subprocess.PIPE,',''))
        staged={}
        for key,(repo,rev) in o.REPOS.items():
            snap=root/'hub'/('models--'+repo.replace('/','--'))/'snapshots'/rev;snap.mkdir(parents=True)
            file=snap/'tiny.safetensors';file.write_bytes(b'CPU fixture bytes, not model tensors')
            refs=snap.parent.parent/'refs';refs.mkdir();(refs/'main').write_text(rev)
            meta=root/'metadata'/(key+'.json');meta.parent.mkdir(exist_ok=True);meta.write_text('{}')
            staged[key]=dict(repo=repo,revision=rev,snapshot=str(snap),metadata=str(meta),metadata_sha256=ev.file_hash(meta),files=[dict(path=str(file),sha256=ev.file_hash(file),bytes=file.stat().st_size)])
        binary=root/'target'/'release'/'sceneworks-rust-api.exe';binary.parent.mkdir(parents=True);binary.write_bytes(b'CPU binary fixture')
        prep=dict(sceneworks_source=b.SOURCE,inference=b.INFERENCE,staged=staged,runtime=runtime,binary=str(binary),binary_sha256=ev.file_hash(binary),harness={p.name:ev.file_hash(p) for p in harness.iterdir()})
        (root/'requirements.lock').write_text('CPU fixture closure');prep['requirements_sha256']=ev.file_hash(root/'requirements.lock');ev.write(root/'prepared.json',prep)
        m=ev.template();m['runtime']=dict(runtime,driver='fixture-driver');m['hardware']={k:'CPU fixture' for k in m['hardware']}
        m['hardware'].update(host=__import__('socket').gethostname(),gpu_uuid=o.RUNNER_GPU_UUID)
        m['ownership']=dict(responsible_chat='CPU coordinator',lease='lease',job='fixture',timeout_seconds=30,resume='fixture',next_action='fixture')
        m['preparation']=dict(path=str(root/'prepared.json'),sha256=ev.file_hash(root/'prepared.json'))
        shared=staged['moss_sft']['snapshot']
        for c in m['configs']:
            c['snapshot']=staged[c['id']]['snapshot'];c['runtime_command']=[str(root/'venv'/'python'),str(harness/'run.py')]
            c['settings']=dict(shared_snapshot=shared,pipeline_code=str(Path(shared)/'pipeline_moss_image.py')) if c['id'] in cs.MOSS else dict(binary=str(binary),binary_sha256=prep['binary_sha256'],hub_cache=str(root/'hub'),device_id='CPU fixture')
            if c['id']=='moss_dmd':c['settings'].update(adapter_validation=True,lora_path=str(Path(c['snapshot'])/'lora.safetensors'),replacement_path=str(Path(c['snapshot'])/'time_embedding.safetensors'))
            paths=set(harness.glob('*.py'))|set(Path(c['snapshot']).rglob('*'))
            if c['id'] in cs.MOSS:paths.update(Path(shared).rglob('*'))
            c['files']=[dict(path=str(p),sha256=ev.file_hash(p),bytes=p.stat().st_size) for p in sorted(paths) if p.is_file()]
        return root,m,prep
    source,old,prep=make_campaign('sc-26322-original',old=True);ev.write(source/'protocol.lock.json',ev.seal(old));sha=ev.read(source/'protocol.lock.json')['sha256']
    run=source/'run';(run/'receipts').mkdir(parents=True);(run/'outputs').mkdir();(run/'logs').mkdir();(run/'parity').mkdir()
    evidence=parity_evidence();dmdlog=run/'logs'/'moss_dmd.log';dmdlog.write_text('CPU fixture parity, not actual model evidence')
    fixture=ev.parity_cells(old,'calibration')[0];parity=ev.seal(dict(lock_sha256=sha,cell=fixture,status='ok',evidence=evidence,log=dict(path='logs/moss_dmd.log',sha256=ev.file_hash(dmdlog))))
    ev.write(run/'parity'/(fixture['id']+'.json'),parity)
    metrics=dict(generation_seconds=.1,peak_gpu_bytes=42,peak_host_rss_bytes=42,download_bytes=0,latency_condition='cold_process',cold_seconds=.2,warm_seconds=None,gpu_memory_method='CPU fixture',component_dtypes={'fixture':'none'},sampler={'fixture':True},runtime_versions={'fixture':'1'},pixels_finite=True)
    for cell in ev.cells(old,'calibration'):
        log=run/'logs'/(cell['config_id']+'.log')
        if not log.exists():log.write_text('CPU fixture startup failure, not observed native availability')
        row=dict(lock_sha256=sha,cell=cell,ownership=dict(old['ownership'],revision=sha),elapsed_seconds=.1,log=dict(path='logs/'+log.name,sha256=ev.file_hash(log)))
        if cell['config_id'] in cs.MOSS:
            out=run/'outputs'/(cell['id']+'.png');out.write_bytes(png(cell['width'],cell['height']))
            metric=deepcopy(metrics)
            if cell['config_id']=='moss_dmd':metric.update(parity_fixture_digests=[parity['sha256']],noise_sha256=evidence['noise_sha256'],sampler=dict(Q_sha256=evidence['Q_sha256'],step_codes=evidence['step_codes'],execution_arm='sft_bf16_unmerged_fp32_adapter',oracle_resident=False))
            row.update(status='ok',attempted=True,output=dict(path='outputs/'+out.name,sha256=ev.file_hash(out)),metrics=metric)
        else:row.update(status='failed',failure_kind='runtime',attempted=True,error='RuntimeError: owned service exited; inspect logs',reproduce='CPU fixture only',output=None,metrics=None)
        ev.write(run/'receipts'/(cell['id']+'.json'),ev.seal(row))
    job=source/'job-123-1-calibration.json';ev.write(job,dict(mode='calibration',status='completed',job='123-1',root=str(source),sceneworks_source=b.SOURCE,inference=b.INFERENCE,transport_sha256=prep['harness']['bootstrap.py'],accelerator_execution_requested=True))
    root,m,current=make_campaign('sc-26322-repaired')
    request=dict(campaign=source.name,protocol_sha256=sha,prepared_sha256=ev.file_hash(source/'prepared.json'),job='123-1',job_receipt_sha256=ev.file_hash(job),calibration_sha256=o.calibration_bundle(source/'protocol.lock.json',run),authorization=dict(id='CPU diagnosed repair',coordinator='CPU coordinator',reason='CPU fixture stdin lifecycle repair',native_services_sha256=ev.file_hash(ROOT/'native_services.py'),new_harness_sha256=ev.digest(current['harness']),remaining_calibration_images=3))
    return dict(source=source,root=root,old=old,manifest=m,request=request,metrics=metrics,make_campaign=make_campaign)


def test_carried_calibration_preserves_original_identity_and_only_skips_calibration(carry_fixture,monkeypatch):
    import calibration_source as cs
    import ownership as o
    import run
    f=carry_fixture;m=f['manifest'];root=f['root'];source=f['source']
    original={p:p.read_bytes() for p in source.rglob('*') if p.is_file()}
    cs.install(root,m,f['request']);ev.write(root/'protocol.lock.json',ev.seal(m))
    _,rows=ev.receipts(root/'protocol.lock.json',root/'run','calibration',complete=False)
    assert len(rows)==3 and {r['lock_sha256'] for r in rows.values()}=={f['request']['protocol_sha256']}
    assert len(cs.carried(m,root/'run')[1])==1
    assert len(ev.parity_evidence(m,root/'run',ev.read(root/'protocol.lock.json')['sha256']))==1
    assert all(path.read_bytes()==body for path,body in original.items())
    for config in cs.MOSS:
        assert run.existing_successes(m,ev.read(root/'protocol.lock.json')['sha256'],root/'run',config,'calibration')==[]
        assert len(run.existing_successes(m,ev.read(root/'protocol.lock.json')['sha256'],root/'run',config,'scored'))==48
    class Guard:
        def __init__(self,*a,**kw):self.prepared_root=root;self.unavailable_configs={}
        def check(self,*a,**kw):cs.check_carry_closure(m,root/'run',execution=True)
    monkeypatch.setattr(o,'ExecutionGuard',Guard);monkeypatch.setattr(run,'validate_manifest',lambda *a,**kw:None)
    monkeypatch.setattr(run,'launch',lambda *a,**kw:pytest.fail('carried MOSS calibration spawned'))
    for config in cs.MOSS:assert run.execute(root/'protocol.lock.json',root/'run',config,'calibration')==0
    native=next(c for c in ev.cells(m,'calibration') if c['config_id']=='qwen_image_2_1')
    assert len(run.existing_successes(m,'unused',root/'run',native['config_id'],'calibration'))==1
    assert not (root/'run'/'receipts').exists() and not (root/'run'/'outputs').exists()


@pytest.mark.parametrize('change',['runtime','driver','recipe','weights','worker','global','guard','sampler','evaluation_dependency','invalidated','status'])
def test_calibration_carry_rejects_incompatible_or_forged_source(carry_fixture,change):
    import calibration_source as cs
    f=carry_fixture;m=f['manifest'];root=f['root'];source=f['source'];request=f['request']
    if change=='runtime':m['runtime']['packages']={'different':'1'}
    elif change=='driver':m['runtime']['driver']='different'
    elif change=='recipe':m['configs'][0]['steps']+=1
    elif change=='weights':next(x for x in m['configs'][0]['files'] if x['path'].endswith('tiny.safetensors'))['sha256']='0'*64
    elif change in ('worker','global','guard','sampler','evaluation_dependency'):
        name='run.py' if change in ('worker','global') else 'ownership.py' if change=='guard' else 'evaluate.py' if change=='evaluation_dependency' else 'reference.py'
        path=root/'harness'/name;text=path.read_text()
        text=text.replace('runtime=Reference(config,mode=mode)','runtime=Reference(config,mode="wrong")') if change=='worker' else text+'\nUNREVIEWED_GLOBAL = 1\n' if change=='global' else text.replace("require(self.now() < self.deadline_epoch, 'device lease expired or execution deadline reached')","pass") if change=='guard' else text.replace('return hashlib.sha256(canonical(value)).hexdigest()',"return 'unreviewed dependency'") if change=='evaluation_dependency' else text+'\n# changed sampler bytes\n'
        path.write_text(text);p=ev.read(root/'prepared.json');p['harness'][name]=ev.file_hash(path);(root/'prepared.json').write_text(json.dumps(p));m['preparation']['sha256']=ev.file_hash(root/'prepared.json');request['authorization']['new_harness_sha256']=ev.digest(p['harness'])
    elif change=='invalidated':ev.write(source/'run'/'invalidations'/'fixture.json',ev.seal(dict(lock_sha256=request['protocol_sha256'],cell_ids=[],parity_cell_ids=[],reason='CPU input invalidation')))
    else:
        path=next((source/'run'/'receipts').glob('calibration.moss_base*.json'));row=ev.unseal(ev.read(path));row.update(status='failed',error='forged failure',reproduce='CPU fixture');path.write_text(json.dumps(ev.seal(row)))
    with pytest.raises(ValueError):cs.install(root,m,request)


@pytest.mark.parametrize('change',['validation','duplicate','hash','missing','duplicate_helper'])
def test_calibration_carry_rejects_parity_helper_validation_mutations(carry_fixture,change):
    import calibration_source as cs
    f=carry_fixture;root=f['root'];m=f['manifest'];request=f['request'];path=root/'harness'/'evaluate.py'
    text=path.read_text()
    mutations={
        'validation':("row=validate_parity(root,lock_hash,doc['cell'],doc)","row=unseal(doc)"),
        'duplicate':("require(row['cell']['id'] not in {r['cell']['id'] for r in result.values()},'duplicate controlled parity fixture')","pass"),
        'hash':("result[doc['sha256']]=row","result['0'*64]=row"),
    }
    if change in mutations:
        before,after=mutations[change];assert before in text;text=text.replace(before,after)
    else:
        start=text.index('def parity_evidence(');end=text.index('def receipts(',start);helper=text[start:end]
        text=text[:start]+('' if change=='missing' else helper+helper)+text[end:]
    path.write_text(text)
    prep=ev.read(root/'prepared.json');prep['harness']['evaluate.py']=ev.file_hash(path);(root/'prepared.json').write_text(json.dumps(prep))
    m['preparation']['sha256']=ev.file_hash(root/'prepared.json');request['authorization']['new_harness_sha256']=ev.digest(prep['harness'])
    with pytest.raises(ValueError,match='parity evidence helper (changed|missing or duplicated)'):cs.install(root,m,request)


@pytest.mark.parametrize('change',['original_bytes','extra_file','new_moss_image','new_moss_receipt','claim'])
def test_carried_archive_and_new_image_budget_remain_strict(carry_fixture,change):
    import calibration_source as cs
    f=carry_fixture;root=f['root'];m=f['manifest'];cs.install(root,m,f['request'])
    archive=cs.source_root(root/'run')
    if change=='original_bytes':next((archive/'run'/'outputs').glob('*.png')).write_bytes(b'tamper')
    elif change=='extra_file':(archive/'injected.pyc').write_bytes(b'extra')
    elif change=='claim':Path(m['calibration_carry_forward']['claim']['path']).write_text('{}')
    else:
        cell=next(ev.cells(m,'calibration'));path=root/'run'/('outputs' if change=='new_moss_image' else 'receipts')/(cell['id']+('.png' if change=='new_moss_image' else '.json'));path.parent.mkdir();path.write_bytes(b'forbidden duplicate')
    with pytest.raises(ValueError):
        if change=='claim':cs.check_carry_closure(m,root/'run',execution=True)
        else:cs.carried(m,root/'run')


def test_calibration_repair_claim_forbids_second_campaign(carry_fixture):
    import calibration_source as cs
    f=carry_fixture;cs.install(f['root'],f['manifest'],f['request'])
    other,m,_=f['make_campaign']('sc-26322-another')
    with pytest.raises(ValueError,match='refusing to overwrite'):cs.install(other,m,f['request'])


def test_repaired_transport_generates_only_three_native_calibrations(carry_fixture,monkeypatch):
    import bootstrap as b
    import ownership as o
    import calibration_source as cs
    import run
    from types import SimpleNamespace
    f=carry_fixture;root=f['root'];m=f['manifest'];cs.install(root,m,f['request']);ev.write(root/'protocol.lock.json',ev.seal(m))
    calls=[]
    class Guard:
        def __init__(self,*a,**kw):self.prepared_root=root;self.unavailable_configs={}
        def check(self,*a,**kw):cs.check_carry_closure(m,root/'run',execution=True);return {'CPU_fixture':True}
    monkeypatch.setattr(b,'ExecutionGuard',Guard);monkeypatch.setattr(o,'ExecutionGuard',Guard)
    monkeypatch.setattr(ev,'validate_manifest',lambda manifest,**kw:manifest);monkeypatch.setattr(run,'validate_manifest',lambda *a,**kw:None)
    class Requests:
        pending=None
        def put(self,item):self.pending=item
    def launch(config,*a,**kw):
        assert config['id'] not in cs.MOSS;request=Requests()
        def replies():
            yield ('ready',.01)
            cell,path=request.pending;Path(path).write_bytes(png(cell['width'],cell['height']))
            yield ('result',dict(f['metrics'],load_seconds=.01))
        return SimpleNamespace(),request,replies()
    monkeypatch.setattr(run,'launch',launch);monkeypatch.setattr(run,'receive',lambda p,q,*a:(next(q),42));monkeypatch.setattr(run,'finish',lambda *a:None)
    class Process:
        returncode=0
        def __init__(self,args,**kw):
            config=args[args.index('--config')+1];calls.append(config);run.execute(root/'protocol.lock.json',root/'run',config,'calibration')
        def poll(self):return 0
    monkeypatch.setattr(b.subprocess,'Popen',Process);monkeypatch.setattr(b,'kill_owned',lambda *a:None)
    deadline='2030-01-01T00:00:00+00:00'
    lease=dict(id=m['ownership']['lease'],owner=m['ownership']['responsible_chat'],host=m['hardware']['host'],gpu_uuid=o.RUNNER_GPU_UUID,expires_utc=deadline,external_owners_reconciled=True,desktop_baseline=[],baseline_max_memory_mib=1000,baseline_max_utilization_percent=2)
    b.execute_phase(root,dict(protocol_sha256=ev.read(root/'protocol.lock.json')['sha256'],deadline_utc=deadline,lease=lease),'calibration')
    assert calls==list(ev.CONFIG_IDS[3:]) and len(list((root/'run'/'outputs').glob('*.png')))==3
    _,rows=ev.receipts(root/'protocol.lock.json',root/'run','calibration');assert len(rows)==6 and all(r['status']=='ok' for r in rows.values())
    assert len({r['lock_sha256'] for r in rows.values()})==2
    estimate=ev.read(root/'calibration-estimate.json')
    assert estimate['calibration_outputs']==6 and estimate['calibration_attempts']==9 and estimate['historical_failed_native_attempts']==3
    assert all(estimate['conditions'][key]['measurement_origin']=='carried_original' for key in cs.MOSS)
    assert all(estimate['conditions'][key]['measurement_origin']=='current_campaign' for key in ev.CONFIG_IDS[3:])
    # Carry blocks calibration only: actual scored execution remains reachable.
    cache={}
    def scored_launch(config,*a,**kw):
        assert config['id']=='moss_base';request=Requests()
        def replies():
            yield ('ready',.01)
            while True:
                cell,path=request.pending;shape=(cell['width'],cell['height'])
                if shape not in cache:cache[shape]=png(*shape)
                Path(path).write_bytes(cache[shape])
                yield ('result',dict(f['metrics'],peak_gpu_reserved_bytes=42,load_seconds=.01))
        return SimpleNamespace(),request,replies()
    monkeypatch.setattr(run,'launch',scored_launch)
    run.execute(root/'protocol.lock.json',root/'run','moss_base','scored')
    _,scored=ev.receipts(root/'protocol.lock.json',root/'run','scored',complete=False)
    assert len(scored)==48 and all(r['status']=='ok' for r in scored.values())
    assert len(list((root/'run'/'outputs').glob('calibration.*.png')))==3



def test_offline_review_reader_validates_original_archive_without_execution_inputs(carry_fixture):
    import subprocess
    import calibration_source as cs
    f=carry_fixture;cs.install(f['root'],f['manifest'],f['request']);lock=f['root']/'protocol.lock.json';ev.write(lock,ev.seal(f['manifest']))
    review=ROOT.parent/'moss_review'/'review.py'
    code=f"import importlib.util; from pathlib import Path; s=importlib.util.spec_from_file_location('review',{str(review)!r}); r=importlib.util.module_from_spec(s); s.loader.exec_module(r); r.ev.validate_manifest=lambda *a,**k: (_ for _ in ()).throw(AssertionError('offline execution validation forbidden')); m,rows=r.ev.receipts(Path({str(lock)!r}),Path({str(f['root']/'run')!r}),'calibration',complete=False); assert len(rows)==3"
    subprocess.run([sys.executable,'-I','-c',code],check=True,capture_output=True,text=True)


def test_scored_parity_reuses_original_fixture_and_executes_only_other_four(carry_fixture,monkeypatch):
    import calibration_source as cs
    import run
    from types import SimpleNamespace
    f=carry_fixture;root=f['root'];m=f['manifest'];cs.install(root,m,f['request']);sha=ev.digest(m)
    requested=[]
    class Guard:
        def check(self,*a,**kw):cs.check_carry_closure(m,root/'run',execution=True)
    class Requests:
        def put(self,item):requested.append(item[0]['id'])
    replies=iter([('ready',.01)]+[('result',parity_evidence()) for _ in range(4)])
    monkeypatch.setattr(run,'launch',lambda *a,**kw:(SimpleNamespace(),Requests(),replies));monkeypatch.setattr(run,'receive',lambda p,q,*a:(next(q),42));monkeypatch.setattr(run,'finish',lambda *a:None)
    rows=run.ensure_parity(m,sha,root/'run','scored',Guard(),'CPU-parity')
    assert len(rows)==5 and len(requested)==4 and 'english_signage-1' not in '|'.join(requested)
    assert sum(row['evidence']['forward_count'] for row in rows)==80
    assert len(list((root/'run'/'parity').glob('*.json')))==4
    assert len(ev.parity_evidence(m,root/'run',sha))==5
    assert not (root/'run'/'outputs').exists()


def test_live_guard_monitors_carried_artifacts_with_existing_input_fingerprints(carry_fixture,monkeypatch):
    import calibration_source as cs
    import ownership as o
    from types import SimpleNamespace
    f=carry_fixture;root=f['root'];m=f['manifest'];archive=cs.install(root,m,f['request'])
    for config in m['configs']:
        config['settings']['calibration_source_root']=str(archive)
        config['files'] += [dict(path=str(p),sha256=ev.file_hash(p),bytes=p.stat().st_size) for p in archive.rglob('*') if p.is_file()]
    sha=ev.digest(m)
    class Process:
        def __init__(self,pid=100):self.pid=pid
        def create_time(self):return 1000
        def parents(self):return [SimpleNamespace(pid=987)]
        def children(self,recursive=True):return []
    monkeypatch.setitem(sys.modules,'psutil',SimpleNamespace(Process=Process))
    monkeypatch.setattr(o,'gpu_rows',lambda:[dict(index='1',uuid=o.RUNNER_GPU_UUID,used_mib='0',utilization_percent='0')]);monkeypatch.setattr(o,'compute_pids',lambda uuid:set())
    lease=dict(lease_id=m['ownership']['lease'],responsible_chat=m['ownership']['responsible_chat'],host=m['hardware']['host'],gpu_uuid=o.RUNNER_GPU_UUID,protocol_sha256=sha,owner_pid=987,phase='calibration',deadline_epoch=5000,lease_expires_epoch=5000,prepared_root=str(root),prepared_sha=m['preparation']['sha256'],external_owners_reconciled=True,desktop_baseline=[],baseline_max_memory_mib=1000,baseline_max_utilization_percent=2)
    lock=root.parent/'device-locks'/(o.RUNNER_GPU_UUID+'.lock');ev.write(lock,lease)
    guard=o.ExecutionGuard(m,sha,environ=dict(MOSS_DEVICE_LOCK=str(lock),MOSS_LEASE_ID=m['ownership']['lease'],MOSS_PROTOCOL_SHA256=sha),now=lambda:1001)
    guard.check()
    next((archive/'run'/'outputs').glob('*.png')).write_bytes(b'tampered carried image')
    with pytest.raises(ValueError,match='frozen input metadata changed'):guard.check()


@pytest.mark.parametrize('reason',['device lease expired','foreign process appeared'])
def test_live_receive_refuses_expiry_or_foreign_owner_during_cell(monkeypatch,reason):
    import queue
    import run
    from types import SimpleNamespace
    clock=[0.]
    monkeypatch.setattr(run.time,'time',lambda:clock[0])
    monkeypatch.setattr(run.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(run,'process_rss',lambda pid:42)
    class Guard:
        deadline_epoch=1.5 if 'expired' in reason else 20.
        def check(self):
            if clock[0]>=1.5:raise ValueError(reason)
    class Replies:
        def get(self,timeout):clock[0]+=1.;raise queue.Empty
    process=SimpleNamespace(pid=42,is_alive=lambda:True)
    with pytest.raises(ValueError,match=reason):run.receive(process,Replies(),100,Guard())
    assert clock[0]<=2.  # did not run to the much larger per-cell timeout


def test_native_cache_mutations_refuse_direct_execution_before_spawn(unavailable_run,monkeypatch,tmp_path):
    import socket
    import types
    import run
    torch=pytest.importorskip('torch')
    manifest,_,_=unavailable_run
    native=manifest['configs'][3];native['unavailable']=None
    hub=tmp_path/'hub';repo=hub/('models--'+native['repo'].replace('/','--'));snapshot=repo/'snapshots'/native['revision']
    snapshot.mkdir(parents=True);(snapshot/'weights.safetensors').write_bytes(b'CONTROLLED NOT REAL WEIGHTS')
    (repo/'refs').mkdir();ref=repo/'refs'/'main';ref.write_text(native['revision'])
    binary=tmp_path/'fixture-native';binary.write_bytes(b'CONTROLLED NOT EXECUTABLE')
    native['snapshot']=str(snapshot);native['settings']=dict(hub_cache=str(hub),binary=str(binary),device_id='GPU-FIXTURE')
    manifest['runtime'].update(python=sys.version,packages={'torch':torch.__version__});manifest['hardware'].update(host=socket.gethostname(),gpu_uuid='GPU-FIXTURE')
    native['files']=ev.inventory(native)
    source=tmp_path/'cache-manifest.json';lock=tmp_path/'cache-lock.json';ev.write(source,manifest);ev.freeze(source,lock)
    calls=[]
    class Guard:
        def __init__(self,*a,**k):calls.append('guard')
        def check(self,*a,**k):pass
    monkeypatch.setitem(sys.modules,'ownership',types.SimpleNamespace(ExecutionGuard=Guard))
    monkeypatch.setattr(run,'launch',lambda *a,**k:pytest.fail('cache mutation reached runtime spawn'))
    ref.write_text('b'*40)
    with pytest.raises(ValueError,match='ref identity'):run.execute(lock,tmp_path/'run-ref',native['id'],'calibration')
    ref.write_text(native['revision']);(snapshot.parent/('b'*40)).mkdir()
    with pytest.raises(ValueError,match='alternative native resolver snapshot'):run.execute(lock,tmp_path/'run-alt',native['id'],'scored')
    assert calls==['guard','guard']


def test_dmd_loader_routes_to_bf16_sft_candidate_and_sampling_uses_that_arm(monkeypatch):
    import reference
    from types import SimpleNamespace
    torch=pytest.importorskip('torch')
    loaded=[]
    class Candidate(torch.nn.Module):
        def forward(self,x,code,embeds,mask):return SimpleNamespace(sample=x*.5+code)
    class Loader:
        @staticmethod
        def from_pretrained(root,**kwargs):loaded.append((root,kwargs));return Candidate()
    config=ev.template()['configs'][2];config['snapshot']='FP32_ORACLE_ROOT';config['settings']=dict(shared_snapshot='BF16_SFT_ROOT',adapter_validation=True,lora_path='LORA',replacement_path='COUPLED')
    installed=[]
    def install(model,lora,replacements):installed.append((lora,replacements));return model
    monkeypatch.setattr(reference,'install_adapter',install)
    candidate=reference.candidate_transformer(SimpleNamespace(MOSSImageTransformer=Loader),config,lambda path:path)
    assert loaded[0][0]=='BF16_SFT_ROOT' and loaded[0][1]['torch_dtype']==torch.bfloat16
    assert loaded[0][1]['local_files_only'] is True and loaded[0][1]['use_safetensors'] is True
    assert installed==[('LORA','COUPLED')]
    z=torch.full((1,1,1,1),2.)
    actual=reference.candidate_latents(candidate,z,torch.tensor([[.25]]),torch.tensor([3.,5.]),None,None,2)
    assert actual.item()==-3.75
    # A distinct oracle velocity would produce a different image latent; it is never a sampler argument.
    assert actual.item()!=reference.dmd_loop(2.,lambda x:.25*x,[3.,5.],lambda x,t:.75*x+t,2)


def test_five_controlled_parity_fixtures_reuse_calibration_without_more_images():
    manifest=ev.template();fixtures=ev.parity_cells(manifest);cal=ev.parity_cells(manifest,'calibration')
    assert len(fixtures)==5 and cal==fixtures[:1] and sum(8*2 for _ in fixtures)==80
    assert {c['scenario'] for c in fixtures}=={'english_signage','chinese_mixed','json_text','product_aspect'}
    assert any(c['width']>c['height'] for c in fixtures) and any(c['width']<c['height'] for c in fixtures)
    assert manifest['configs'][2]['precision']=='bf16+fp32-unmerged'


def test_invalidated_success_cannot_be_resumed(unavailable_run):
    manifest,lock,root=unavailable_run;cell=next(ev.cells(manifest))
    (root/'invalidations').mkdir()
    ev.write(root/'invalidations'/'fixture.json',ev.seal(dict(lock_sha256=ev.read(lock)['sha256'],cell_ids=[cell['id']],reason='CONTROLLED mutated input')))
    with pytest.raises(ValueError,match='invalidated'):ev.receipts(lock,root)


def parity_evidence():
    return dict(candidate_arm='sft_bf16_unmerged_fp32_adapter',oracle_arm='published_full_fp32_transformer',generated_images=0,forward_count=16,oracle_autocast=False,noise_sha256='a'*64,Q_sha256='b'*64,context_sha256='c'*64,mask_sha256='d'*64,step_codes=list(range(8)),elapsed_seconds=.2,initialization_seconds=.1,total_process_seconds=.3,
        steps=[dict(code=i,candidate_finite=True,input_sha256='e'*64,candidate_sha256='f'*64,oracle_sha256='0'*64,max_abs=.001,mean_abs=.0001,reference_rms=.5) for i in range(8)])


def test_parity_process_stops_before_candidate_and_fixture_reuses_without_forwards(monkeypatch,tmp_path):
    import run
    from types import SimpleNamespace
    manifest=ev.template();manifest['ownership']['timeout_seconds']=10
    (tmp_path/'invalidations').mkdir();events=[];alive=[False]
    class Guard:
        def check(self):pass
    def launch(config,m,log,guard,mode='candidate'):
        assert not alive[0], 'oracle process remained resident when another runtime was launched'
        alive[0]=True;events.append(mode)
        return SimpleNamespace(is_alive=lambda:alive[0]),SimpleNamespace(put=lambda value:None),None
    responses=[(('ready',.01),1),(('result',parity_evidence()),1)]
    monkeypatch.setattr(run,'launch',launch)
    monkeypatch.setattr(run,'receive',lambda *a,**k:responses.pop(0))
    def finish(process,requests):
        alive[0]=False;events.append('joined')
    monkeypatch.setattr(run,'finish',finish)
    proof=run.ensure_parity(manifest,'a'*64,tmp_path,'calibration',Guard(),'CONTROLLED_FIXTURE')
    assert len(proof)==1 and events==['parity','joined'] and not alive[0]
    # The exact same fixture is retained; it does not execute another sixteen forwards.
    again=run.ensure_parity(manifest,'a'*64,tmp_path,'calibration',Guard(),'CONTROLLED_RESUME')
    assert proof==again and events==['parity','joined']
    launch(manifest['configs'][2],manifest,None,Guard(),'candidate')
    assert events[-1]=='candidate'


def test_oracle_resident_runtime_cannot_render_even_before_cuda_is_touched():
    from reference import Reference
    runtime=Reference.__new__(Reference);runtime.torch=None;runtime.pipe=None;runtime.config={};runtime.mode='parity';runtime.oracle=object()
    with pytest.raises(ValueError,match='forbids oracle residency'):runtime.render({},None)


def test_success_resume_skips_validated_output_but_refuses_orphan_and_failed_rows(unavailable_run):
    import run
    manifest,lock,root=unavailable_run;config=manifest['configs'][0];cell=next(c for c in ev.cells(manifest,'calibration') if c['config_id']==config['id'])
    (root/'outputs').mkdir();output=root/'outputs'/(cell['id']+'.png');output.write_bytes(png(cell['width'],cell['height']))
    with pytest.raises(ValueError,match='orphan image'):run.existing_successes(manifest,ev.read(lock)['sha256'],root,config['id'],'calibration')
    row=dict(lock_sha256=ev.read(lock)['sha256'],cell=cell,status='ok',ownership=dict(manifest['ownership'],revision='FIXTURE'),elapsed_seconds=.1,output=dict(path=str(output.relative_to(root)),sha256=ev.file_hash(output)),log=dict(path='log',sha256=ev.file_hash(root/'log')),
        metrics=dict(generation_seconds=.1,peak_gpu_bytes=42,peak_host_rss_bytes=42,download_bytes=0,latency_condition='cold_process',cold_seconds=.1,warm_seconds=None,gpu_memory_method='CONTROLLED FIXTURE',component_dtypes={'fixture':'float32'},sampler={'fixture':'test'},runtime_versions={'fixture':'1'},pixels_finite=True))
    path=root/'receipts'/(cell['id']+'.json');ev.write(path,ev.seal(row))
    assert run.existing_successes(manifest,ev.read(lock)['sha256'],root,config['id'],'calibration')==[]
    row.update(status='failed',error='controlled fixture',reproduce='test');path.write_text(json.dumps(ev.seal(row)))
    with pytest.raises(ValueError,match='failed attempts'):run.existing_successes(manifest,ev.read(lock)['sha256'],root,config['id'],'calibration')


def test_parity_tampering_and_wrong_candidate_are_rejected(tmp_path):
    manifest=ev.template();cell=ev.parity_cells(manifest)[0];log=tmp_path/'log';log.write_text('CONTROLLED FIXTURE')
    row=dict(lock_sha256='a'*64,cell=cell,status='ok',evidence=parity_evidence(),log=dict(path='log',sha256=ev.file_hash(log)))
    assert ev.validate_parity(tmp_path,'a'*64,cell,ev.seal(row))['evidence']['forward_count']==16
    for key,value in [('candidate_arm','full_fp32_oracle'),('forward_count',8),('generated_images',1),('oracle_autocast',True)]:
        mutant=deepcopy(row);mutant['evidence'][key]=value
        with pytest.raises(ValueError):ev.validate_parity(tmp_path,'a'*64,cell,ev.seal(mutant))


def test_dmd_image_receipt_binds_candidate_arm_to_separate_fixture_dataspace_and_noise(tmp_path):
    manifest=ev.template();cell=next(c for c in ev.cells(manifest,'calibration') if c['config_id']=='moss_dmd');fixture=ev.parity_cells(manifest,'calibration')[0]
    (tmp_path/'parity').mkdir();(tmp_path/'log').write_text('CONTROLLED FIXTURE')
    proof=ev.seal(dict(lock_sha256='a'*64,cell=fixture,status='ok',evidence=parity_evidence(),log=dict(path='log',sha256=ev.file_hash(tmp_path/'log'))))
    ev.write(tmp_path/'parity'/(fixture['id']+'.json'),proof)
    output=tmp_path/'fixture.png';output.write_bytes(png(cell['width'],cell['height']))
    metrics=dict(generation_seconds=.1,peak_gpu_bytes=42,peak_host_rss_bytes=42,download_bytes=0,latency_condition='cold_process',cold_seconds=.1,warm_seconds=None,gpu_memory_method='CONTROLLED FIXTURE',component_dtypes={'fixture':'mixed'},runtime_versions={'fixture':'1'},pixels_finite=True,
        parity_fixture_digests=[proof['sha256']],noise_sha256=proof['evidence']['noise_sha256'],sampler=dict(Q_sha256=proof['evidence']['Q_sha256'],step_codes=proof['evidence']['step_codes'],execution_arm='sft_bf16_unmerged_fp32_adapter',oracle_resident=False))
    ownership={k:'CONTROLLED FIXTURE' for k in ('responsible_chat','job','revision','timeout_seconds','resume','next_action')}
    row=dict(lock_sha256='a'*64,cell=cell,status='ok',ownership=ownership,elapsed_seconds=.1,output=dict(path='fixture.png',sha256=ev.file_hash(output)),log=dict(path='log',sha256=ev.file_hash(tmp_path/'log')),metrics=metrics)
    assert ev.validate_receipt(tmp_path,'a'*64,cell,ev.seal(row))['status']=='ok'
    for key,value in [('Q_sha256','c'*64),('execution_arm','full_fp32_oracle'),('oracle_resident',True)]:
        mutant=deepcopy(row);mutant['metrics']['sampler'][key]=value
        with pytest.raises(ValueError):ev.validate_receipt(tmp_path,'a'*64,cell,ev.seal(mutant))
    mutant=deepcopy(row);mutant['metrics']['noise_sha256']='d'*64
    with pytest.raises(ValueError,match='measured input'):ev.validate_receipt(tmp_path,'a'*64,cell,ev.seal(mutant))


@pytest.fixture
def mixed_calibration(tmp_path):
    """Synthetic CPU evidence, never a claim about actual config availability."""
    import bootstrap as b
    import ownership as o
    manifest=ev.template()
    manifest['hardware'].update(host=__import__('socket').gethostname(),gpu_uuid=o.RUNNER_GPU_UUID)
    manifest['ownership']=dict(responsible_chat='CPU fixture coordinator',lease='fixture',job='fixture',timeout_seconds=30,resume='fixture',next_action='fixture')
    lock=tmp_path/'protocol.lock.json';ev.write(lock,ev.seal(manifest));lock_hash=ev.read(lock)['sha256']
    root=tmp_path/'run';(root/'receipts').mkdir(parents=True);(root/'outputs').mkdir()
    logs=root/'logs';logs.mkdir();metrics=dict(generation_seconds=.1,peak_gpu_bytes=42,peak_host_rss_bytes=42,download_bytes=0,latency_condition='cold_process',cold_seconds=.1,warm_seconds=None,gpu_memory_method='CONTROLLED CPU FIXTURE',component_dtypes={'fixture':'none'},sampler={'fixture':True},runtime_versions={'fixture':'1'},pixels_finite=True)
    for cell in ev.cells(manifest,'calibration'):
        log=logs/(cell['config_id']+'.log');log.write_text('CONTROLLED CPU runtime failure fixture; not observed model evidence')
        row=dict(lock_sha256=lock_hash,cell=cell,ownership=dict(manifest['ownership'],revision=lock_hash),elapsed_seconds=.1,log=dict(path='logs/'+log.name,sha256=ev.file_hash(log)))
        if cell['config_id']=='moss_dmd':row.update(status='failed',failure_kind='runtime',error='Synthetic owned runtime failure',reproduce='CPU fixture only',output=None,metrics=None)
        else:
            out=root/'outputs'/(cell['id']+'.png');out.write_bytes(png(cell['width'],cell['height']))
            row.update(status='ok',output=dict(path='outputs/'+out.name,sha256=ev.file_hash(out)),metrics=deepcopy(metrics))
        ev.write(root/'receipts'/(cell['id']+'.json'),ev.seal(row))
    results=o.calibration_results(manifest,lock_hash,root)
    ev.write(tmp_path/'calibration-estimate.json',b.calibration_estimate(manifest,results))
    failed=next(r for r in results.values() if r['status']=='failed')
    value=dict(protocol_sha256=lock_hash,calibration_receipts_sha256=o.calibration_bundle(lock,root),calibration_estimate_sha256=ev.file_hash(tmp_path/'calibration-estimate.json'),authorization=dict(id='CPU disposition',coordinator=manifest['ownership']['responsible_chat'],accepted_calibration=True,max_scored_outputs=288,acknowledge_shape_uncertainty=True),unavailable_configs={'moss_dmd':dict(calibration_receipt_sha256=ev.file_hash(root/'receipts'/(failed['cell']['id']+'.json')),calibration_log_sha256=failed['log']['sha256'],reason='Synthetic genuinely unavailable runtime fixture',reproduce='CPU fixture only; no actual model failure',code_and_acquisition_defects_excluded=True)})
    return manifest,lock,root,value,metrics


def test_mixed_calibration_requires_explicit_actual_failure_disposition(mixed_calibration):
    import bootstrap as b
    import ownership as o
    m,lock,root,value,_=mixed_calibration;sha=ev.read(lock)['sha256']
    assert set(o.availability_decisions(m,sha,root,value))=={'moss_dmd'}
    for change in ('missing','healthy','log','coordinator','defect'):
        bad=deepcopy(value)
        if change=='missing':bad.pop('unavailable_configs')
        elif change=='healthy':bad['unavailable_configs']['moss_base']=bad['unavailable_configs']['moss_dmd']
        elif change=='log':bad['unavailable_configs']['moss_dmd']['calibration_log_sha256']='0'*64
        elif change=='coordinator':bad['authorization']['coordinator']='other'
        else:bad['unavailable_configs']['moss_dmd']['code_and_acquisition_defects_excluded']=False
        with pytest.raises(ValueError):o.availability_decisions(m,sha,root,bad)
    b.accepted_scored_authorization(root.parent,value,m)
    changed=deepcopy(value);changed['authorization']['id']='posthoc replacement'
    with pytest.raises(ValueError,match='authorization changed'):b.accepted_scored_authorization(root.parent,changed,m)
    (root/'logs'/'moss_dmd.log').write_text('tampered log')
    with pytest.raises(ValueError,match='artifact hash mismatch'):o.availability_decisions(m,sha,root,value)


def test_invalidations_and_postscored_exclusion_cannot_be_approved(mixed_calibration):
    import bootstrap as b
    import ownership as o
    m,lock,root,value,_=mixed_calibration;sha=ev.read(lock)['sha256']
    marker=root/'outputs'/'scored.orphan.png';marker.write_bytes(b'fixture')
    with pytest.raises(ValueError,match='precede'):b.accepted_scored_authorization(root.parent,value,m)
    marker.unlink()
    ev.write(root/'invalidations'/'guard.json',ev.seal(dict(lock_sha256=sha,cell_ids=[],parity_cell_ids=[],reason='Synthetic foreign owner',attempt='fixture')))
    with pytest.raises(ValueError,match='invalidations cannot be approved'):o.availability_decisions(m,sha,root,value)
    assert not (root.parent/'scored-authorization.json').exists()


def test_approved_failed_config_emits_all48_cells_without_gpu_or_calibration_rewrites(mixed_calibration,monkeypatch,tmp_path):
    import bootstrap as b
    import ownership as o
    import run
    m,lock,root,value,metrics=mixed_calibration;sha=ev.read(lock)['sha256']
    before={p:p.read_bytes() for p in (root/'receipts').glob('calibration.*.json')}
    before.update({p:p.read_bytes() for p in (root/'outputs').glob('calibration.*.png')})
    b.accepted_scored_authorization(root.parent,value,m)
    class Guard:
        def __init__(self,*a,**k):self.prepared_root=root.parent;self.unavailable_configs=o.availability_decisions(m,sha,root,value)
        def check(self,*a,**k):pass
    monkeypatch.setattr(o,'ExecutionGuard',Guard)
    monkeypatch.setattr(run,'validate_manifest',lambda *a,**k:None)  # CPU fixture has no weights/hardware.
    monkeypatch.setattr(run,'launch',lambda *a,**k:pytest.fail('approved failed config spawned a runtime'))
    run.execute(lock,root,'moss_dmd','scored')
    records=list((root/'receipts').glob('scored.*.json'));assert len(records)==48
    saved={p:p.read_bytes() for p in records}
    assert all(ev.read(p)['status']=='unavailable' for p in records)
    assert run.execute(lock,root,'moss_dmd','scored')==0
    assert all(p.read_bytes()==body for p,body in {**before,**saved}.items())
    cases={c['id']:c for c in m['corpus']['cases']};annotations={};images={}
    for cell in ev.cells(m):
        if cell['config_id']=='moss_dmd':continue
        shape=(cell['width'],cell['height'])
        if shape not in images:images[shape]=png(*shape)
        out=root/'outputs'/(cell['id']+'.png');out.write_bytes(images[shape])
        row=dict(lock_sha256=sha,cell=cell,status='ok',ownership=dict(m['ownership'],revision=sha),elapsed_seconds=.1,output=dict(path='outputs/'+out.name,sha256=ev.file_hash(out)),log=dict(path='logs/moss_base.log',sha256=ev.file_hash(root/'logs'/'moss_base.log')),metrics=deepcopy(metrics),availability_authorization=dict(path='availability-authorization.json',sha256=ev.file_hash(root/'availability-authorization.json')))
        ev.write(root/'receipts'/(cell['id']+'.json'),ev.seal(row))
        case=cases[cell['case_id']];annotations[cell['id']]=dict(output_sha256=row['output']['sha256'],annotator='CPU synthetic fixture',method='fixture',tool_revision='fixture',text=case['expected_text'],boxes={box['id']:box['box'] for box in case['boxes']})
    _,complete=ev.receipts(lock,root);assert len(complete)==288
    public=tmp_path/'mixed-pairs.json';ev.blind_pairs(lock,root,public)
    pairs=ev.unseal(ev.read(public))['pairs'];assert len(pairs)==432
    judgments=[dict(pair_id=p['id'],winner='unavailable' if any(a['image'] is None for a in p['arms']) else 'tie',rubric={k:3 for k in ('visual_quality','instruction_following','composition')},reviewer='CPU fixture',reviewer_kind='machine_assisted',judge_version='fixture',notes='Synthetic CPU records; not real judgments',uncertainty='high') for p in pairs]
    ev.write(tmp_path/'annotations.json',annotations);ev.write(tmp_path/'judgments.json',judgments)
    report=ev.report(lock,root,public,tmp_path/'judgments.json',tmp_path/'annotations.json')
    assert report['missing_outputs']==48 and report['usable_blind_pairs']==288 and report['blind_pairs']==432
    assert report['recommendation_eligible'] is False
    # An offline reader must reject whole-campaign removal, not only one row.
    for path in (root/'receipts').glob('scored.*.json'):
        row=ev.unseal(ev.read(path));row.pop('availability_authorization');path.write_text(json.dumps(ev.seal(row)))
    with pytest.raises(ValueError,match='lacks original availability'):ev.receipts(lock,root)
    (root/'availability-authorization.json').unlink()
    with pytest.raises(ValueError,match='unavailable status requires'):ev.receipts(lock,root)


def test_transport_finishes_independent_calibrations_and_resumes_terminal_attempts_without_rerender(tmp_path,monkeypatch):
    """Real transport/runner control flow with synthetic workers; no hardware imports."""
    import bootstrap as b
    import ownership as o
    import run
    import socket
    from types import SimpleNamespace
    m=ev.template();m['hardware'].update(host=socket.gethostname(),gpu_uuid=o.RUNNER_GPU_UUID)
    m['ownership']=dict(responsible_chat='fixture',lease='fixture',job='fixture',timeout_seconds=30,resume='fixture',next_action='fixture')
    lock=tmp_path/'protocol.lock.json';ev.write(lock,ev.seal(m));sha=ev.read(lock)['sha256'];calls=[]
    class Guard:
        unavailable_configs={}
        def __init__(self,*a,**k):self.prepared_root=tmp_path
        def check(self,*a,**k):return {'cpu_fixture':True}
    monkeypatch.setattr(b,'ExecutionGuard',Guard);monkeypatch.setattr(o,'ExecutionGuard',Guard)
    ev.write(tmp_path/'prepared.json',{'CPU_fixture':True})
    monkeypatch.setattr(b,'prepared',lambda root:{})
    monkeypatch.setattr(ev,'load_lock',lambda *a,**k:m)
    monkeypatch.setattr(run,'validate_manifest',lambda *a,**k:None)
    def parity(*a,**k):raise run.RuntimeFailure('Synthetic guarded DMD worker failed; CPU fixture only')
    monkeypatch.setattr(run,'ensure_parity',parity)
    metrics=dict(generation_seconds=.01,peak_gpu_bytes=42,peak_gpu_reserved_bytes=42,peak_host_rss_bytes=42,download_bytes=0,gpu_memory_method='CPU fixture',component_dtypes={'fixture':'none'},sampler={'fixture':True},runtime_versions={'fixture':'1'},pixels_finite=True)
    class Requests:
        def put(self,item):
            if item:
                cell,out=item;Path(out).write_bytes(png(cell['width'],cell['height']))
    def launch(*a,**k):return SimpleNamespace(),Requests(),iter([('ready',.01),('result',deepcopy(metrics))])
    monkeypatch.setattr(run,'launch',launch)
    monkeypatch.setattr(run,'receive',lambda p,q,*a:(next(q),42))
    monkeypatch.setattr(run,'finish',lambda *a:None);monkeypatch.setattr(run,'stop',lambda *a:None)
    class Process:
        returncode=0
        def __init__(self,args,**kwargs):
            config=args[args.index('--config')+1];calls.append(config)
            run.execute(lock,tmp_path/'run',config,'calibration')
        def poll(self):return 0
    monkeypatch.setattr(b.subprocess,'Popen',Process);monkeypatch.setattr(b,'kill_owned',lambda *a:None)
    deadline='2030-01-01T00:00:00+00:00'
    value=dict(protocol_sha256=sha,deadline_utc=deadline,lease=dict(id='fixture',owner='fixture',host=socket.gethostname(),gpu_uuid=o.RUNNER_GPU_UUID,expires_utc=deadline,external_owners_reconciled=True,desktop_baseline=[],baseline_max_memory_mib=1000,baseline_max_utilization_percent=2))
    monkeypatch.setenv('GITHUB_RUN_ID','fixture-1')
    b.execute_phase(tmp_path,value,'calibration')
    assert calls==list(ev.CONFIG_IDS)
    _,results=ev.receipts(lock,tmp_path/'run','calibration')
    assert sum(r['status']=='ok' for r in results.values())==5
    assert next(r for r in results.values() if r['status']=='failed')['failure_kind']=='runtime'
    assert len(list((tmp_path/'run'/'outputs').glob('*.png')))==5
    saved={p:p.read_bytes() for p in (tmp_path/'run'/'receipts').glob('calibration.*.json')}
    monkeypatch.setenv('GITHUB_RUN_ID','fixture-2')
    b.execute_phase(tmp_path,value,'calibration')
    assert calls==list(ev.CONFIG_IDS) and all(p.read_bytes()==body for p,body in saved.items())
    estimate=ev.read(tmp_path/'calibration-estimate.json')
    assert estimate['calibration_attempts']==estimate['calibration_output_cap']==6
    assert estimate['calibration_outputs']==estimate['successful_calibration_outputs']==5
    assert estimate['conditions']['moss_dmd']['candidate_generation_seconds'] is None


def test_mixed_campaign_healthy_generation_binds_same_decision_before_receipt_write(mixed_calibration,monkeypatch):
    import bootstrap as b
    import ownership as o
    import run
    from types import SimpleNamespace
    m,lock,root,value,metrics=mixed_calibration;sha=ev.read(lock)['sha256']
    b.accepted_scored_authorization(root.parent,value,m)
    class Guard:
        def __init__(self,*a,**k):self.prepared_root=root.parent;self.unavailable_configs=o.availability_decisions(m,sha,root,value)
        def check(self,*a,**k):pass
    monkeypatch.setattr(o,'ExecutionGuard',Guard);monkeypatch.setattr(run,'validate_manifest',lambda *a,**k:None)
    cache={}
    class Requests:
        def put(self,item):
            if item:
                cell,path=item;shape=(cell['width'],cell['height'])
                if shape not in cache:cache[shape]=png(*shape)
                Path(path).write_bytes(cache[shape])
    def replies():
        yield ('ready',.01)
        while True:yield ('result',dict(deepcopy(metrics),peak_gpu_reserved_bytes=42))
    calls=[]
    def launch(config,*a,**k):calls.append(config['id']);return SimpleNamespace(),Requests(),replies()
    monkeypatch.setattr(run,'launch',launch);monkeypatch.setattr(run,'receive',lambda p,q,*a:(next(q),42));monkeypatch.setattr(run,'finish',lambda *a:None)
    run.execute(lock,root,'moss_base','scored')
    assert calls==['moss_base']
    records=list((root/'receipts').glob('scored.*.json'));assert len(records)==48
    assert all(ev.read(p)['availability_authorization']['sha256']==ev.file_hash(root/'availability-authorization.json') for p in records)
    original=records[0].read_bytes();row=ev.unseal(ev.read(records[0]));row.pop('availability_authorization');records[0].write_text(json.dumps(ev.seal(row)))
    with pytest.raises(ValueError,match='post-scored exclusion forbidden'):o.availability_decisions(m,sha,root,value)
    records[0].write_bytes(original)
    mirror=root/'availability-authorization.json';mirror.write_bytes(mirror.read_bytes()+b' ')
    with pytest.raises(ValueError):b.accepted_scored_authorization(root.parent,value,m)


@pytest.mark.parametrize('failure',['worker_error','owned_timeout','guard','invalid_metrics','parity_integrity'])
def test_calibration_runtime_failures_are_terminal_but_integrity_errors_invalidate(tmp_path,monkeypatch,failure):
    import ownership as o
    import run
    from types import SimpleNamespace
    m=ev.template();m['ownership']=dict(responsible_chat='fixture',lease='fixture',job='fixture',timeout_seconds=30,resume='fixture',next_action='fixture')
    lock=tmp_path/'protocol.lock.json';ev.write(lock,ev.seal(m));root=tmp_path/'run'
    config='moss_dmd' if failure=='parity_integrity' else 'moss_base'
    class Guard:
        unavailable_configs={}
        def __init__(self,*a,**k):self.prepared_root=tmp_path
        def check(self,*a,**k):pass
    monkeypatch.setattr(o,'ExecutionGuard',Guard);monkeypatch.setattr(run,'validate_manifest',lambda *a,**k:None)
    class Requests:
        def put(self,item):
            if item:
                cell,out=item;Path(out).write_bytes(png(cell['width'],cell['height']))
    monkeypatch.setattr(run,'launch',lambda *a,**k:(SimpleNamespace(),Requests(),None))
    monkeypatch.setattr(run,'finish',lambda *a:None);monkeypatch.setattr(run,'stop',lambda *a:None)
    def parity(*a,**k):raise ValueError('Synthetic malformed parity proof')
    monkeypatch.setattr(run,'ensure_parity',parity)
    calls=[]
    def receive(*a,**k):
        calls.append(True)
        if failure=='worker_error':return (('error','Synthetic CUDA RuntimeError'),0)
        if failure=='owned_timeout':raise run.RuntimeTimeout('Synthetic owned timeout')
        if failure=='guard':raise ValueError('Synthetic foreign owner guard')
        if len(calls)==1:return (('ready',.01),0)
        return (('result',dict(generation_seconds=.1,peak_gpu_reserved_bytes=42,pixels_finite=False)),0)
    monkeypatch.setattr(run,'receive',receive)
    if failure in ('worker_error','owned_timeout'):
        run.execute(lock,root,config,'calibration')
        row=ev.read(next((root/'receipts').glob('calibration.*.json')))
        assert row['failure_kind']=='runtime' and row['status']==('timeout' if failure=='owned_timeout' else 'failed')
        assert not list((root/'invalidations').glob('*.json'))
    else:
        with pytest.raises(ValueError):run.execute(lock,root,config,'calibration')
        assert not list((root/'receipts').glob('calibration.*.json'))
        assert list((root/'invalidations').glob('*.json'))
    assert run.worker_failure(ValueError('nonfinite or shape/hash validation'))=='integrity_error'
    assert run.worker_failure(ImportError('dependency defect'))=='integrity_error'
    for defect in (AttributeError('code defect'),NameError('code defect'),IndexError('code defect'),Exception('unknown defect')):assert run.worker_failure(defect)=='integrity_error'
    assert run.worker_failure(RuntimeError('CUDA runtime'))=='error'
    assert run.worker_failure(TimeoutError('owned timeout'))=='timeout'


def test_all_success_default_keeps_finite_parity_differences_as_quality_evidence(mixed_calibration):
    import bootstrap as b
    import ownership as o
    m,lock,root,value,metrics=mixed_calibration;sha=ev.read(lock)['sha256']
    proof=parity_evidence()
    for step in proof['steps']:step.update(max_abs=1.,mean_abs=.1)  # Deliberately above frozen report thresholds.
    fixture=ev.parity_cells(m,'calibration')[0]
    parity=ev.seal(dict(lock_sha256=sha,cell=fixture,status='ok',evidence=proof,log=dict(path='logs/moss_dmd.log',sha256=ev.file_hash(root/'logs'/'moss_dmd.log'))))
    ev.write(root/'parity'/(fixture['id']+'.json'),parity)
    cell=next(c for c in ev.cells(m,'calibration') if c['config_id']=='moss_dmd')
    out=root/'outputs'/(cell['id']+'.png');out.write_bytes(png(cell['width'],cell['height']))
    path=root/'receipts'/(cell['id']+'.json');row=ev.unseal(ev.read(path))
    row.update(status='ok',output=dict(path='outputs/'+out.name,sha256=ev.file_hash(out)),metrics=dict(deepcopy(metrics),parity_fixture_digests=[parity['sha256']],noise_sha256=proof['noise_sha256'],sampler=dict(Q_sha256=proof['Q_sha256'],step_codes=proof['step_codes'],execution_arm='sft_bf16_unmerged_fp32_adapter',oracle_resident=False)))
    row.pop('failure_kind');row.pop('error');row.pop('reproduce');path.write_text(json.dumps(ev.seal(row)))
    results=o.calibration_results(m,sha,root)
    assert len(results)==6 and all(r['status']=='ok' for r in results.values())
    assert all(s['max_abs']>m['thresholds']['dmd_prediction_max_abs'] for s in ev.validate_parity(root,sha,fixture,parity)['evidence']['steps'])
    estimate=root.parent/'calibration-estimate.json';estimate.write_text(json.dumps(b.calibration_estimate(m,results)))
    value.pop('unavailable_configs');value.update(calibration_receipts_sha256=o.calibration_bundle(lock,root),calibration_estimate_sha256=ev.file_hash(estimate))
    b.accepted_scored_authorization(root.parent,value,m)
    assert set(ev.read(root.parent/'scored-authorization.json'))=={'protocol_sha256','calibration_receipts_sha256','calibration_estimate_sha256','authorization'}
    assert not (root/'availability-authorization.json').exists()
    assert len(list(ev.cells(m)))==288  # Finite numerical differences do not remove DMD image cases.


def test_direct_cli_cannot_fork_receipt_history_outside_held_campaign(mixed_calibration,monkeypatch):
    import ownership as o
    import run
    m,lock,root,_,_=mixed_calibration
    class Guard:
        unavailable_configs={}
        def __init__(self,*a,**k):self.prepared_root=root.parent
        def check(self,*a,**k):pass
    monkeypatch.setattr(o,'ExecutionGuard',Guard);monkeypatch.setattr(run,'validate_manifest',lambda *a,**k:None)
    monkeypatch.setattr(run,'launch',lambda *a,**k:pytest.fail('fork reached GPU spawn'))
    with pytest.raises(ValueError,match='original prepared campaign authority'):run.execute(lock,root.parent/'forked-run','moss_base','calibration')
    copied=root.parent/'copied-lock.json';copied.write_bytes(lock.read_bytes())
    with pytest.raises(ValueError,match='original prepared campaign authority'):run.execute(copied,root,'moss_base','calibration')


@pytest.mark.parametrize('status,exception',[('failed',RuntimeError),('canceled',ValueError)])
def test_native_terminal_failure_preserves_error_without_canceling_terminal_job(status,exception,tmp_path):
    from native import Native
    import run
    runtime=Native.__new__(Native);runtime.config=deepcopy(ev.template()['configs'][3]);runtime.config['settings']['job_timeout_seconds']=1;runtime.project={'id':'CPU fixture'}
    calls=[]
    def call(method,path,body=None,raw=False):
        calls.append((method,path))
        if method=='POST':return {'id':'CPU-owned-job'}
        return {'status':status,'error':'CPU synthetic runtime error; not actual evidence'}
    runtime.call=call
    cell=next(c for c in ev.cells(ev.template()) if c['config_id']=='qwen_image_2_1')
    with pytest.raises(exception) as result:runtime.render(cell,tmp_path/'never-created.png')
    assert run.worker_failure(result.value)==('error' if status=='failed' else 'integrity_error')
    assert not any(path.endswith('/cancel') for _,path in calls)


@pytest.mark.parametrize('failure_state',['error','integrity_error'])
def test_scored_runtime_failure_censors_only_its_config_and_finishes_other_configs(mixed_calibration,monkeypatch,tmp_path,failure_state):
    import bootstrap as b
    import ownership as o
    import run
    from types import SimpleNamespace
    m,lock,root,value,metrics=mixed_calibration;sha=ev.read(lock)['sha256'];calls=[];cache={}
    b.accepted_scored_authorization(root.parent,value,m);ev.write(root.parent/'prepared.json',{'CPU_fixture':True})
    original={p:p.read_bytes() for p in (root/'receipts').glob('calibration.*.json')}
    class Guard:
        def __init__(self,*a,**k):self.prepared_root=root.parent;self.unavailable_configs=o.availability_decisions(m,sha,root,value)
        def check(self,*a,**k):return {'CPU_fixture':True}
    monkeypatch.setattr(b,'ExecutionGuard',Guard);monkeypatch.setattr(o,'ExecutionGuard',Guard)
    monkeypatch.setattr(b,'prepared',lambda *a:{})
    monkeypatch.setattr(ev,'load_lock',lambda *a,**k:m);monkeypatch.setattr(run,'validate_manifest',lambda *a,**k:None)
    class Requests:
        pending=None
        def put(self,item):self.pending=item
    def launch(config,*a,**k):
        q=Requests()
        def replies():
            yield ('ready',.01)
            count=0
            while True:
                count+=1
                if config['id']=='moss_base' and count==2:yield (failure_state,'Synthetic owned runtime error, CPU fixture only')
                else:
                    cell,path=q.pending;shape=(cell['width'],cell['height'])
                    if shape not in cache:cache[shape]=png(*shape)
                    Path(path).write_bytes(cache[shape])
                    yield ('result',dict(deepcopy(metrics),peak_gpu_reserved_bytes=42))
        return SimpleNamespace(),q,replies()
    monkeypatch.setattr(run,'launch',launch);monkeypatch.setattr(run,'receive',lambda p,q,*a:(next(q),42));monkeypatch.setattr(run,'finish',lambda *a:None);monkeypatch.setattr(run,'stop',lambda *a:None)
    class Process:
        returncode=0
        def __init__(self,args,**kwargs):
            config=args[args.index('--config')+1];calls.append(config)
            run.execute(lock,root,config,'scored')
        def poll(self):return 0
    monkeypatch.setattr(b.subprocess,'Popen',Process);monkeypatch.setattr(b,'kill_owned',lambda *a:None)
    deadline='2030-01-01T00:00:00+00:00'
    control=dict(value,deadline_utc=deadline,lease=dict(id=m['ownership']['lease'],owner=m['ownership']['responsible_chat'],host=m['hardware']['host'],gpu_uuid=o.RUNNER_GPU_UUID,expires_utc=deadline,external_owners_reconciled=True,desktop_baseline=[],baseline_max_memory_mib=1000,baseline_max_utilization_percent=2))
    monkeypatch.setenv('GITHUB_RUN_ID','CPU-scored-1')
    if failure_state=='integrity_error':
        with pytest.raises(ValueError,match='Synthetic owned runtime error'):b.execute_phase(root.parent,control,'scored')
        assert calls==['moss_base'] and list((root/'invalidations').glob('*.json'))
        with pytest.raises(ValueError,match='invalidations cannot be approved'):o.availability_decisions(m,sha,root,value)
        return
    with pytest.raises(ValueError,match='attempts complete with runtime failures'):b.execute_phase(root.parent,control,'scored')
    assert calls==list(ev.CONFIG_IDS)  # The first config's second render did not stop later configs.
    _,results=ev.receipts(lock,root);assert len(results)==288
    failed=[r for r in results.values() if r.get('failure_kind')=='runtime'];censored=[r for r in results.values() if r.get('failure_kind')=='runtime_censored']
    assert len(failed)==1 and failed[0]['attempted'] is True and len(censored)==46
    assert all(r['attempted'] is False and r['runtime_failure_cell']==failed[0]['cell']['id'] for r in censored)
    assert sum(r['status']=='ok' for r in results.values())==193
    assert all(p.read_bytes()==body for p,body in original.items())
    public=tmp_path/'terminal-pairs.json';ev.blind_pairs(lock,root,public)
    pairs=ev.unseal(ev.read(public))['pairs'];assert len(pairs)==432
    cases={c['id']:c for c in m['corpus']['cases']};annotations={}
    for key,row in results.items():
        if row['status']=='ok':
            case=cases[row['cell']['case_id']];annotations[key]=dict(output_sha256=row['output']['sha256'],annotator='CPU fixture',method='fixture',tool_revision='fixture',text=case['expected_text'],boxes={v['id']:v['box'] for v in case['boxes']})
    judgments=[dict(pair_id=p['id'],winner='unavailable' if any(a['image'] is None for a in p['arms']) else 'tie',rubric={k:3 for k in ('visual_quality','instruction_following','composition')},reviewer='CPU fixture',reviewer_kind='machine_assisted',judge_version='fixture',notes='Synthetic records only',uncertainty='high') for p in pairs]
    ev.write(tmp_path/'terminal-annotations.json',annotations);ev.write(tmp_path/'terminal-judgments.json',judgments)
    report=ev.report(lock,root,public,tmp_path/'terminal-judgments.json',tmp_path/'terminal-annotations.json')
    assert report['failed_attempts']==1 and report['censored_outputs']==46 and report['missing_outputs']==95
    before={p:p.read_bytes() for p in (root/'receipts').glob('*.json')}
    monkeypatch.setenv('GITHUB_RUN_ID','CPU-scored-2')
    with pytest.raises(ValueError,match='attempts complete with runtime failures'):b.execute_phase(root.parent,control,'scored')
    assert calls==list(ev.CONFIG_IDS) and all(p.read_bytes()==body for p,body in before.items())


def test_interrupted_stopped_config_history_finishes_only_later_censored_rows_without_spawn(mixed_calibration,monkeypatch):
    import bootstrap as b
    import ownership as o
    import run
    m,lock,root,value,metrics=mixed_calibration;sha=ev.read(lock)['sha256']
    b.accepted_scored_authorization(root.parent,value,m)
    ordered=[c for c in ev.cells(m) if c['config_id']=='moss_base'];before,cause=ordered[:2]
    binding=dict(path='availability-authorization.json',sha256=ev.file_hash(root/'availability-authorization.json'))
    log=dict(path='logs/moss_base.log',sha256=ev.file_hash(root/'logs'/'moss_base.log'))
    out=root/'outputs'/(before['id']+'.png');out.write_bytes(png(before['width'],before['height']))
    common=dict(lock_sha256=sha,ownership=dict(m['ownership'],revision=sha),elapsed_seconds=.1,log=log,availability_authorization=binding)
    ev.write(root/'receipts'/(before['id']+'.json'),ev.seal(dict(common,cell=before,status='ok',attempted=True,output=dict(path='outputs/'+out.name,sha256=ev.file_hash(out)),metrics=deepcopy(metrics))))
    ev.write(root/'receipts'/(cause['id']+'.json'),ev.seal(dict(common,cell=cause,status='failed',failure_kind='runtime',attempted=True,runtime_failure_cell=cause['id'],error='Synthetic owned failure before interruption',reproduce='CPU fixture only',output=None,metrics=None)))
    original={p:p.read_bytes() for p in (root/'receipts').glob('scored.*.json')}
    class Guard:
        def __init__(self,*a,**k):self.prepared_root=root.parent;self.unavailable_configs=o.availability_decisions(m,sha,root,value)
        def check(self,*a,**k):pass
    monkeypatch.setattr(o,'ExecutionGuard',Guard);monkeypatch.setattr(run,'validate_manifest',lambda *a,**k:None)
    monkeypatch.setattr(run,'launch',lambda *a,**k:pytest.fail('interrupted stopped config relaunched GPU work'))
    run.execute(lock,root,'moss_base','scored')
    assert all(p.read_bytes()==body for p,body in original.items())
    records=[ev.read(root/'receipts'/(cell['id']+'.json')) for cell in ordered]
    assert len(records)==48 and all(r['failure_kind']=='runtime_censored' and r['attempted'] is False for r in records[2:])
    row=ev.unseal(records[2]);cell=ordered[2]
    for change in (dict(status='ok'),dict(output={'path':'made-up.png','sha256':'0'*64}),dict(metrics={'generation_seconds':1})):
        with pytest.raises(ValueError,match='cannot claim images or measurements'):ev.validate_receipt(root,sha,cell,ev.seal(dict(row,**change)),manifest=m)
    with pytest.raises(ValueError,match='preceding its runtime failure'):ev.validate_receipt(root,sha,before,ev.seal(dict(row,cell=before)),manifest=m)
    # A missing earlier receipt is a damaged ledger, not permission to regenerate it.
    (root/'receipts'/(before['id']+'.json')).unlink();out.unlink()
    with pytest.raises(ValueError,match='missing receipt before actual runtime failure'):run.execute(lock,root,'moss_base','scored')


@pytest.fixture
def repaired_v6_fixture(carry_fixture):
    """Two synthetic source locks, one failed proof asset; no actual model run."""
    import calibration_source as cs
    import calibration_repair as cr
    import bootstrap as b
    f=carry_fixture;source,old,prep=f['make_campaign'](cr.V6)
    root,m,current=f['make_campaign']('sc-26322-reference-v7')
    for campaign,manifest,prepared in ((source,old,prep),(root,m,current)):
        c=next(c for c in manifest['configs'] if c['id']==cr.KLEIN)
        c['settings']['runtime_versions']=dict(sceneworks_source=b.SOURCE,inference=b.INFERENCE)
        snap=Path(c['snapshot']);entry=prepared['staged'][cr.KLEIN]
        for part in ('transformer','text_encoder'):
            p=snap/'q4'/part/'config.json';p.parent.mkdir(parents=True);p.write_bytes((Path(__file__).parent/'fixtures/moss-klein'/(part+'-config.json')).read_bytes())
            row=dict(path=str(p),sha256=ev.file_hash(p),bytes=p.stat().st_size);entry['files'].append(row);c['files'].append(row)
        prepared['native_build']={'fixture':'no real native build'}
        (campaign/'prepared.json').write_text(json.dumps(prepared));manifest['preparation']['sha256']=ev.file_hash(campaign/'prepared.json')
    first=deepcopy(f['request']);first['authorization']['new_harness_sha256']=ev.digest(prep['harness'])
    cs.install(source,old,first);ev.write(source/'protocol.lock.json',ev.seal(old));sha=ev.read(source/'protocol.lock.json')['sha256']
    run=source/'run';(run/'receipts').mkdir();(run/'outputs').mkdir();(run/'logs').mkdir()
    cell=next(c for c in ev.cells(old,'calibration') if c['config_id']==cr.QWEN)
    out=run/'outputs'/(cell['id']+'.png');out.write_bytes(png(cell['width'],cell['height']));log=run/'logs'/'qwen.log';log.write_text('CPU fixture successful Qwen')
    row=dict(lock_sha256=sha,cell=cell,ownership=dict(old['ownership'],revision=sha),elapsed_seconds=.1,status='ok',attempted=True,output=dict(path='outputs/'+out.name,sha256=ev.file_hash(out)),log=dict(path='logs/qwen.log',sha256=ev.file_hash(log)),metrics=deepcopy(f['metrics']))
    receipt=run/'receipts'/(cell['id']+'.json');ev.write(receipt,ev.seal(row))
    klein=next(c for c in ev.cells(old,'calibration') if c['config_id']==cr.KLEIN)
    for name,reason in [('inner','ValueError: native actual worker/model/tier/settings mismatch'),('outer','outer owned-runtime watchdog stopped attempt: calibration process failed; inspect fixture.log')]:
        ev.write(run/'invalidations'/(name+'.json'),ev.seal(dict(lock_sha256=sha,cell_ids=[klein['id']],parity_cell_ids=[],reason=reason,attempt='CPU fixture')))
    job=source/'job-456-1-calibration.json';ev.write(job,dict(mode='calibration',status='failed',job='456-1',root=str(source),sceneworks_source=b.SOURCE,inference=b.INFERENCE,transport_sha256=prep['harness']['bootstrap.py'],accelerator_execution_requested=True))
    diagnostic=root/'repair-evidence';diagnostic.mkdir();jid='job_cpu_fixture'
    recipe=dict(model=cr.KLEIN,rawAdapterSettings=dict(mlxQuantize=4,realModelInference=True,numInferenceSteps=4))
    asset=dict(id='asset_fixture',recipe=recipe,lineage=dict(jobId=jid))
    state=dict(job=dict(id=jid,type='image_generate',status='completed',worker_id='sc26322-'+cr.KLEIN,payload_json=json.dumps(dict(model=cr.KLEIN,seed=klein['seed'],width=klein['width'],height=klein['height'],count=1,prompt=klein['prompt'],advanced=dict(mlxQuantize=4,mlxQuantizeExplicit=True))),result_json=json.dumps(dict(assets=[asset]))),metrics=dict(job_id=jid,model=cr.KLEIN,quant_label='bf16',quant_bits=None,steps=4,seed=klein['seed'],width=klein['width'],height=klein['height'],image_count=1,backend='cuda'))
    ev.write(diagnostic/'selected-state.json',state);ev.write(diagnostic/'asset.recipe.json',recipe);ev.write(diagnostic/'asset.sceneworks.json',asset);ev.write(diagnostic/'snapshot-index.json',dict(fixture=True))
    (diagnostic/'original.png').write_bytes(png(klein['width'],klein['height']))
    ev.write(diagnostic/'selected-index.json',dict(status='complete',job_id=jid,selected_state_sha256=ev.file_hash(diagnostic/'selected-state.json'),files=[dict(artifact='original.png',sha256=ev.file_hash(diagnostic/'original.png'))]))
    events=[dict(event='image_memory_strategy_selected',job_id=jid,model=cr.KLEIN,route=cr.KLEIN,actual_tier='q4'),dict(event='image_inference_complete',jobId=jid,backend='candle')]
    (diagnostic/'worker.log').write_text('\n'.join(json.dumps(e) for e in events))
    request=dict(kind='klein_metrics_v6',campaign=cr.V6,protocol_sha256=sha,prepared_sha256=ev.file_hash(source/'prepared.json'),job='456-1',job_receipt_sha256=ev.file_hash(job),qwen_receipt_sha256=ev.file_hash(receipt),invalidations={p.name:ev.file_hash(p) for p in (run/'invalidations').glob('*.json')},diagnostic_files={p.name:ev.file_hash(p) for p in diagnostic.iterdir()},failed_native_job=jid,binary_sha256=prep['binary_sha256'],prior_claim=old['calibration_carry_forward']['claim'],authorization=dict(id='CPU exact diagnosed repair',coordinator=m['ownership']['responsible_chat'],reason='CPU fixture reporting-only defect',new_harness_sha256=ev.digest(current['harness']),remaining_calibration_images=2,failed_attempt_classification='diagnosed_harness_tier_reporting_proof_failure'))
    return dict(source=source,old=old,root=root,manifest=m,request=request,make_campaign=f['make_campaign'],metrics=f['metrics'])


def test_fixed_v6_repair_preserves_four_and_only_two_calibration_slots(repaired_v6_fixture,monkeypatch):
    import calibration_source as cs
    import calibration_repair as cr
    import ownership as o
    import run
    f=repaired_v6_fixture;root=f['root'];m=f['manifest'];source=f['source']
    originals={p:p.read_bytes() for p in source.rglob('*') if p.is_file()}
    previous_claim=Path(f['request']['prior_claim']['path']);claim_bytes=previous_claim.read_bytes()
    cs.install(root,m,f['request']);ev.write(root/'protocol.lock.json',ev.seal(m))
    rows,parity=cs.carried(m,root/'run')
    assert len(rows)==4 and len(parity)==1 and len({r['lock_sha256'] for r in rows.values()})==2
    assert all(p.read_bytes()==data for p,data in originals.items()) and previous_claim.read_bytes()==claim_bytes
    assert len(ev.receipts(root/'protocol.lock.json',root/'run','calibration',complete=False)[1])==4
    for config in cr.CARRIED:
        assert run.existing_successes(m,'current',root/'run',config,'calibration')==[]
        assert len(run.existing_successes(m,'current',root/'run',config,'scored'))==48
    for config in (cr.KLEIN,'z_image_turbo'):assert len(run.existing_successes(m,'current',root/'run',config,'calibration'))==1
    assert cs.calibration_record_root(m,root/'run',cr.QWEN)==root/'run'/cs.DIRECTORY/'run'
    assert cs.calibration_record_root(m,root/'run','moss_dmd')==root/'run'/cs.DIRECTORY/'run'/cs.DIRECTORY/'run'
    class Guard:
        def __init__(self,*a,**kw):self.prepared_root=root;self.unavailable_configs={}
        def check(self,*a,**kw):cs.check_carry_closure(m,root/'run',execution=True)
    monkeypatch.setattr(o,'ExecutionGuard',Guard);monkeypatch.setattr(run,'validate_manifest',lambda *a,**kw:None)
    monkeypatch.setattr(run,'launch',lambda *a,**kw:pytest.fail('carried calibration spawned'))
    for config in cr.CARRIED:assert run.execute(root/'protocol.lock.json',root/'run',config,'calibration')==0
    assert len(list(ev.cells(m,'scored')))==288
    with pytest.raises(ValueError):cs.install(root,m,f['request'])


@pytest.mark.parametrize('change',['extra_invalidation','different_fault','qwen_tamper','binary','driver','prior_claim','native_dependency','native_helper','scored','budget','unsealed_claim','new_carried_image','new_carried_receipt','diagnostic_tier','diagnostic_event','diagnostic_png'])
def test_fixed_v6_repair_rejects_mutations(repaired_v6_fixture,change):
    import calibration_source as cs
    f=repaired_v6_fixture;root=f['root'];m=f['manifest'];s=f['source'];r=f['request']
    if change=='extra_invalidation':ev.write(s/'run/invalidations'/'extra.json',ev.seal(dict(lock_sha256=r['protocol_sha256'],cell_ids=[],parity_cell_ids=[],reason='lease expiry')))
    elif change=='different_fault':
        p=s/'run/invalidations/inner.json';d=ev.unseal(ev.read(p));d['reason']='ValueError: foreign owner';p.write_text(json.dumps(ev.seal(d)));r['invalidations'][p.name]=ev.file_hash(p)
    elif change=='qwen_tamper':next((s/'run/outputs').glob('*.png')).write_bytes(b'wrong')
    elif change=='binary':m['configs'][3]['settings']['binary_sha256']='0'*64
    elif change=='driver':m['runtime']['driver']='changed'
    elif change=='prior_claim':r['prior_claim']['sha256']='0'*64
    elif change in ('native_dependency','native_helper'):
        p=root/'harness/native.py';text=p.read_text().replace('timeout=30','timeout=31') if change=='native_dependency' else p.read_text().replace("'Klein defect proof requires exact frozen source/pin/q4'","'altered proof helper'")
        p.write_text(text);prepared=ev.read(root/'prepared.json');prepared['harness']['native.py']=ev.file_hash(p);(root/'prepared.json').write_text(json.dumps(prepared));m['preparation']['sha256']=ev.file_hash(root/'prepared.json');r['authorization']['new_harness_sha256']=ev.digest(prepared['harness'])
    elif change=='scored':(s/'run/outputs/scored.fake.png').write_bytes(b'forbidden')
    elif change=='budget':r['authorization']['remaining_calibration_images']=3
    elif change.startswith('diagnostic_'):
        p=root/'repair-evidence'/('selected-state.json' if change=='diagnostic_tier' else 'worker.log' if change=='diagnostic_event' else 'original.png')
        if change=='diagnostic_tier':d=ev.read(p);d['metrics']['quant_label']='q8';p.write_text(json.dumps(d))
        elif change=='diagnostic_event':p.write_text(p.read_text().replace('q4','q8'))
        else:p.write_bytes(b'not a PNG')
        r['diagnostic_files'][p.name]=ev.file_hash(p)
    else:
        cs.install(root,m,r)
        if change=='unsealed_claim':
            path=root/'run/calibration-source/repair-claim.json';d=ev.unseal(ev.read(path));path.write_text(json.dumps(d))
            # Even a rebuilt bundle/hash cannot bless an unsealed new authority.
            bundle=root/'run/calibration-source/bundle.json';v=ev.unseal(ev.read(bundle));v['files']['repair-claim.json']=ev.file_hash(path);v['claim']['sha256']=ev.file_hash(path);bundle.write_text(json.dumps(ev.seal(v)));m['calibration_carry_forward']['sha256']=ev.file_hash(bundle);m['calibration_carry_forward']['claim']['sha256']=ev.file_hash(path)
        else:
            cell=next(c for c in ev.cells(m,'calibration') if c['config_id']=='qwen_image_2_1');p=root/'run'/('outputs' if change=='new_carried_image' else 'receipts')/(cell['id']+('.png' if change=='new_carried_image' else '.json'));p.parent.mkdir();p.write_bytes(b'forbidden')
        with pytest.raises(ValueError):cs.carried(m,root/'run')
        return
    with pytest.raises(ValueError):cs.install(root,m,r)


@pytest.mark.parametrize('change',[None,'request_only','wrong_job','wrong_backend','config_hash','dense_te','binary','raw_tier','reported_signature','conflicting_job_fields'])
def test_klein_metric_defect_requires_actual_closed_resolved_proof(repaired_v6_fixture,change):
    from native import klein_tier_proof
    import calibration_repair as cr
    f=repaired_v6_fixture;c=deepcopy(next(c for c in f['old']['configs'] if c['id']==cr.KLEIN))
    c['settings']['runtime_versions']=dict(sceneworks_source=__import__('bootstrap').SOURCE,inference=__import__('bootstrap').INFERENCE)
    c['files'].append(dict(path=c['settings']['binary'],sha256=c['settings']['binary_sha256']))
    e=f['root']/'repair-evidence';metrics=dict(quantLabel='bf16',quantBits=None);recipe=ev.read(e/'asset.recipe.json');log=e/'actual-worker.log'
    log.write_bytes((Path(__file__).parent/'fixtures/moss-klein/v6-worker.txt').read_bytes());job_id='job_1676d4441a51fbf3a368846cbb8de482'
    if change=='request_only':log.write_text('requested q4, no actual event')
    elif change=='wrong_job':log.write_text(log.read_text().replace(job_id,'job_other'))
    elif change=='wrong_backend':log.write_text(log.read_text().replace('candle','mlx'))
    elif change in ('config_hash','dense_te'):
        part='transformer' if change=='config_hash' else 'text_encoder';p=Path(c['snapshot'])/'q4'/part/'config.json';p.write_text('{}')
    elif change=='binary':c['settings']['binary_sha256']='0'*64
    elif change=='raw_tier':recipe['rawAdapterSettings']['mlxQuantize']=8
    elif change=='reported_signature':metrics['quantLabel']='q8'
    elif change=='conflicting_job_fields':
        lines=log.read_text().splitlines();i=next(i for i,line in enumerate(lines) if 'actual_tier' in line);row=json.loads(lines[i]);row['jobId']='other';lines[i]=json.dumps(row);log.write_text('\n'.join(lines))
    if change:
        with pytest.raises(ValueError):klein_tier_proof(c,job_id,metrics,recipe,log)
    else:
        proof=klein_tier_proof(c,job_id,metrics,recipe,log)
        assert proof['resolved_tier']=='q4' and proof['reported_quant_label']=='bf16' and proof['reported_quant_bits'] is None
        assert metrics==dict(quantLabel='bf16',quantBits=None)


@pytest.mark.parametrize('change',[None,'route','backend','raw_bits','component','binary','revision'])
def test_z_null_recipe_requires_packed_artifact_and_completed_exact_route(tmp_path,change):
    import native
    c=deepcopy(next(c for c in ev.template()['configs'] if c['id']=='z_image_turbo'));snap=tmp_path/'snapshot';snap.mkdir();c['snapshot']=str(snap)
    binary=tmp_path/'app.exe';binary.write_bytes(b'CPU fixture only')
    c['settings']=dict(binary=str(binary),binary_sha256=ev.file_hash(binary),runtime_versions=dict(sceneworks_source=native.FROZEN_SOURCE,inference=native.FROZEN_INFERENCE))
    c['files']=[dict(path=str(binary),sha256=ev.file_hash(binary))]
    for part in ('transformer','text_encoder'):
        p=snap/'q4'/part/'config.json';p.parent.mkdir(parents=True);p.write_bytes((Path(__file__).parent/'fixtures/moss-klein'/('z-'+part+'-config.json')).read_bytes());c['files'].append(dict(path=str(p),sha256=ev.file_hash(p)))
    events=[dict(event='image_memory_strategy_selected',job_id='exact_job',model='z_image_turbo',route='z_image_turbo',actual_tier='q4'),dict(event='image_inference_complete',jobId='exact_job',backend='candle')]
    metrics=dict(quantLabel='q4',quantBits=4);recipe=dict(model='z_image_turbo',rawAdapterSettings=dict(mlxQuantize=None,realModelInference=True))
    if change=='route':events[0]['job_id']='unrelated'
    elif change=='backend':events[1]['backend']='mlx'
    elif change=='raw_bits':recipe['rawAdapterSettings']['mlxQuantize']=4
    elif change=='component':(snap/'q4/text_encoder/config.json').write_text('{}')
    elif change=='binary':binary.write_bytes(b'drift')
    elif change=='revision':c['revision']='0'*40
    log=tmp_path/'worker.log';log.write_text('\n'.join(json.dumps(e) for e in events))
    if change:
        with pytest.raises(ValueError):native.z_tier_proof(c,'exact_job',metrics,recipe,log)
    else:
        proof=native.z_tier_proof(c,'exact_job',metrics,recipe,log)
        assert proof['resolved_tier']=='q4' and proof['reported_raw_mlx_quantize'] is None and recipe['rawAdapterSettings']['mlxQuantize'] is None


@pytest.mark.parametrize('failed',[False,True])
def test_fixed_v6_transport_finishes_exact_two_pending_cells_and_reports_origins(repaired_v6_fixture,monkeypatch,failed):
    import bootstrap as b
    import ownership as o
    import calibration_source as cs
    import run
    from types import SimpleNamespace
    f=repaired_v6_fixture;root=f['root'];m=f['manifest'];cs.install(root,m,f['request']);ev.write(root/'protocol.lock.json',ev.seal(m))
    calls=[]
    class Guard:
        def __init__(self,*a,**kw):self.prepared_root=root;self.unavailable_configs={}
        def check(self,*a,**kw):cs.check_carry_closure(m,root/'run',execution=True);return {'CPU_fixture':True}
    monkeypatch.setattr(b,'ExecutionGuard',Guard);monkeypatch.setattr(o,'ExecutionGuard',Guard)
    monkeypatch.setattr(ev,'validate_manifest',lambda manifest,**kw:manifest);monkeypatch.setattr(run,'validate_manifest',lambda *a,**kw:None)
    class Requests:
        pending=None
        def put(self,item):self.pending=item
    def launch(config,*a,**kw):
        assert config['id'] in ('flux2_klein_9b','z_image_turbo');request=Requests()
        def replies():
            yield ('ready',.01)
            if failed and config['id']=='z_image_turbo':
                yield ('error','RuntimeError: CPU deliberate native failure');return
            cell,path=request.pending;Path(path).write_bytes(png(cell['width'],cell['height']))
            yield ('result',dict(f['metrics'],load_seconds=.01))
        return SimpleNamespace(),request,replies()
    monkeypatch.setattr(run,'launch',launch);monkeypatch.setattr(run,'receive',lambda p,q,*a:(next(q),42));monkeypatch.setattr(run,'finish',lambda *a:None);monkeypatch.setattr(run,'stop',lambda *a:None)
    class Process:
        returncode=0
        def __init__(self,args,**kw):
            config=args[args.index('--config')+1];calls.append(config);run.execute(root/'protocol.lock.json',root/'run',config,'calibration')
        def poll(self):return 0
    monkeypatch.setattr(b.subprocess,'Popen',Process);monkeypatch.setattr(b,'kill_owned',lambda *a:None)
    deadline='2030-01-01T00:00:00+00:00'
    lease=dict(id=m['ownership']['lease'],owner=m['ownership']['responsible_chat'],host=m['hardware']['host'],gpu_uuid=o.RUNNER_GPU_UUID,expires_utc=deadline,external_owners_reconciled=True,desktop_baseline=[],baseline_max_memory_mib=1000,baseline_max_utilization_percent=2)
    b.execute_phase(root,dict(protocol_sha256=ev.read(root/'protocol.lock.json')['sha256'],deadline_utc=deadline,lease=lease),'calibration')
    assert calls==list(ev.CONFIG_IDS[4:]) and len(list((root/'run'/'outputs').glob('*.png')))==2-int(failed)
    _,rows=ev.receipts(root/'protocol.lock.json',root/'run','calibration');assert len(rows)==6 and sum(r['status']=='ok' for r in rows.values())==6-int(failed)
    assert len({r['lock_sha256'] for r in rows.values()})==3
    estimate=ev.read(root/'calibration-estimate.json')
    assert estimate['calibration_outputs']==6-int(failed) and estimate['calibration_attempts']==10 and estimate['historical_failed_native_attempts']==4
    assert all(estimate['conditions'][key]['measurement_origin']=='carried_original' for key in (*cs.MOSS,'qwen_image_2_1'))
    assert all(estimate['conditions'][key]['measurement_origin']=='current_campaign' for key in ev.CONFIG_IDS[4:])
    assert estimate['accepted_calibration_outputs']==6-int(failed) and estimate['known_generated_calibration_outputs']==7-int(failed) and estimate['excluded_failed_generated_outputs']==1
    assert estimate['carried_calibration_outputs']==4 and estimate['current_calibration_output_cap']==2


@pytest.mark.parametrize('change',['misplaced_branch','missing_helper','duplicate_branch','unrelated_method','global','nested_constant'])
def test_native_compatibility_projection_cannot_mask_misplaced_code(tmp_path,change):
    import calibration_repair as cr
    path=tmp_path/'native.py';text=(ROOT/'native.py').read_text()
    if change=='misplaced_branch':text=text.replace("    def call(self,method,path,body=None,raw=False):", "    def call(self,method,path,body=None,raw=False):\n        if config['id']=='z_image_turbo':\n            tier_proof=z_tier_proof(config,job_id,metrics,job['result']['assets'][0]['recipe'],self.context['worker_log'])")
    elif change=='nested_constant':text=text.replace('    def call(self,method,path,body=None,raw=False):',"    def call(self,method,path,body=None,raw=False):\n        FROZEN_SOURCE = 'ba2954f47bfd75a3186002e3cef859c4e4e1d37d'")
    elif change=='missing_helper':text=text.replace('def z_tier_proof(', 'def renamed_z_tier_proof(')
    elif change=='duplicate_branch':text=text.replace('        tier_proof=None','        tier_proof=None\n        tier_proof=None')
    elif change=='unrelated_method':text=text.replace('timeout=30','timeout=31')
    else:text+='\nUNREVIEWED_NATIVE_GLOBAL=1\n'
    path.write_text(text)
    try:projected=cr.native_source(path)
    except ValueError:return
    assert projected!=cr.native_source(ROOT/'native.py')
