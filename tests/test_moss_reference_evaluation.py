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
