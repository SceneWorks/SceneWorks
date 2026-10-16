"""Controlled CPU fixtures only: none of these files are campaign evidence."""
from copy import deepcopy
import importlib.util
import json
import os
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
    assert all(c['width']==c['height']==512 for c in calibration)
    assert all(c['phase']=='scored' for c in scored)
    assert {c['config_id'] for c in scored}==set(ev.CONFIG_IDS)


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


def test_device_lease_requires_live_owner_correct_freeze_and_unexpired_identity(tmp_path):
    import socket
    from run import validate_lease
    manifest=ev.template();manifest['ownership'].update(lease='fixture-lease',responsible_chat='fixture-chat');manifest['hardware'].update(host=socket.gethostname(),gpu_uuid='GPU-FIXTURE')
    path=tmp_path/'lease.json';lock_hash='a'*64
    lease=dict(lease_id='fixture-lease',host=socket.gethostname(),gpu_uuid='GPU-FIXTURE',protocol_sha256=lock_hash,deadline_epoch=200.,owner_pid=42,responsible_chat='fixture-chat',job='FIXTURE')
    env=dict(MOSS_DEVICE_LOCK=str(path),MOSS_LEASE_ID='fixture-lease',MOSS_PROTOCOL_SHA256=lock_hash)
    ev.write(path,lease)
    assert validate_lease(manifest,lock_hash,env,now=100.,ancestor_pids={42})['owner_pid']==42
    for key,value in [('gpu_uuid','GPU-OTHER'),('deadline_epoch',99.),('owner_pid',43),('protocol_sha256','b'*64),('responsible_chat','other')]:
        mutant=dict(lease,**{key:value});path.write_text(json.dumps(mutant))
        with pytest.raises(ValueError):validate_lease(manifest,lock_hash,env,now=100.,ancestor_pids={42})
    with pytest.raises(ValueError,match='coordinator-held'):validate_lease(manifest,lock_hash,{},now=100.,ancestor_pids={42})
