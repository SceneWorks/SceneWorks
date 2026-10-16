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
    assert all(c['width']==c['height']==1024 for c in calibration)
    first=m['corpus']['cases'][0]
    assert all(c['width']==first['width'] and c['height']==first['height'] for c in calibration)
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
