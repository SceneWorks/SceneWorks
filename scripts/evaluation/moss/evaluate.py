#!/usr/bin/env python3
"""CPU-only planning, integrity validation, blind pairing and evidence reporting.

Rendering is an explicit separate command. All run artifacts belong outside git.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import socket
import json
import math
from pathlib import Path
import random
import struct
import sys
import zlib

ROOT = Path(__file__).resolve().parent
CONFIG_IDS = ('moss_base', 'moss_sft', 'moss_dmd', 'qwen_image_2_1', 'flux2_klein_9b', 'z_image_turbo')
REVISIONS = {
 'moss_base': ('OpenMOSS-Team/MOSS-Image-Base', '1ceede80badddd6a7812069619edaa5633587192'),
 'moss_sft': ('OpenMOSS-Team/MOSS-Image-SFT', 'fc5de0d56fc910cbca6f6625a70f5f582b26deb1'),
 'moss_dmd': ('lodestones/MOSS-Image-SFT-8-steps-dmd', '5ff314b4c1470876b114b810146085db8a181a6e'),
 'qwen_image_2_1': ('SceneWorks/qwen-image-2-1-mlx', 'ef03535eac756dd21feb7ac47c4f2df3e9b5b3c0'),
 'flux2_klein_9b': ('SceneWorks/flux2-klein-9b-mlx', '1902693279fcfb828919370dfac2b8922d99499a'),
 'z_image_turbo': ('SceneWorks/z-image-turbo-mlx', 'bb2bc9893b3c49ae96c813350775f791a2e8bc80'),
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    require(not path.exists(), f'refusing to overwrite {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def seal(value):
    return dict(value, sha256=digest(value))


def unseal(value):
    body = {k: v for k, v in value.items() if k != 'sha256'}
    require(value.get('sha256') == digest(body), 'evidence digest mismatch')
    return body


def corpus():
    value = read(ROOT / 'corpus.json')
    cases = value['cases']
    require(len(cases) == 24 and len({c['id'] for c in cases}) == 24, 'require 24 unique prompts')
    require(sorted(Counter(c['scenario'] for c in cases).values()) == [2] * 12, 'require 12 scenarios x2 prompts')
    require(len(value['seeds']) == 2 and len(set(value['seeds'])) == 2, 'require two distinct seeds')
    for case in cases:
        require(case['prompt_raw'] and case['prompt_enhanced'] and case['requirements'], 'missing corpus annotation')
        require(case['width'] % 16 == case['height'] % 16 == 0, 'canvas must be multiple of16')
        for box in case['boxes']:
            x1, y1, x2, y2 = box['box']
            require(0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1, 'invalid ground truth box')
    return value


def template():
    configs = []
    for key in CONFIG_IDS:
        repo, revision = REVISIONS[key]
        configs.append(dict(id=key, repo=repo, revision=revision,
            precision='fp32' if key == 'moss_dmd' else ('bf16' if key.startswith('moss') else 'q4'),
            steps=8 if key in ('moss_dmd', 'z_image_turbo') else (4 if key == 'flux2_klein_9b' else (40 if key == 'qwen_image_2_1' else 50)),
            guidance=3.0 if key in ('moss_base','moss_sft') else (0.0 if key in ('moss_dmd','z_image_turbo') else 1.0),
            prompt_mode='raw', negative_prompt='', batch_size=1, snapshot=None,
            # Every local weight/config/tokenizer/code input is frozen by bytes.
            files=[], runtime_command=[sys.executable, str(ROOT / 'run.py')], settings={}, unavailable=None))
    return dict(schema_version=1, corpus=corpus(), configs=configs,
        runtime=dict(python=None, packages=None, os=None, cuda=None, driver=None),
        hardware=dict(host=None, gpu_uuid=None, gpu_name=None, gpu_memory_bytes=None, host_memory_bytes=None),
        ownership=dict(responsible_chat=None, lease=None, job=None, timeout_seconds=1800, resume=None, next_action=None),
        thresholds=dict(text_similarity=0.95, box_iou=0.5, dmd_prediction_max_abs=0.01, dmd_prediction_mean_abs=0.001), scoring_protocol='manual transcription and normalized boxes; blind visual rubric v1',
        visual_rubric={'visual_quality':'1 unusable defects; 3 visible defects but usable; 5 no material defects', 'instruction_following':'1 misses task; 3 partially follows; 5 all requirements', 'composition':'1 incoherent; 3 readable; 5 clear hierarchy and balance'},
        condition='matched corpus canvas; per-model declared steps and precision; raw and enhanced prompt bytes frozen separately')


def validate_manifest(manifest):
    require(manifest['corpus'] == corpus(), 'corpus changed: new freeze required')
    require(tuple(c['id'] for c in manifest['configs']) == CONFIG_IDS, 'six ordered configuration identities required')
    for category in ('runtime', 'hardware', 'ownership'):
        require(all(v is not None and v != '' for v in manifest[category].values()), f'incomplete {category}')
    require(0 < manifest['ownership']['timeout_seconds'] <= 86400, 'invalid timeout')
    require(all(0 < v <= 1 and math.isfinite(v) for v in manifest['thresholds'].values()), 'invalid scoring thresholds')
    if any(not c['unavailable'] for c in manifest['configs']):
        require(manifest['runtime']['python']==sys.version,'Python runtime version changed')
        require(manifest['hardware']['host'].casefold()==socket.gethostname().casefold(),'host identity changed')
        packages=manifest['runtime']['packages']
        require(isinstance(packages,dict) and packages,'runtime package lock missing')
        for name,version in packages.items():
            require(importlib.metadata.version(name)==version,f'runtime dependency changed: {name}')
    for config in manifest['configs']:
        require((config['repo'], config['revision']) == REVISIONS[config['id']], 'unexpected model revision')
        require(config['prompt_mode'] in ('raw','enhanced'), 'prompt_mode must be declared')
        expected=next(c for c in template()['configs'] if c['id']==config['id'])
        require(all(config[k]==expected[k] for k in ('steps','guidance','precision','batch_size','negative_prompt')), 'published model settings changed')
        if config['unavailable']:
            require(all(config['unavailable'].get(k) for k in ('reason', 'reproduce', 'evidence_path', 'evidence_sha256')), 'unavailable config needs reproducible evidence')
            require(file_hash(config['unavailable']['evidence_path']) == config['unavailable']['evidence_sha256'], 'unavailable evidence hash mismatch')
        else:
            if not config['id'].startswith('moss_'):
                require(config['settings']['device_id']==manifest['hardware']['gpu_uuid'],'native physical device differs from freeze')
            require(config['snapshot'] and config['files'] and config['runtime_command'], f"unprepared {config['id']}")
            require(all(Path(f['path']).is_absolute() for f in config['files']), 'input paths must be absolute')
            require({f['path']:f['sha256'] for f in config['files']} == {f['path']:f['sha256'] for f in inventory(config)}, 'input closure incomplete or changed')
            for item in config['files']:
                path = Path(item['path'])
                require(path.is_file() and not path.is_symlink(), f'invalid input {path}')
                require(item['sha256'] == file_hash(path), f'input hash mismatch {path}')
    return manifest


def inventory(config):
    """Hash all declared model directories, runtime scripts and local binary by bytes."""
    paths=set(ROOT.glob('*.py'))
    roots=[config['snapshot']]
    settings=config['settings']
    if config['id'].startswith('moss_'):
        roots.append(settings.get('shared_snapshot',config['snapshot']))
        paths.add(Path(settings['pipeline_code']))
        if settings.get('adapter_validation'):
            paths.update(Path(settings[k]) for k in ('lora_path','replacement_path'))
    else:
        paths.add(Path(settings['binary']))
    for root in roots:
        require(root and Path(root).is_dir(), f'missing snapshot {root}')
        paths.update(p for p in Path(root).rglob('*') if p.is_file())
    return [dict(path=str(p.resolve()),sha256=file_hash(p.resolve()),bytes=p.stat().st_size) for p in sorted(paths)]


def freeze(source, destination):
    manifest = validate_manifest(read(source))
    write(destination, seal(manifest))


def load_lock(path, verify_inputs=False):
    manifest = unseal(read(path))
    if verify_inputs:
        validate_manifest(manifest)
    return manifest


def cells(manifest, phase='scored'):
    entries = manifest['corpus']['cases']
    seeds = manifest['corpus']['seeds']
    for config in manifest['configs']:
        for case in (entries[:1] if phase == 'calibration' else entries):
            for seed in (seeds[:1] if phase == 'calibration' else seeds):
                yield dict(id=f"{phase}.{config['id']}.{case['id']}.{seed}", phase=phase,
                    config_id=config['id'], case_id=case['id'], scenario=case['scenario'], seed=seed,
                    width=512 if phase == 'calibration' else case['width'],
                    height=512 if phase == 'calibration' else case['height'],
                    prompt=case['prompt_' + config['prompt_mode']], negative_prompt=config['negative_prompt'])


def image_dimensions(path):
    data=Path(path).read_bytes()
    require(data[:8]==b'\x89PNG\r\n\x1a\n','output must be PNG')
    offset=8;compressed=[];width=height=channels=None;ended=False
    while offset+12<=len(data):
        size=struct.unpack('>I',data[offset:offset+4])[0];kind=data[offset+4:offset+8]
        body=data[offset+8:offset+8+size]
        require(len(body)==size and offset+12+size<=len(data),'truncated PNG')
        crc=struct.unpack('>I',data[offset+8+size:offset+12+size])[0]
        require(zlib.crc32(kind+body)&0xffffffff==crc,'PNG CRC mismatch')
        if kind==b'IHDR':
            require(offset==8 and size==13,'invalid PNG header')
            width,height,depth,color,compression,filter_method,interlace=struct.unpack('>IIBBBBB',body)
            require(0<width<=8192 and 0<height<=8192 and depth==8 and color in (2,6) and compression==filter_method==interlace==0,'unsupported PNG encoding')
            channels={2:3,6:4}[color]
        elif kind==b'IDAT':compressed.append(body)
        elif kind==b'IEND':ended=True;require(size==0 and offset+12==len(data),'invalid PNG end');break
        offset+=12+size
    require(ended and width and compressed,'incomplete PNG')
    decoder=zlib.decompressobj();expected=height*(1+width*channels)
    pixels=decoder.decompress(b''.join(compressed),expected+1)
    require(decoder.eof and not decoder.unused_data and len(pixels)==expected,'invalid PNG pixels')
    require(all(pixels[i] in range(5) for i in range(0,expected,1+width*channels)),'invalid PNG row filter')
    return width,height


def artifact(root, item):
    path = (root / item['path']).resolve()
    require(path.is_relative_to(root.resolve()) and path.is_file() and not (root / item['path']).is_symlink(), 'artifact escapes run or missing')
    require(file_hash(path) == item['sha256'], f'artifact hash mismatch {path}')
    return path


def validate_receipt(root, lock_hash, cell, receipt):
    r = unseal(receipt)
    require(r['lock_sha256'] == lock_hash and r['cell'] == cell, 'receipt identity mismatch')
    require(r['status'] in ('ok', 'failed', 'unavailable', 'timeout'), 'invalid status')
    require(all(r['ownership'].get(k) for k in ('responsible_chat','job','revision','timeout_seconds','resume','next_action')), 'missing resume ownership')
    require(r['elapsed_seconds'] >= 0 and math.isfinite(r['elapsed_seconds']), 'invalid elapsed time')
    if r['status'] != 'ok':
        require(r.get('error') and r.get('reproduce'), 'failure needs reason and reproduction')
        artifact(root, r['log'])
        return r
    path = artifact(root, r['output'])
    require(image_dimensions(path) == (cell['width'],cell['height']), 'wrong canvas')
    artifact(root, r['log'])
    metrics = r['metrics']
    for field in ('generation_seconds','peak_gpu_bytes','peak_host_rss_bytes','download_bytes'):
        require(type(metrics.get(field)) in (int,float) and math.isfinite(metrics[field]) and metrics[field] >= 0, f'missing metric {field}')
    condition=metrics.get('latency_condition')
    require(condition in ('cold_process','resident_model'), 'missing latency condition')
    measured=metrics.get('cold_seconds' if condition=='cold_process' else 'warm_seconds')
    require(type(measured) in (int,float) and math.isfinite(measured) and measured>=0, 'missing latency measurement')
    require(metrics['peak_host_rss_bytes']>0 and metrics['peak_gpu_bytes']>0 and metrics.get('gpu_memory_method'), 'missing peak memory evidence')
    require(metrics.get('component_dtypes') and metrics.get('sampler') and metrics.get('runtime_versions'), 'missing runtime/sampler evidence')
    require(metrics.get('pixels_finite') is True, 'finite pixels not verified')
    return r


def receipts(lock, run, phase='scored', complete=True):
    run = Path(run)
    lock_doc = read(lock)
    manifest = load_lock(lock)
    expected = list(cells(manifest, phase))
    results = {}
    for cell in expected:
        path = run / 'receipts' / (cell['id'] + '.json')
        if not path.exists():
            require(not complete, f"missing cell {cell['id']}")
            continue
        results[cell['id']] = validate_receipt(run, lock_doc['sha256'], cell, read(path))
    return manifest, results


def blind_pairs(lock, run, destination):
    manifest, results = receipts(lock, run)
    all_cells = {c['id']:c for c in cells(manifest)}
    rng = random.Random(int(read(lock)['sha256'][:16],16))
    pairs=[]; mapping=[]
    for case in manifest['corpus']['cases']:
        for seed in manifest['corpus']['seeds']:
            for moss in CONFIG_IDS[:3]:
                for baseline in CONFIG_IDS[3:]:
                    ids=[f'scored.{moss}.{case["id"]}.{seed}',f'scored.{baseline}.{case["id"]}.{seed}']
                    rng.shuffle(ids)
                    pair_id=hashlib.sha256((read(lock)['sha256']+'|'.join(ids)).encode()).hexdigest()[:24]
                    arms=[]
                    for arm,cell_id in zip(('A','B'),ids):
                        r=results[cell_id]
                        # Neutral filenames prevent model leakage through URLs; copy bytes once.
                        public_path=Path(destination).parent/'blind-images'/f'{pair_id}-{arm}.png'
                        if r['status']=='ok':
                            source=artifact(Path(run),r['output'])
                            require(not public_path.exists(),'blind export exists')
                            public_path.parent.mkdir(exist_ok=True,parents=True)
                            public_path.write_bytes(source.read_bytes())
                        arms.append(dict(arm=arm,image=str(public_path) if r['status']=='ok' else None,sha256=r['output']['sha256'] if r['status']=='ok' else None,status='available' if r['status']=='ok' else 'unavailable'))
                    pairs.append(dict(id=pair_id,case_id=case['id'],seed=seed,prompt=case['prompt_raw'],requirements=case['requirements'],arms=arms))
                    mapping.append(dict(id=pair_id,A=ids[0],B=ids[1]))
    require(len(pairs)==432,'wrong pair count')
    rng.shuffle(pairs)
    write(destination, seal(dict(lock_sha256=read(lock)['sha256'],pairs=pairs)))
    write(str(destination)+'.private.json',seal(dict(lock_sha256=read(lock)['sha256'],mapping=mapping)))


def edit_similarity(expected, observed):
    # Unicode code points, case and punctuation retained; whitespace collapsed only.
    expected=' '.join(expected.split());observed=' '.join(observed.split())
    previous=list(range(len(observed)+1))
    for i,a in enumerate(expected,1):
        current=[i]
        for j,b in enumerate(observed,1):
            current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(a!=b)))
        previous=current
    return 1-previous[-1]/max(1,len(expected),len(observed))


def iou(a,b):
    require(len(b)==4 and all(type(v) in (float,int) and math.isfinite(v) for v in b), 'invalid observed box')
    require(0<=b[0]<b[2]<=1 and 0<=b[1]<b[3]<=1,'invalid observed box')
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    area=lambda q:(q[2]-q[0])*(q[3]-q[1])
    return intersection/(area(a)+area(b)-intersection)


def objective_scores(case, annotation):
    require(annotation.get('annotator') and annotation.get('method') and annotation.get('tool_revision'), 'annotation provenance missing')
    require(len(annotation['text'])==len(case['expected_text']), 'text annotation length mismatch')
    require(set(annotation['boxes'])=={b['id'] for b in case['boxes']}, 'box annotation missing or extra IDs')
    return dict(text=[edit_similarity(e,o) for e,o in zip(case['expected_text'],annotation['text'])],
        box=[iou(b['box'],annotation['boxes'][b['id']]) if annotation['boxes'][b['id']] is not None else 0 for b in case['boxes']])


def report(lock,run,pairs_path,judgments_path,annotations_path):
    manifest, results=receipts(lock,run)
    public=unseal(read(pairs_path));private=unseal(read(str(pairs_path)+'.private.json'))
    require(public['lock_sha256']==private['lock_sha256']==read(lock)['sha256'],'pair freeze mismatch')
    pair_ids={p['id'] for p in public['pairs']}
    require(len(pair_ids)==432 and {p['id'] for p in private['mapping']}==pair_ids,'incomplete pair mapping')
    expected_matchups={frozenset((f'scored.{m}.{case["id"]}.{seed}',f'scored.{b}.{case["id"]}.{seed}')) for case in manifest['corpus']['cases'] for seed in manifest['corpus']['seeds'] for m in CONFIG_IDS[:3] for b in CONFIG_IDS[3:]}
    observed_matchups=[frozenset((p['A'],p['B'])) for p in private['mapping']]
    require(len(observed_matchups)==432 and set(observed_matchups)==expected_matchups,'missing or duplicated model/case pair')
    mapping_by_id={p['id']:p for p in private['mapping']}
    for pair in public['pairs']:
        mapping=mapping_by_id[pair['id']]
        for arm in pair['arms']:
            row=results[mapping[arm['arm']]]
            if row['status']=='ok':
                require(arm.get('sha256')==row['output']['sha256'] and file_hash(arm['image'])==arm['sha256'],'blind image hash mismatch')
            else:require(arm['image'] is None and arm['status']=='unavailable','unavailable blind arm has image')
    judgments=read(judgments_path); annotations=read(annotations_path)
    require(len(judgments)==432 and {j['pair_id'] for j in judgments}==pair_ids,'require432 unique blind judgments')
    votes=Counter(); comparisons=defaultdict(list)
    for j in judgments:
        require(j.get('reviewer_kind') in ('human','machine_assisted') and j.get('judge_version') and j.get('reviewer') and j.get('notes') and j.get('uncertainty') in ('low','medium','high'),'blind review provenance missing')
        require(j['winner'] in ('A','B','tie','unavailable'),'invalid judgment')
        mapping=next(p for p in private['mapping'] if p['id']==j['pair_id'])
        unavailable=any(results[mapping[a]]['status']!='ok' for a in ('A','B'))
        require((j['winner']=='unavailable')==unavailable,'missing output must be unavailable')
        require(all(type(j.get('rubric',{}).get(k)) is int and 1<=j['rubric'][k]<=5 for k in ('visual_quality','instruction_following','composition')) if not unavailable else True,'missing visual rubric')
        if j['winner'] in ('A','B'):votes[mapping[j['winner']].split('.')[1]]+=1
        comparisons[results[mapping['A']]['cell']['scenario']].append(j)
    output=[]
    for case in manifest['corpus']['cases']:
        for config in manifest['configs']:
            rows=[r for r in results.values() if r['cell']['case_id']==case['id'] and r['cell']['config_id']==config['id']]
            scores=[]
            for row in rows:
                if row['status']=='ok':
                    ann=annotations.get(row['cell']['id'])
                    require(ann and ann['output_sha256']==row['output']['sha256'],'missing or mismatched annotation')
                    scores.append(objective_scores(case,ann))
            output.append(dict(case_id=case['id'],scenario=case['scenario'],config_id=config['id'],statuses=dict(Counter(r['status'] for r in rows)),objective=scores,
                costs=[r['metrics'] for r in rows if r['status']=='ok']))
    dmd=[r for r in results.values() if r['cell']['config_id']=='moss_dmd' and r['status']=='ok']
    parity_complete=len(dmd)==48 and all(len(r['metrics']['sampler'].get('parity',[]))==8 and all(p['candidate_finite'] and p['max_abs']<=manifest['thresholds']['dmd_prediction_max_abs'] and p['mean_abs']<=manifest['thresholds']['dmd_prediction_mean_abs'] for p in r['metrics']['sampler']['parity']) for r in dmd)
    usable_pairs=sum(j['winner']!='unavailable' for j in judgments)
    all_outputs=len(results)==288 and all(r['status']=='ok' for r in results.values())
    return dict(status='evidence assembled; analyst shortlist and Michael inclusion decision remain pending', rows=output,
        recommendation_eligible=all_outputs and usable_pairs==432 and parity_complete,
        dmd_adapter_prediction_parity_complete=parity_complete, usable_blind_pairs=usable_pairs,
        blind_wins=dict(votes),blind_pairs=432,missing_outputs=sum(r['status']!='ok' for r in results.values()),
        uncertainty=dict(Counter(j['uncertainty'] for j in judgments)),thresholds=manifest['thresholds'],
        conclusions='No model superiority or native adoption inferred automatically; analyze per scenario, failures and quality/cost tradeoffs.')


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    x=sub.add_parser('template');x.add_argument('output')
    x=sub.add_parser('inventory');x.add_argument('manifest');x.add_argument('output')
    x=sub.add_parser('freeze');x.add_argument('manifest');x.add_argument('output')
    x=sub.add_parser('plan');x.add_argument('--lock');x.add_argument('--output')
    x=sub.add_parser('validate');x.add_argument('lock');x.add_argument('run');x.add_argument('--phase',choices=('scored','calibration'),default='scored')
    x=sub.add_parser('pairs');x.add_argument('lock');x.add_argument('run');x.add_argument('output')
    x=sub.add_parser('report');x.add_argument('lock');x.add_argument('run');x.add_argument('pairs');x.add_argument('judgments');x.add_argument('annotations');x.add_argument('output')
    args=p.parse_args()
    if args.action=='template':write(args.output,template())
    elif args.action=='inventory':
        m=read(args.manifest)
        for c in m['configs']:
            if not c['unavailable']:c['files']=inventory(c)
        write(args.output,m)
    elif args.action=='freeze':freeze(args.manifest,args.output)
    elif args.action=='plan':
        m=load_lock(args.lock) if args.lock else template()
        value=dict(scored=list(cells(m)),calibration=list(cells(m,'calibration')),native_probe_cap=24,blind_pairs=432)
        if args.output:write(args.output,value)
        else:print(json.dumps(dict(scored=len(value['scored']),calibration_cap=len(value['calibration']),blind_pairs=432,native_probe_cap=24)))
    elif args.action=='validate':
        _,rows=receipts(args.lock,args.run,args.phase);print(json.dumps(dict(cells=len(rows),statuses=dict(Counter(r['status'] for r in rows.values())))))
    elif args.action=='pairs':blind_pairs(args.lock,args.run,args.output)
    elif args.action=='report':write(args.output,report(args.lock,args.run,args.pairs,args.judgments,args.annotations))


if __name__=='__main__':
    try:main()
    except (ValueError,KeyError,OSError,TypeError) as exc:
        print(f'evidence rejected: {exc}',file=sys.stderr);sys.exit(1)
