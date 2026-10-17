"""One fixed v6 Klein reporting-defect repair, not a general retry/chain API."""
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil

from calibration_source import MOSS, DIRECTORY, need, normalized, compatible, config_binding

KLEIN_HELPER_SOURCE='''def klein_tier_proof(config, job_id, metrics, recipe, worker_log):
    """Independent resolved-route proof for the frozen Windows metrics defect.

    A requested tier is insufficient. Bind the closed artifact inventory/configs,
    frozen binary/source, job-correlated resolved strategy and Candle completion.
    Raw metrics remain unchanged; this is indirect source/bytes proof, not an
    executed tensor dtype receipt.
    """
    from pathlib import Path
    from evaluate import file_hash, read
    def need(ok, message):
        if not ok: raise ValueError(message)
    settings=config['settings'];versions=settings['runtime_versions']
    need(config['id']=='flux2_klein_9b' and config['precision']=='q4' and
         versions['sceneworks_source']==FROZEN_SOURCE and versions['inference']==FROZEN_INFERENCE,
         'Klein defect proof requires exact frozen source/pin/q4')
    snapshot=Path(config['snapshot'])
    need(config['revision']=='1902693279fcfb828919370dfac2b8922d99499a', 'Klein proof revision mismatch')
    inventory={f['path']:f['sha256'] for f in config['files']}
    configs={}
    for name,sha in (('transformer',KLEIN_TRANSFORMER_CONFIG),('text_encoder',KLEIN_TEXT_ENCODER_CONFIG)):
        path=snapshot/'q4'/name/'config.json'
        need(inventory.get(str(path))==sha and file_hash(path)==sha,'Klein actual artifact config/hash mismatch')
        configs[name]=read(path)
    need(configs['transformer'].get('quantization',{}).get('bits')==4 and not configs['text_encoder'].get('quantization') and configs['text_encoder'].get('dtype')=='bfloat16','Klein packed DiT/dense TE contract mismatch')
    need(inventory.get(settings['binary'])==settings['binary_sha256'] and file_hash(settings['binary'])==settings['binary_sha256'],'Klein frozen binary proof mismatch')
    events=[]
    for line in Path(worker_log).read_text(encoding='utf-8',errors='strict').splitlines():
        try: event=json.loads(line)
        except json.JSONDecodeError: continue
        need(not ('jobId' in event and 'job_id' in event) or event['jobId']==event['job_id'],'conflicting native event job identities')
        if event.get('jobId',event.get('job_id'))==job_id:events.append(event)
    strategy=[e for e in events if e.get('event')=='image_memory_strategy_selected' and e.get('job_id')==job_id]
    need(len(strategy)==1 and strategy[0].get('model')==config['id'] and strategy[0].get('route')==config['id'] and strategy[0].get('actual_tier')=='q4','missing exact-job resolved Klein q4 event')
    need(any(e.get('event')=='image_inference_complete' and e.get('backend')=='candle' for e in events),'missing exact-job completed Candle inference')
    need(recipe.get('model')==config['id'] and recipe['rawAdapterSettings'].get('mlxQuantize')==4 and recipe['rawAdapterSettings'].get('realModelInference') is True,'Klein actual recipe tier mismatch')
    need((metrics.get('quantLabel'),metrics.get('quantBits'))==('bf16',None),'unexpected frozen Klein metrics defect signature')
    return dict(resolved_tier='q4',reported_quant_label=metrics['quantLabel'],reported_quant_bits=metrics.get('quantBits'),
        transformer_config_sha256=KLEIN_TRANSFORMER_CONFIG,text_encoder_config_sha256=KLEIN_TEXT_ENCODER_CONFIG,
        binary_sha256=settings['binary_sha256'],revision=config['revision'],source=FROZEN_SOURCE,inference=FROZEN_INFERENCE,
        strategy_event=strategy[0],job_id=job_id,method='closed staged bytes + pinned packed loader + actual job/route/backend; runtime component dtypes not exposed')'''
Z_HELPER_SOURCE='''def z_tier_proof(config,job_id,metrics,recipe,worker_log):
    """Frozen packed Z recipe-null defect; require actual bytes/route, never request."""
    from pathlib import Path
    from evaluate import file_hash,read
    def need(ok,message):
        if not ok:raise ValueError(message)
    settings=config['settings'];versions=settings['runtime_versions']
    need(config['id']=='z_image_turbo' and config['precision']=='q4' and config['revision']=='bb2bc9893b3c49ae96c813350775f791a2e8bc80' and versions['sceneworks_source']==FROZEN_SOURCE and versions['inference']==FROZEN_INFERENCE,'Z packed proof requires exact frozen source/pin/artifact')
    inventory={f['path']:f['sha256'] for f in config['files']}
    for part,sha in (('transformer',Z_TRANSFORMER_CONFIG),('text_encoder',Z_TEXT_ENCODER_CONFIG)):
        path=Path(config['snapshot'])/'q4'/part/'config.json'
        need(inventory.get(str(path))==sha and file_hash(path)==sha and read(path).get('quantization',{}).get('bits')==4,'Z packed component config/hash mismatch')
    need(inventory.get(settings['binary'])==settings['binary_sha256'] and file_hash(settings['binary'])==settings['binary_sha256'],'Z frozen binary mismatch')
    events=[]
    for line in Path(worker_log).read_text(encoding='utf-8').splitlines():
        try:row=json.loads(line)
        except json.JSONDecodeError:continue
        need(not ('jobId' in row and 'job_id' in row) or row['jobId']==row['job_id'],'conflicting native event job identities')
        if row.get('jobId',row.get('job_id'))==job_id:events.append(row)
    strategy=[e for e in events if e.get('event')=='image_memory_strategy_selected' and e.get('job_id')==job_id]
    need(len(strategy)==1 and strategy[0].get('model')==config['id'] and strategy[0].get('route')==config['id'] and strategy[0].get('actual_tier')=='q4','missing exact-job resolved Z q4 event')
    need(any(e.get('event')=='image_inference_complete' and e.get('backend')=='candle' for e in events),'missing completed exact-job Candle Z inference')
    need(metrics.get('quantLabel')=='q4' and metrics.get('quantBits')==4 and recipe.get('model')==config['id'] and recipe['rawAdapterSettings'].get('mlxQuantize') is None and recipe['rawAdapterSettings'].get('realModelInference') is True,'unexpected frozen Z recipe-null defect signature')
    return dict(resolved_tier='q4',reported_raw_mlx_quantize=None,transformer_config_sha256=Z_TRANSFORMER_CONFIG,text_encoder_config_sha256=Z_TEXT_ENCODER_CONFIG,binary_sha256=settings['binary_sha256'],revision=config['revision'],source=FROZEN_SOURCE,inference=FROZEN_INFERENCE,job_id=job_id,strategy_event=strategy[0],method='closed staged packed configs + pinned loader + exact job route/backend; runtime component dtypes not exposed')'''
KLEIN_CONSTANTS={'KLEIN_TRANSFORMER_CONFIG': '1ef42c2f91ff0060b147f7ff6790b5dc6f2e4c14d31dbeae3005ff339cea7922', 'KLEIN_TEXT_ENCODER_CONFIG': '4b8a90efaacbc0eaa1f3f3a2006335bbbbad2683fbd0688ada2c436e71342134', 'FROZEN_SOURCE': 'ba2954f47bfd75a3186002e3cef859c4e4e1d37d', 'FROZEN_INFERENCE': '91ffb4a819aa2990c652db2d881ba0b9b10126ce', 'Z_TRANSFORMER_CONFIG': '3a634d2dd4c84aa9ae511ad88a61fa2a427c05f61940fda45a15aa97cb20a7c6', 'Z_TEXT_ENCODER_CONFIG': '6237fccdd922663bd3d8bce89bc390ca073ca22c10e8b3e1fe81a58ccf2169db'}
V6='sc-26322-reference-v6'
QWEN='qwen_image_2_1'
KLEIN='flux2_klein_9b'
CARRIED=(*MOSS,QWEN)
DIAGNOSTIC_NAMES={'selected-state.json','selected-index.json','snapshot-index.json','original.png','asset.recipe.json','asset.sceneworks.json','worker.log'}


def archive_files(root):
    from ownership import physical
    physical(root)
    for p in root.rglob('*'):physical(p)
    return {p.relative_to(root).as_posix():p for p in root.rglob('*') if p.is_file()}


def qwen_config(manifest,root):
    c=next(c for c in manifest['configs'] if c['id']==QWEN)
    body={k:v for k,v in c.items() if k!='files'}
    body['settings']={k:v for k,v in c['settings'].items() if k!='calibration_source_root'}
    return normalized(body,root)


def qwen_binding(manifest):
    c=next(c for c in manifest['configs'] if c['id']==QWEN)
    body={k:v for k,v in c.items() if k!='files'}
    body['settings']={k:v for k,v in c['settings'].items() if k!='calibration_source_root'}
    return body


def validate_diagnostic(old,root,request):
    from evaluate import read,file_hash,unseal,cells,image_dimensions
    from native import KLEIN_TRANSFORMER_CONFIG,KLEIN_TEXT_ENCODER_CONFIG,FROZEN_SOURCE,FROZEN_INFERENCE
    evidence=root/'diagnostic'
    need(set(archive_files(evidence))==DIAGNOSTIC_NAMES,'exact diagnosed Klein evidence closure required')
    need({n:file_hash(evidence/n) for n in DIAGNOSTIC_NAMES}==request['diagnostic_files'],'Klein diagnostic evidence changed')
    state=read(evidence/'selected-state.json');job=state['job'];metrics=state['metrics']
    cell=next(c for c in cells(old,'calibration') if c['config_id']==KLEIN)
    need(job['id']==request['failed_native_job'] and job['type']=='image_generate' and job['status']=='completed' and job['worker_id']=='sc26322-'+KLEIN,'diagnosed Klein job identity mismatch')
    payload=json.loads(job['payload_json']);result=json.loads(job['result_json'])
    need(payload['model']==KLEIN and payload['seed']==cell['seed'] and payload['width']==cell['width'] and payload['height']==cell['height'] and payload['count']==1 and payload['prompt']==cell['prompt'] and payload['advanced']['mlxQuantize']==4 and payload['advanced']['mlxQuantizeExplicit'] is True,'diagnosed request/cell mismatch')
    expected=dict(job_id=job['id'],model=KLEIN,quant_label='bf16',quant_bits=None,steps=4,seed=cell['seed'],width=cell['width'],height=cell['height'],image_count=1,backend='cuda')
    need(all(metrics.get(k)==v for k,v in expected.items()),'diagnosed failure was not the exact frozen tier-label defect')
    assets=result['assets'];need(len(assets)==1,'diagnosed failed asset count mismatch');asset=assets[0]
    recipe=read(evidence/'asset.recipe.json');sidecar=read(evidence/'asset.sceneworks.json')
    need(asset['recipe']==recipe==sidecar['recipe'] and asset['id']==sidecar['id'] and asset['lineage']['jobId']==job['id']==sidecar['lineage']['jobId'],'diagnosed asset lineage/recipe mismatch')
    raw=recipe['rawAdapterSettings']
    need(recipe['model']==KLEIN and raw['mlxQuantize']==4 and raw['realModelInference'] is True and raw['numInferenceSteps']==4,'diagnosed actual recipe tier mismatch')
    need(image_dimensions(evidence/'original.png')==(cell['width'],cell['height']),'diagnosed failed PNG canvas mismatch')
    index=read(evidence/'selected-index.json')
    need(index['status']=='complete' and index['job_id']==job['id'] and index['selected_state_sha256']==file_hash(evidence/'selected-state.json'),'diagnostic SQL identity mismatch')
    need(any(f.get('artifact')=='original.png' and f['sha256']==file_hash(evidence/'original.png') for f in index['files']),'diagnostic original PNG binding missing')
    c=next(c for c in old['configs'] if c['id']==KLEIN)
    need(c['revision']=='1902693279fcfb828919370dfac2b8922d99499a' and c['precision']=='q4','diagnosed artifact revision/tier mismatch')
    versions=c['settings']['runtime_versions']
    need(versions['sceneworks_source']==FROZEN_SOURCE and versions['inference']==FROZEN_INFERENCE and c['settings']['binary_sha256']==request['binary_sha256'],'diagnosed loader/source/binary proof mismatch')
    files={f['path'].replace('\\','/'):f['sha256'] for f in c['files']}
    for suffix,sha in (('/q4/transformer/config.json',KLEIN_TRANSFORMER_CONFIG),('/q4/text_encoder/config.json',KLEIN_TEXT_ENCODER_CONFIG)):
        need(sum(p.endswith(suffix) and v==sha for p,v in files.items())==1,'diagnosed packed/dense component config proof missing')
    events=[]
    for line in (evidence/'worker.log').read_text(encoding='utf-8-sig').splitlines():
        try:events.append(json.loads(line))
        except json.JSONDecodeError:continue
    strategy=[e for e in events if e.get('event')=='image_memory_strategy_selected' and e.get('job_id')==job['id']]
    need(len(strategy)==1 and strategy[0].get('model')==KLEIN and strategy[0].get('route')==KLEIN and strategy[0].get('actual_tier')=='q4','diagnosed exact-job resolved q4 event missing')
    need(any(e.get('event')=='image_inference_complete' and e.get('jobId')==job['id'] and e.get('backend')=='candle' for e in events),'diagnosed completed Candle backend proof missing')
    return dict(job_id=job['id'],png_sha256=file_hash(evidence/'original.png'),classification='failed harness proof attempt; not runtime unavailability; excluded from accepted calibration budget',missing=['sealed successful cell receipt','wrapper elapsed/cold time','host peak RSS'])


def source_records(root,request):
    """Verify four successes under their original two locks, without resealing."""
    from evaluate import read,unseal,file_hash,validate_receipt,cells
    from calibration_source import carried
    old=unseal(read(root/'protocol.lock.json'));prep=read(root/'prepared.json')
    need(read(root/'protocol.lock.json')['sha256']==request['protocol_sha256'] and file_hash(root/'prepared.json')==request['prepared_sha256'] and old['preparation']['sha256']==request['prepared_sha256'],'v6 source identity changed')
    need({p.name:file_hash(p) for p in (root/'harness').iterdir() if p.is_file()}==prep['harness'],'archived v6 executed harness changed')
    need(old.get('calibration_carry_forward') and 'kind' not in read(root/'run'/DIRECTORY/'bundle.json'),'only fixed v5 to v6 prior carry permitted')
    rows,parity=carried(old,root/'run')
    need(set(r['cell']['config_id'] for r in rows.values())==set(MOSS),'exact original three MOSS outputs required')
    prior=old['calibration_carry_forward']['claim']
    need(prior==request['prior_claim'] and file_hash(root/'run'/DIRECTORY/'repair-claim.json')==prior['sha256'],'original v5 one-time claim changed')
    from ownership import physical
    invalidations=root/'run'/'invalidations'
    need({p.name:file_hash(p) for p in invalidations.glob('*.json')}==request['invalidations'] and len(request['invalidations'])==2,'only exact two diagnosed Klein invalidations permitted')
    klein=next(c for c in cells(old,'calibration') if c['config_id']==KLEIN)
    reasons=[]
    for path in invalidations.glob('*.json'):
        value=unseal(read(path));need(value['lock_sha256']==request['protocol_sha256'] and value['cell_ids']==[klein['id']] and value['parity_cell_ids']==[],'unrelated/scoped integrity invalidation cannot be repaired')
        reasons.append(value['reason'])
    need('ValueError: native actual worker/model/tier/settings mismatch' in reasons and any(r.startswith('outer owned-runtime watchdog stopped attempt: calibration process failed; inspect ') for r in reasons),'different integrity failure cannot be repaired')
    qwen=next(c for c in cells(old,'calibration') if c['config_id']==QWEN)
    path=root/'run'/'receipts'/(qwen['id']+'.json')
    need(file_hash(path)==request['qwen_receipt_sha256'],'original Qwen receipt changed')
    row=validate_receipt(root/'run',request['protocol_sha256'],qwen,read(path),manifest=old)
    need(row['status']=='ok','Qwen source is not successful');rows[qwen['id']]=row
    need({p.name for p in (root/'run'/'receipts').glob('*.json')}=={path.name},'additional v6 cell attempts forbidden')
    need({p.name for p in (root/'run'/'outputs').iterdir()}=={qwen['id']+'.png'},'unexpected v6 generated output')
    need(not list((root/'run').rglob('scored.*.png')) and not list((root/'run'/'receipts').glob('scored.*.json')),'repair forbidden after scored work')
    job=read(root/'source-job.json')
    need(file_hash(root/'source-job.json')==request['job_receipt_sha256'] and job['status']=='failed' and job['mode']=='calibration' and job['job']==request['job'] and job['transport_sha256']==prep['harness']['bootstrap.py'] and job['sceneworks_source']==prep['sceneworks_source'] and job['inference']==prep['inference'] and job['accelerator_execution_requested'] is True,'v6 failed transport identity changed')
    failure=validate_diagnostic(old,root,request)
    return old,prep,rows,parity,failure


def carried_repair(manifest,run,value):
    from evaluate import read,file_hash
    root=Path(run)/DIRECTORY
    old,prep,rows,parity,failure=source_records(root,value['source'])
    proof=value['compatibility']
    need(proof['new_runtime']==manifest['runtime'] and proof['new_hardware']==manifest['hardware'] and proof['new_prepared']==manifest['preparation'] and proof['new_moss_configs']==config_binding(manifest),'repaired compatibility binding changed')
    need(proof['new_qwen_config']==qwen_binding(manifest),'carried Qwen recipe/runtime/binary conditions changed')
    need(value['failed_attempt']==failure,'failed-attempt accounting changed')
    from evaluate import unseal
    claim=unseal(read(root/'repair-claim.json'))
    need(file_hash(root/'repair-claim.json')==value['claim']['sha256'] and claim['source_protocol_sha256']==value['source']['protocol_sha256'] and claim['prior_claim']==value['source']['prior_claim'] and claim['destination_prepared_sha256']==manifest['preparation']['sha256'] and claim['authorization_id']==value['source']['authorization']['id'],'v6 append-only repair claim changed')
    from evaluate import cells
    current_cells={c['id']:c for c in cells(manifest,'calibration')}
    for cell_id in parity:
        need(not (Path(run)/'parity'/(cell_id+'.json')).exists(),'duplicate carried parity fixture forbidden')
    for row in rows.values():
        need(row['cell']==current_cells[row['cell']['id']],'carried cell changed under new protocol')
        need(not (Path(run)/'receipts'/(row['cell']['id']+'.json')).exists() and not (Path(run)/'outputs'/(row['cell']['id']+'.png')).exists(),'new carried calibration artifact forbidden')
    return rows,parity



def native_source(path):
    tree=ast.parse(Path(path).read_text(encoding='utf-8'))
    approved=ast.parse("""tier_proof=None
if config['id']=='flux2_klein_9b':
 tier_proof=klein_tier_proof(config,job_id,metrics,job['result']['assets'][0]['recipe'],self.context['worker_log'])
 expected.pop('quantLabel')
if config['id']=='z_image_turbo':
 tier_proof=z_tier_proof(config,job_id,metrics,job['result']['assets'][0]['recipe'],self.context['worker_log'])""").body
    removed={ast.dump(n) for n in approved}
    classes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Native']
    need(len(classes)==1,'one native runtime class required')
    renders=[n for n in classes[0].body if isinstance(n,ast.FunctionDef) and n.name=='render']
    need(len(renders)==1,'one native render method required')
    render=renders[0]
    additions=[n for n in render.body if ast.dump(n) in removed]
    has_helpers=any(isinstance(n,ast.FunctionDef) and n.name in ('klein_tier_proof','z_tier_proof') for n in tree.body)
    need([ast.dump(n) for n in additions]==[ast.dump(n) for n in approved] if has_helpers else not additions,'packed proof plumbing missing, duplicated or misplaced')
    allowed={id(n) for n in additions}
    # Match complete guards and the exact final return field. No expression or
    # dictionary elsewhere in render is an authorized compatibility omission.
    old_recipe_guard=ast.parse("""if recipe.get('model')!=config['id'] or raw.get('realModelInference') is not True or not raw.get('repo') or raw.get('numInferenceSteps')!=config['steps'] or raw.get('mlxQuantize')!={'bf16':None,'q8':8,'q4':4}[config['precision']]:
 raise ValueError('native actual asset route/repo/tier mismatch')""").body[0]
    new_recipe_guard=ast.parse("""if recipe.get('model')!=config['id'] or raw.get('realModelInference') is not True or not raw.get('repo') or raw.get('numInferenceSteps')!=config['steps'] or raw.get('mlxQuantize')!=(None if config['id']=='z_image_turbo' else {'bf16':None,'q8':8,'q4':4}[config['precision']]):
 raise ValueError('native actual asset route/repo/tier mismatch')""").body[0]
    guards=[n for n in render.body if ast.dump(n)==ast.dump(new_recipe_guard if has_helpers else old_recipe_guard)]
    need(len(guards)==1,'exact native recipe guard missing or duplicated')
    recipe_guard=guards[0]
    returns=[n for n in render.body if isinstance(n,ast.Return)]
    need(len(returns)==1 and returns[0] is render.body[-1] and isinstance(returns[0].value,ast.Call) and isinstance(returns[0].value.func,ast.Name) and returns[0].value.func.id=='dict','exact final native metrics return required')
    fields=[k.value for k in returns[0].value.keywords if k.arg=='native_proof']
    old_proof=ast.parse("{'submitted_request':body,'job':job,'metrics':metrics,'asset':assets[0],'route_events':self.route_events,'job_events':events,'runtime_context':self.context}",mode='eval').body
    new_proof=ast.parse("{'submitted_request':body,'job':job,'metrics':metrics,'asset':assets[0],'route_events':self.route_events,'job_events':events,'runtime_context':self.context,'resolved_tier_proof':tier_proof}",mode='eval').body
    need(len(fields)==1 and ast.dump(fields[0])==ast.dump(new_proof if has_helpers else old_proof),'exact returned native proof missing, duplicated or changed')
    proof_field=fields[0]
    class Project(ast.NodeTransformer):
        def visit(self,n):
            if isinstance(n,ast.FunctionDef) and n.name in ('klein_tier_proof','z_tier_proof'):
                contract=KLEIN_HELPER_SOURCE if n.name=='klein_tier_proof' else Z_HELPER_SOURCE
                need(ast.dump(n)==ast.dump(ast.parse(contract).body[0]),'Klein helper changed outside reviewed proof contract')
                return None
            if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in KLEIN_CONSTANTS:
                need(n in tree.body and isinstance(n.value,ast.Constant) and n.value.value==KLEIN_CONSTANTS[n.targets[0].id],'Klein proof dependency changed')
                return None
            if id(n) in allowed:return None
            if has_helpers and n is recipe_guard:return old_recipe_guard
            if has_helpers and n is proof_field:return old_proof
            return super().visit(n)
    for name in ('klein_tier_proof','z_tier_proof'):
        helpers=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name]
        referenced=any(isinstance(n,ast.Name) and n.id==name for n in ast.walk(tree))
        need((len(helpers)==1 and helpers[0] in tree.body) if referenced else not helpers,'missing/duplicated/nested packed proof helper')
    return ast.dump(Project().visit(tree))

def install(root,manifest,request):
    from evaluate import read,write,seal,unseal,file_hash,cells
    from ownership import physical
    import bootstrap
    required={'kind','campaign','protocol_sha256','prepared_sha256','job','job_receipt_sha256','qwen_receipt_sha256','invalidations','diagnostic_files','failed_native_job','binary_sha256','prior_claim','authorization'}
    need(set(request)==required and request['kind']=='klein_metrics_v6' and request['campaign']==V6 and root.name!=V6,'only fixed diagnosed v6 repair source permitted')
    need(re.fullmatch(r'[0-9]+-[0-9]+',request['job']) and len(request['invalidations'])==2 and set(request['diagnostic_files'])==DIAGNOSTIC_NAMES,'exact repair job/evidence bindings required')
    need(all(re.fullmatch(r'[0-9a-f]{64}',v) for v in [request[k] for k in ('protocol_sha256','prepared_sha256','job_receipt_sha256','qwen_receipt_sha256','binary_sha256')]+list(request['invalidations'].values())+list(request['diagnostic_files'].values())),'exact repair evidence SHA256 required')
    auth=request['authorization']
    need(set(auth)=={'id','coordinator','reason','new_harness_sha256','remaining_calibration_images','failed_attempt_classification'} and auth['id'] and auth['reason'] and auth['coordinator']==manifest['ownership']['responsible_chat'] and type(auth['remaining_calibration_images']) is int and auth['remaining_calibration_images']==2 and auth['failed_attempt_classification']=='diagnosed_harness_tier_reporting_proof_failure','explicit diagnosed repair authorization required')
    source=root.parent/V6;physical(source);need(not (root/'run').exists(),'repair must precede any current run work')
    need(file_hash(source/'prepared.json')==request['prepared_sha256'],'v6 preparation changed');prep=bootstrap.prepared(source);current=bootstrap.prepared(root)
    need(current['binary_sha256']==request['binary_sha256']==prep['binary_sha256'] and current['native_build']==prep['native_build'],'comparison native binary/tool identity must remain unchanged')
    need(auth['new_harness_sha256']==hashlib.sha256(__import__('evaluate').canonical(current['harness'])).hexdigest(),'coordinator-approved repaired harness closure changed')
    old=unseal(read(source/'protocol.lock.json'))
    need(old['preparation']['path']==str((source/'prepared.json').resolve()),'v6 preparation root mismatch')
    compatible(old,manifest,prep,current,source/'harness',root/'harness',source,root)
    need(qwen_config(old,source)==qwen_config(manifest,root),'carried Qwen runtime/settings/source/binary changed')
    # Same measurement/worker/guard semantics as the already checked MOSS path.
    # Native adapter change is restricted to the separately reviewed Klein branch.
    need(native_source(source/'harness'/'native.py')==native_source(root/'harness'/'native.py'),'carried Qwen native execution changed outside exact Klein proof plumbing')
    archive=root/'run'/DIRECTORY;archive.mkdir(parents=True)
    def copy(src,name):
        physical(src);dest=archive/name;physical(dest);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest);need(file_hash(src)==file_hash(dest),'repair archive copy drift')
    for name in ('protocol.lock.json','prepared.json'):copy(source/name,name)
    copy(source/('job-'+request['job']+'-calibration.json'),'source-job.json')
    for rel,p in archive_files(source/'run').items():copy(p,'run/'+rel)
    for name in prep['harness']:copy(source/'harness'/name,'harness/'+name)
    evidence=root/'repair-evidence'
    for name in DIAGNOSTIC_NAMES:copy(evidence/name,'diagnostic/'+name)
    old,prep,rows,parity,failure=source_records(archive,request)
    prior=Path(request['prior_claim']['path']);physical(prior)
    need(prior==root.parent/'calibration-repairs'/(read(archive/'run'/DIRECTORY/'protocol.lock.json')['sha256']+'.json') and file_hash(prior)==request['prior_claim']['sha256'],'prior v5 claim missing/changed; never overwrite or steal it')
    claims=root.parent/'calibration-repairs';claim=claims/(request['protocol_sha256']+'.json');physical(claim)
    bootstrap.write(claim,seal(dict(source_protocol_sha256=request['protocol_sha256'],prior_claim=request['prior_claim'],destination_campaign=root.name,destination_prepared_sha256=manifest['preparation']['sha256'],authorization_id=auth['id'],responsible_chat=auth['coordinator'])))
    copy(claim,'repair-claim.json')
    binding=dict(path=str(claim.resolve()),sha256=file_hash(claim))
    files={name:file_hash(p) for name,p in archive_files(archive).items()}
    value=seal(dict(kind='klein_metrics_v6',source=request,files=files,claim=binding,failed_attempt=failure,compatibility=dict(new_runtime=manifest['runtime'],new_hardware=manifest['hardware'],new_prepared=manifest['preparation'],new_moss_configs=config_binding(manifest),new_qwen_config=qwen_binding(manifest)),limitations='Four original successful measurements remain under their original v5/v6 protocol/desktop/process conditions. Failed Klein asset is retained with missing wrapper/host metrics; only Klein and Z may produce new calibration outputs.'))
    write(archive/'bundle.json',value);manifest['calibration_carry_forward']=dict(path=DIRECTORY+'/bundle.json',sha256=file_hash(archive/'bundle.json'),claim=binding)
    carried_repair(manifest,root/'run',value)
    return archive


# One reviewed policy transition, not a caller-selected source/hash exception.
OWNERSHIP_V7_SHA256='54ddc476a1e4e924f0b6f7c25be1c6713ab9791c2e11a31c6f5ae2a9d1ccaf92'
OWNERSHIP_V8_SHA256='db699fd2f69d693e169485c05486e98294986aed9bb53621ec4bd9bc1f4e231e'
OWNERSHIP_BLOCKED_JOBS={
    '39032307669-1':'ValueError: foreign process appeared on leased physical GPU1',
    '39032793002-1':'ValueError: desktop baseline identity changed for PID4',
}


def ownership_policy_bridge(old,new):
    """Permit exactly the independently reviewed System/idle-subset patch."""
    from evaluate import file_hash
    need(file_hash(old)==OWNERSHIP_V7_SHA256 and file_hash(new)==OWNERSHIP_V8_SHA256,
         'ownership repair differs from the exact reviewed policy transition')


def ownership_config_binding(manifest):
    values=json.loads(json.dumps(manifest['configs']))
    for config in values:
        config.pop('files')
        config['settings'].pop('calibration_source_root',None)
    return values


def ownership_request(request):
    required={'kind','campaign','protocol_sha256','prepared_sha256','prepare_job','prepare_job_receipt_sha256','failed_jobs','prior_claim','authorization'}
    need(set(request)==required and request['kind']=='ownership_policy_v7' and request['campaign']=='sc-26322-reference-v7','only fixed v7 ownership repair permitted')
    need(isinstance(request['prepare_job'],str) and re.fullmatch(r'[0-9]+-[0-9]+',request['prepare_job']),'exact ownership source preparation job required')
    need(set(request['failed_jobs'])==set(OWNERSHIP_BLOCKED_JOBS),'exact two pre-execution blocked jobs required')
    hashes=[request[k] for k in ('protocol_sha256','prepared_sha256','prepare_job_receipt_sha256')]+list(request['failed_jobs'].values())
    need(all(isinstance(v,str) and re.fullmatch(r'[0-9a-f]{64}',v) for v in hashes),'exact ownership repair receipt hashes required')


def ownership_source_records(root,request):
    """Read the unchanged v7 archive; no new calibration or scoring ever started."""
    from evaluate import read,unseal,file_hash
    from calibration_source import carried
    ownership_request(request)
    old=unseal(read(root/'protocol.lock.json'));prep=read(root/'prepared.json')
    need(read(root/'protocol.lock.json')['sha256']==request['protocol_sha256'] and file_hash(root/'prepared.json')==request['prepared_sha256']==old['preparation']['sha256'],'v7 original protocol/preparation changed')
    need(prep['harness']['ownership.py']==OWNERSHIP_V7_SHA256 and {p.name:file_hash(p) for p in (root/'harness').iterdir() if p.is_file()}==prep['harness'],'v7 original executed harness changed')
    need(read(root/'run'/DIRECTORY/'bundle.json').get('kind')=='klein_metrics_v6','only original fixed v5/v6/v7 archive allowed')
    need({p.name for p in (root/'run').iterdir()}=={DIRECTORY},'v7 contains new execution artifacts; ownership repair cannot retry them')
    rows,parity=carried(old,root/'run')
    need({r['cell']['config_id'] for r in rows.values()}==set(CARRIED) and all(r['status']=='ok' for r in rows.values()) and len(parity)==1,'ownership repair requires unchanged four successes and first parity')
    need(old['calibration_carry_forward']['claim']==request['prior_claim'],'v6 to v7 claim binding changed')
    need(file_hash(root/'run'/DIRECTORY/'repair-claim.json')==request['prior_claim']['sha256'],'v6 to v7 claim bytes changed')
    prepare=read(root/'source-prepare-job.json')
    need(file_hash(root/'source-prepare-job.json')==request['prepare_job_receipt_sha256'] and prepare['mode']=='prepare' and prepare['status']=='completed' and prepare['job']==request['prepare_job'] and prepare['transport_sha256']==prep['harness']['bootstrap.py'] and prepare['sceneworks_source']==prep['sceneworks_source'] and prepare['inference']==prep['inference'] and prepare['accelerator_execution_requested'] is False,'v7 source preparation job changed')
    blocked=root/'blocked-jobs'
    need({p.name for p in blocked.iterdir()}=={j+'.json' for j in OWNERSHIP_BLOCKED_JOBS},'blocked job archive closure changed')
    original_root=old['preparation']['path'].replace('\\','/').rsplit('/',1)[0]
    need(prepare['root'].replace('\\','/')==original_root,'v7 original preparation root changed')
    for job_id,error in OWNERSHIP_BLOCKED_JOBS.items():
        path=blocked/(job_id+'.json');job=read(path)
        need(file_hash(path)==request['failed_jobs'][job_id] and job['mode']=='calibration' and job['status']=='failed' and job['job']==job_id and job['error']==error and job['root'].replace('\\','/')==original_root and job['transport_sha256']==prep['harness']['bootstrap.py'] and job['sceneworks_source']==prep['sceneworks_source'] and job['inference']==prep['inference'] and job['accelerator_execution_requested'] is True,'blocked job was not the exact pre-execution ownership rejection')
    return old,prep,rows,parity


def check_ownership_prior_claims(root,base):
    """Check both earlier global claims without replacing either one."""
    from evaluate import read,file_hash
    from ownership import physical
    for archived in (root,root/'run'/DIRECTORY):
        old=read(archived/'protocol.lock.json');binding=old['calibration_carry_forward']['claim']
        protocol=read(archived/'run'/DIRECTORY/'protocol.lock.json')['sha256']
        path=Path(binding['path']);physical(path)
        need(path==base/'calibration-repairs'/(protocol+'.json') and file_hash(path)==binding['sha256']==file_hash(archived/'run'/DIRECTORY/'repair-claim.json'),'prior one-time ownership repair chain changed')


def carried_ownership_repair(manifest,run,value):
    from evaluate import read,unseal,file_hash,cells
    root=Path(run)/DIRECTORY;old,prep,rows,parity=ownership_source_records(root,value['source'])
    proof=value['compatibility']
    need(proof['new_runtime']==manifest['runtime'] and proof['new_hardware']==manifest['hardware'] and proof['new_prepared']==manifest['preparation'] and proof['new_configs']==ownership_config_binding(manifest),'ownership repair runtime/model/prepared compatibility changed')
    need(proof['old_guard_sha256']==OWNERSHIP_V7_SHA256 and proof['new_guard_sha256']==OWNERSHIP_V8_SHA256,'ownership repair policy bridge changed')
    current_guard=manifest['preparation']['path'].replace('\\','/').rsplit('/',1)[0]+'/harness/ownership.py'
    for config in manifest['configs']:
        guards=[f['sha256'] for f in config['files'] if f['path'].replace('\\','/')==current_guard]
        need(guards==[OWNERSHIP_V8_SHA256],'current configuration does not bind reviewed ownership policy')
    claim=unseal(read(root/'repair-claim.json'))
    need(file_hash(root/'repair-claim.json')==value['claim']['sha256'] and claim['source_protocol_sha256']==value['source']['protocol_sha256'] and claim['prior_claim']==value['source']['prior_claim'] and claim['destination_campaign']=='sc-26322-reference-v8' and claim['destination_prepared_sha256']==manifest['preparation']['sha256'] and claim['authorization_id']==value['source']['authorization']['id'] and claim['guard_transition']==[OWNERSHIP_V7_SHA256,OWNERSHIP_V8_SHA256],'v7 to v8 append-only ownership claim changed')
    need(value['failed_attempt']==read(root/'run'/DIRECTORY/'bundle.json')['failed_attempt'],'original excluded failed Klein evidence changed')
    current={c['id']:c for c in cells(manifest,'calibration')}
    need(all(row['cell']==current[key] for key,row in rows.items()),'ownership repair carried calibration cell changed')
    need(all(not (Path(run)/'parity'/(key+'.json')).exists() for key in parity),'duplicate carried parity fixture forbidden')
    return rows,parity


def install_ownership_repair(root,manifest,request):
    """One explicit v7 -> v8 bridge before any current run output or claim."""
    from evaluate import read,write,seal,file_hash,unseal,canonical
    from ownership import physical
    import bootstrap
    ownership_request(request)
    need(root.name=='sc-26322-reference-v8','ownership repair destination must be fresh v8')
    source=root.parent/'sc-26322-reference-v7';physical(source)
    need(not (root/'run').exists(),'ownership repair must precede all current run artifacts')
    need(file_hash(source/'prepared.json')==request['prepared_sha256'],'v7 preparation changed')
    prep=bootstrap.prepared(source);current=bootstrap.prepared(root);old=unseal(read(source/'protocol.lock.json'))
    auth=request['authorization']
    need(set(auth)=={'id','coordinator','reason','new_harness_sha256','remaining_calibration_images','repair'} and auth['id'] and auth['reason'] and auth['coordinator']==manifest['ownership']['responsible_chat'] and type(auth['remaining_calibration_images']) is int and auth['remaining_calibration_images']==2 and auth['repair']=='windows_system_zero_epoch_and_verified_idle_subset','explicit fixed ownership policy repair authority required')
    need(auth['new_harness_sha256']==hashlib.sha256(canonical(current['harness'])).hexdigest(),'authorized ownership repair harness changed')
    need(current['binary_sha256']==prep['binary_sha256'] and current['native_build']==prep['native_build'],'ownership repair cannot rebuild or change native runtime identity')
    need(current['reuse']['source_prepared_sha256']==prep['reuse']['source_prepared_sha256'],'original immutable acquisition source changed')
    reused=read(root/'reuse.json')['native']
    need(reused['mode']=='exact_v7_binary_copy_no_build' and reused['source_prepared_sha256']==request['prepared_sha256'] and reused['binary']['method']=='copy' and reused['binary']['sha256']==prep['binary_sha256'],'ownership repair requires independent exact v7 binary copy, no rebuild')
    compatible(old,manifest,prep,current,source/'harness',root/'harness',source,root,ownership_repair=True)
    need(normalized(ownership_config_binding(old),source)==normalized(ownership_config_binding(manifest),root),'ownership repair changes a generation configuration')
    for key in prep['staged']:
        a,b=prep['staged'][key],current['staged'][key]
        need(all(a[k]==b[k] for k in ('repo','revision','metadata_sha256')),'ownership repair changes model metadata')
        need(normalized(a['files'],source)==normalized(b['files'],root),'ownership repair changes staged model bytes')
    need(qwen_config(old,source)==qwen_config(manifest,root) and file_hash(source/'harness/native.py')==file_hash(root/'harness/native.py') and file_hash(source/'harness/native_services.py')==file_hash(root/'harness/native_services.py'),'ownership repair changes native generation or service semantics')
    need({p.name for p in source.glob('job-*-calibration.json')}=={'job-'+j+'-calibration.json' for j in OWNERSHIP_BLOCKED_JOBS},'unreviewed extra v7 calibration jobs cannot be transferred')
    archive=root/'run'/DIRECTORY;archive.mkdir(parents=True)
    def copy(src,name):
        physical(src);dest=archive/name;physical(dest);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest);need(file_hash(src)==file_hash(dest),'ownership archive copy changed bytes')
    for name in ('protocol.lock.json','prepared.json'):copy(source/name,name)
    copy(source/('job-'+request['prepare_job']+'-prepare.json'),'source-prepare-job.json')
    for job in OWNERSHIP_BLOCKED_JOBS:copy(source/('job-'+job+'-calibration.json'),'blocked-jobs/'+job+'.json')
    for name,p in archive_files(source/'run').items():copy(p,'run/'+name)
    for name in prep['harness']:copy(source/'harness'/name,'harness/'+name)
    old,prep,rows,parity=ownership_source_records(archive,request)
    check_ownership_prior_claims(archive,root.parent)
    claim=root.parent/'calibration-repairs'/(request['protocol_sha256']+'.json');physical(claim)
    bootstrap.write(claim,seal(dict(source_protocol_sha256=request['protocol_sha256'],prior_claim=request['prior_claim'],destination_campaign=root.name,destination_prepared_sha256=manifest['preparation']['sha256'],authorization_id=auth['id'],responsible_chat=auth['coordinator'],guard_transition=[OWNERSHIP_V7_SHA256,OWNERSHIP_V8_SHA256])))
    copy(claim,'repair-claim.json');binding=dict(path=str(claim.resolve()),sha256=file_hash(claim))
    proof=dict(new_runtime=manifest['runtime'],new_hardware=manifest['hardware'],new_prepared=manifest['preparation'],new_configs=ownership_config_binding(manifest),old_guard_sha256=OWNERSHIP_V7_SHA256,new_guard_sha256=OWNERSHIP_V8_SHA256)
    value=seal(dict(kind='ownership_policy_v7',source=request,files={n:file_hash(p) for n,p in archive_files(archive).items()},claim=binding,compatibility=proof,failed_attempt=read(archive/'run'/DIRECTORY/'bundle.json')['failed_attempt'],limitations='Two exact v7 jobs failed before runtime allocation; no new outputs/receipts/invalidations. Four original v5/v6 successes and parity16 remain unchanged. Only Klein/Z calibration pending.'))
    write(archive/'bundle.json',value);manifest['calibration_carry_forward']=dict(path=DIRECTORY+'/bundle.json',sha256=file_hash(archive/'bundle.json'),claim=binding)
    carried_ownership_repair(manifest,root/'run',value)
    return archive
