"""One explicit native-stdin repair: retain three original MOSS calibrations.

Original receipts are archived unchanged and always validated under their original
protocol. This is not a retry mechanism or permission to regenerate MOSS images.
"""
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil

MOSS=('moss_base','moss_sft','moss_dmd')
DIRECTORY='calibration-source'


def need(ok,message):
    if not ok:raise ValueError(message)


def normalized(value,root):
    if isinstance(value,str):
        text=value.replace('\\','/');prefix=str(root).replace('\\','/').rstrip('/')+'/'
        return '$CAMPAIGN/'+text[len(prefix):] if text.startswith(prefix) else value
    if isinstance(value,list):return [normalized(v,root) for v in value]
    if isinstance(value,dict):return {k:normalized(v,root) for k,v in value.items()}
    return value


def execution_source(path):
    """Remove only the explicit separately-bound archive lookup statements."""
    omitted=["if cell['id'] in carried_rows:\n continue", "if cell['id'] in prior:\n records.append(prior[cell['id']]); continue",
        "if cell['id'] in results:\n continue", "carried_rows, _ = carried(manifest, run)","_, prior = carried(manifest, run)",
        "receipt_root = calibration_record_root(self.manifest, self.prepared_root / 'run', cell['config_id'])",
        "check_carry_closure(self.manifest, self.prepared_root / 'run', execution=True)"]
    remove={ast.dump(ast.parse(text).body[0]) for text in omitted}
    class Project(ast.NodeTransformer):
        def visit(self,n):
            if isinstance(n,ast.ImportFrom) and n.module=='calibration_source' and len(n.names)==1 and n.names[0].name in ('carried','calibration_record_root','check_carry_closure') and n.names[0].asname is None:return None
            if ast.dump(n) in remove:return None
            if isinstance(n,ast.Assign) and ast.dump(n)==ast.dump(ast.parse('results, _ = carried(manifest, run_root)').body[0]):
                return ast.parse('results = {}').body[0]
            if ast.dump(n)==ast.dump(ast.parse("path = receipt_root / 'receipts' / (cell['id'] + '.json')").body[0]):
                return ast.parse("path = self.prepared_root / 'run' / 'receipts' / (cell['id'] + '.json')").body[0]
            if ast.dump(n)==ast.dump(ast.parse("evidence.extend([path, artifact(receipt_root, read(path)['log'])])").body[0]):
                return ast.parse("evidence.extend([path, artifact(self.prepared_root / 'run', read(path)['log'])])").body[0]
            return super().visit(n)
    return ast.dump(Project().visit(ast.parse(Path(path).read_text(encoding='utf-8'))))


def evaluation_source(path):
    """Compare all evaluation dependencies, excluding only archive plumbing."""
    # Independent reviewed contract: do not derive this from the candidate module.
    # Only this exact helper may be removed when comparing against the old source.
    approved_helper=ast.parse("""def parity_evidence(manifest,root,lock_hash):
    from calibration_source import carried
    _,prior=carried(manifest,root)
    result={digest(row):row for row in prior.values()}
    for path in (Path(root)/'parity').glob('*.json'):
        doc=read(path);row=validate_parity(root,lock_hash,doc['cell'],doc)
        require(row['cell']['id'] not in {r['cell']['id'] for r in result.values()},'duplicate controlled parity fixture')
        result[doc['sha256']]=row
    return result
""").body[0]
    omitted=["""if manifest.get('calibration_carry_forward'):
 from calibration_source import carried
 carried(manifest,Path(manifest['preparation']['path']).parent/'run')""",
        """if manifest.get('calibration_carry_forward'):
 require(config['settings'].get('calibration_source_root')==str(Path(manifest['preparation']['path']).parent/'run'/'calibration-source'),'carried archive is outside current campaign')""",
        """if settings.get('calibration_source_root'):
 roots.append(settings['calibration_source_root'])""",
        """if phase=='calibration':
 from calibration_source import carried
 results,_=carried(manifest,run)""", "if cell['id'] in results:\n continue"]
    old_loop="""for path in (Path(root)/'parity').glob('*.json'):
 doc=read(path)
 if doc.get('sha256') in declared:
  row=validate_parity(root,lock_hash,doc['cell'],doc)
  actual[doc['sha256']]=row"""
    new_loop="""for sha,row in parity_evidence(manifest or {},root,lock_hash).items():
 if sha in declared:actual[sha]=row"""
    old_report="""parity=[]
for cell in parity_cells(manifest):
 path=Path(run)/'parity'/(cell['id']+'.json')
 if path.exists():parity.append(validate_parity(run,read(lock)['sha256'],cell,read(path)))"""
    replacements={ast.dump(ast.parse(new_loop).body[0]):ast.parse(old_loop).body,
        ast.dump(ast.parse("parity=list(parity_evidence(manifest,run,read(lock)['sha256']).values())").body[0]):ast.parse(old_report).body}
    remove={ast.dump(ast.parse(text).body[0]) for text in omitted}
    class Project(ast.NodeTransformer):
        def visit(self,n):
            if isinstance(n,ast.FunctionDef) and n.name=='parity_evidence':
                need(ast.dump(n)==ast.dump(approved_helper),'carried MOSS parity evidence helper changed')
                return None
            if ast.dump(n) in remove:return None
            if ast.dump(n) in replacements:return replacements[ast.dump(n)]
            return super().visit(n)
    tree=ast.parse(Path(path).read_text(encoding='utf-8'))
    helpers=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='parity_evidence']
    referenced=any(isinstance(n,ast.Name) and n.id=='parity_evidence' for n in ast.walk(tree))
    need((len(helpers)==1 and helpers[0] in tree.body) if referenced else not helpers,'carried MOSS parity evidence helper missing or duplicated')
    return ast.dump(Project().visit(tree))


def config_binding(manifest):
    values=json.loads(json.dumps([c for c in manifest['configs'] if c['id'] in MOSS]))
    archive=manifest['preparation']['path'].replace('\\','/').rsplit('/',1)[0]+'/run/'+DIRECTORY+'/'
    for c in values:
        c['settings'].pop('calibration_source_root',None)
        c['files']=[f for f in c['files'] if not f['path'].replace('\\','/').startswith(archive)]
    return values


def compatible(old,new,old_prep,new_prep,old_harness,new_harness,old_root,new_root):
    from evaluate import file_hash
    for key in ('corpus','dmd_parity_fixtures','runtime','hardware','thresholds','scoring_protocol','visual_rubric','condition'):
        need(old[key]==new[key],f'carried MOSS {key} changed')
    for key in ('responsible_chat','timeout_seconds'):
        need(old['ownership'][key]==new['ownership'][key],f'carried MOSS ownership {key} changed')
    need(old_prep['runtime']==new_prep['runtime'],'carried installed runtime changed')
    for key in MOSS:
        a=next(c for c in old['configs'] if c['id']==key);b=next(c for c in new['configs'] if c['id']==key)
        def config(c,root):
            body={k:v for k,v in c.items() if k!='files'}
            body['settings']={k:v for k,v in c['settings'].items() if k!='calibration_source_root'}
            return normalized(body,root)
        need(config(a,old_root)==config(b,new_root),f'carried MOSS recipe changed: {key}')
        def inputs(c,root):
            rows=normalized(c['files'],root)
            need(all(f['path'].startswith(('$CAMPAIGN/hub/','$CAMPAIGN/harness/','$CAMPAIGN/run/'+DIRECTORY+'/')) for f in rows),'carried MOSS input escaped owned model/harness/archive roots')
            return {f['path']:f['sha256'] for f in rows if f['path'].startswith('$CAMPAIGN/hub/')}
        need(inputs(a,old_root) and inputs(a,old_root)==inputs(b,new_root),f'carried MOSS model bytes changed: {key}')
    need(file_hash(old_harness/'reference.py')==file_hash(new_harness/'reference.py'),'carried MOSS sampler/adapter source changed')
    for name in ('run.py','ownership.py'):
        need(execution_source(old_harness/name)==execution_source(new_harness/name),f'carried MOSS worker/measurement/live guard execution changed: {name}')
    need(evaluation_source(old_harness/'evaluate.py')==evaluation_source(new_harness/'evaluate.py'),'carried MOSS evaluation execution/dependencies changed')


def source_root(run):return Path(run)/DIRECTORY


def check_carry_closure(manifest,run,*,execution=False):
    """Cheap continuous identity/closure check; input fingerprints cover its bytes."""
    from evaluate import read,unseal,file_hash,cells
    from ownership import physical
    binding=manifest.get('calibration_carry_forward')
    if not binding:return None
    need(set(binding)=={'path','sha256','claim'} and binding['path']==DIRECTORY+'/bundle.json','invalid calibration source binding')
    root=source_root(run);physical(root);bundle=root/'bundle.json';physical(bundle)
    need(file_hash(bundle)==binding['sha256'],'calibration source bundle changed')
    value=unseal(read(bundle))
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    need(actual==set(value['files'])|{'bundle.json'},'calibration source archive closure changed')
    for name in actual:physical(root/name)
    need(binding['claim']==value['claim'],'calibration repair claim changed')
    if execution:
        claim=Path(binding['claim']['path']);physical(claim)
        need(claim==Path(run).parent.parent/'calibration-repairs'/(value['source']['protocol_sha256']+'.json'),'calibration repair claim escaped owned authority')
        need(file_hash(claim)==binding['claim']['sha256'],'one-time calibration repair claim changed')
    baseline={c['id']+'.png' for c in cells(manifest,'calibration') if c['config_id'] not in MOSS}
    current={p.name for p in (Path(run)/'outputs').glob('calibration.*.png')}
    need(current<=baseline and len(current)<=3,'remaining calibration budget permits only three native images')
    need(all(not (Path(run)/'receipts'/(c['id']+'.json')).exists() for c in cells(manifest,'calibration') if c['config_id'] in MOSS),'new MOSS calibration receipt forbidden')
    return value


def carried(manifest,run):
    from evaluate import read,unseal,file_hash,receipts,canonical,cells,validate_parity
    value=check_carry_closure(manifest,run)
    if value is None:return {},{}
    root=source_root(run)
    for name,sha in value['files'].items():need(file_hash(root/name)==sha,'carried original artifact changed')
    old=unseal(read(root/'protocol.lock.json'));prep=read(root/'prepared.json')
    need(not old.get('calibration_carry_forward'),'chained calibration carry-forward forbidden')
    need(read(root/'protocol.lock.json')['sha256']==value['source']['protocol_sha256'],'carried original protocol changed')
    need(file_hash(root/'prepared.json')==value['source']['prepared_sha256'],'carried original preparation changed')
    need(old['preparation']['sha256']==value['source']['prepared_sha256'],'original protocol/preparation binding changed')
    job=read(root/'source-job.json')
    need(file_hash(root/'source-job.json')==value['source']['job_receipt_sha256'] and job['status']=='completed' and job['mode']=='calibration' and job['job']==value['source']['job'] and job['sceneworks_source']==prep['sceneworks_source'] and job['inference']==prep['inference'] and job['transport_sha256']==prep['harness']['bootstrap.py'] and job['accelerator_execution_requested'] is True,'carried original job identity changed')
    _,rows=receipts(root/'protocol.lock.json',root/'run','calibration')
    need(hashlib.sha256(canonical(rows)).hexdigest()==value['source']['calibration_sha256'],'carried calibration attempt bundle changed')
    need({r['cell']['config_id'] for r in rows.values() if r['status']=='ok'}==set(MOSS),'carry requires exactly three original MOSS successes')
    need(all(r['status']=='failed' and r.get('failure_kind')=='runtime' and r.get('output') is None and r['error']=='RuntimeError: owned service exited; inspect logs' for r in rows.values() if r['cell']['config_id'] not in MOSS),'carry requires original native startup failures, not arbitrary retries')
    source_prepared=manifest['preparation']
    # Offline review uses the frozen compatibility proof; execution also verifies
    # current prepared/source bytes through the ordinary manifest and lease guard.
    need(value['compatibility']['new_runtime']==manifest['runtime'] and value['compatibility']['new_hardware']==manifest['hardware'],'carried runtime/hardware conditions changed')
    need(value['compatibility']['new_moss_configs']==config_binding(manifest),'carried configuration binding changed')
    need(value['compatibility']['new_prepared']==source_prepared,'carried current preparation identity changed')
    claim=read(root/'repair-claim.json')
    need(file_hash(root/'repair-claim.json')==value['claim']['sha256'] and claim['source_protocol_sha256']==value['source']['protocol_sha256'] and claim['destination_prepared_sha256']==manifest['preparation']['sha256'] and claim['authorization_id']==value['source']['authorization']['id'],'carried one-time repair authorization changed')
    original={k:r for k,r in rows.items() if r['cell']['config_id'] in MOSS}
    for cell in cells(manifest,'calibration'):
        if cell['config_id'] in MOSS:
            need(original[cell['id']]['cell']==cell,'carried original cell differs from current corpus')
            need(not (Path(run)/'receipts'/(cell['id']+'.json')).exists() and not (Path(run)/'outputs'/(cell['id']+'.png')).exists(),'new MOSS calibration artifact forbidden; original evidence must stay unchanged')
    parity={}
    for path in (root/'run'/'parity').glob('*.json'):
        doc=read(path);row=validate_parity(root/'run',read(root/'protocol.lock.json')['sha256'],doc['cell'],doc)
        need(not (Path(run)/'parity'/path.name).exists(),'duplicate carried parity fixture forbidden')
        parity[row['cell']['id']]=row
    need(len(parity)==1,'exactly first original parity fixture required')
    return original,parity


def calibration_record_root(manifest,run,config_id):
    if manifest.get('calibration_carry_forward') and config_id in MOSS:
        carried(manifest,run)
        return source_root(run)/'run'
    return Path(run)


def install(root,manifest,request):
    """Called only during a fresh freeze, before any current calibration output."""
    from evaluate import read,write,seal,file_hash,unseal,receipts,canonical,artifact
    from ownership import physical,reject_run_invalidated
    import bootstrap
    required={'campaign','protocol_sha256','prepared_sha256','job','job_receipt_sha256','calibration_sha256','authorization'}
    need(set(request)==required,'exact original calibration source bindings required')
    need(re.fullmatch(r'sc-26322-[a-z0-9][a-z0-9-]{0,48}',request['campaign']) and request['campaign']!=root.name,'different owned calibration source campaign required')
    need(re.fullmatch(r'[0-9]+-[0-9]+',request['job']),'exact original calibration job required')
    need(all(re.fullmatch(r'[0-9a-f]{64}',request[k]) for k in ('protocol_sha256','prepared_sha256','job_receipt_sha256','calibration_sha256')),'original calibration digests required')
    auth=request['authorization']
    need(set(auth)=={'id','coordinator','reason','native_services_sha256','new_harness_sha256','remaining_calibration_images'} and auth['id'] and auth['reason'] and auth['coordinator']==manifest['ownership']['responsible_chat'] and type(auth['remaining_calibration_images']) is int and auth['remaining_calibration_images']==3 and auth['native_services_sha256']==file_hash(Path(__file__).parent/'native_services.py'),'explicit diagnosed native-stdin repair authorization required')
    source=root.parent/request['campaign'];physical(source)
    need(not (root/'run').exists(),'carry must precede all current run artifacts')
    need(file_hash(source/'prepared.json')==request['prepared_sha256'],'source prepared receipt changed')
    prep=bootstrap.prepared(source);old=unseal(read(source/'protocol.lock.json'))
    need(old['preparation']==dict(path=str((source/'prepared.json').resolve()),sha256=request['prepared_sha256']),'original protocol/preparation authority mismatch')
    need(read(source/'protocol.lock.json')['sha256']==request['protocol_sha256'] and not old.get('calibration_carry_forward'),'original protocol changed or chained carry')
    reject_run_invalidated(source/'run',request['protocol_sha256'])
    need(not list((source/'run'/'receipts').glob('scored.*.json')) and not list((source/'run'/'outputs').glob('scored.*.png')),'carry forbidden after scored work')
    job=source/('job-'+request['job']+'-calibration.json');physical(job)
    need(file_hash(job)==request['job_receipt_sha256'],'source calibration job receipt changed')
    status=read(job)
    need(status['mode']=='calibration' and status['status']=='completed' and status['accelerator_execution_requested'] is True and status['transport_sha256']==prep['harness']['bootstrap.py'] and status['job']==request['job'] and Path(status['root']).resolve()==source.resolve() and status['sceneworks_source']==prep['sceneworks_source'] and status['inference']==prep['inference'],'source calibration job is not complete at exact source')
    _,rows=receipts(source/'protocol.lock.json',source/'run','calibration')
    need(hashlib.sha256(canonical(rows)).hexdigest()==request['calibration_sha256'],'source calibration bundle changed')
    need({r['cell']['config_id'] for r in rows.values() if r['status']=='ok'}==set(MOSS),'exactly three original MOSS successful outputs required')
    need(all(r['status']=='failed' and r.get('failure_kind')=='runtime' and r.get('output') is None and r.get('metrics') is None and r['error']=='RuntimeError: owned service exited; inspect logs' for r in rows.values() if r['cell']['config_id'] not in MOSS),'only diagnosed native startup failures may be repaired')
    need(len(list((source/'run'/'parity').glob('*.json')))==1,'exactly first original controlled parity fixture required')
    outputs={r['output']['path'].replace('\\','/') for r in rows.values() if r.get('output')}
    need({p.relative_to(source/'run').as_posix() for p in (source/'run'/'outputs').glob('*')}==outputs,'source image budget has extra or orphan outputs')
    current=bootstrap.prepared(root)
    need(auth['new_harness_sha256']==hashlib.sha256(canonical(current['harness'])).hexdigest(),'coordinator-approved new harness closure changed')
    compatible(old,manifest,prep,current,source/'harness',root/'harness',source,root)
    archive=source_root(root/'run');archive.mkdir(parents=True)
    files={}
    def copy(src,name):
        physical(src);target=archive/name;physical(target);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(src,target);files[name]=file_hash(target)
        need(files[name]==file_hash(src),'original calibration copy drift')
    copy(source/'protocol.lock.json','protocol.lock.json');copy(source/'prepared.json','prepared.json');copy(job,'source-job.json')
    for name in ('reuse.json','calibration-estimate.json'):
        if (source/name).is_file():copy(source/name,name)
    for name,sha in prep['harness'].items():
        need(re.fullmatch(r'[A-Za-z0-9_.-]+',name),'original harness basename required')
        need(file_hash(source/'harness'/name)==sha,'original executed harness changed');copy(source/'harness'/name,'harness/'+name)
    for cell_id,row in rows.items():
        copy(source/'run'/'receipts'/(cell_id+'.json'),'run/receipts/'+cell_id+'.json')
        for key in ('log','output'):
            if row.get(key):copy(artifact(source/'run',row[key]),'run/'+row[key]['path'].replace('\\','/'))
    for path in (source/'run'/'parity').iterdir():
        need(path.is_file(),'unexpected parity archive directory');copy(path,'run/parity/'+path.name)
    for path in (source/'run').glob('calibration-background-*.json'):copy(path,'run/'+path.name)
    claims=root.parent/'calibration-repairs';physical(claims);claims.mkdir(exist_ok=True)
    claim=claims/(request['protocol_sha256']+'.json');physical(claim)
    need(not claim.exists(),f'refusing to overwrite {claim}')
    bootstrap.write(claim,dict(source_protocol_sha256=request['protocol_sha256'],source_calibration_sha256=request['calibration_sha256'],destination_campaign=root.name,destination_prepared_sha256=manifest['preparation']['sha256'],authorization_id=auth['id'],responsible_chat=auth['coordinator']))
    copy(claim,'repair-claim.json')
    claim_binding=dict(path=str(claim.resolve()),sha256=file_hash(claim))
    value=seal(dict(source=request,files=files,claim=claim_binding,compatibility=dict(new_runtime=manifest['runtime'],new_hardware=manifest['hardware'],new_prepared=manifest['preparation'],new_moss_configs=config_binding(manifest)),
        limitations='Original MOSS costs retain original process/desktop conditions and original protocol identity; only three native startup cells may execute in this new campaign.'))
    write(archive/'bundle.json',value)
    manifest['calibration_carry_forward']=dict(path=DIRECTORY+'/bundle.json',sha256=file_hash(archive/'bundle.json'),claim=claim_binding)
    carried(manifest,root/'run')
    return archive
