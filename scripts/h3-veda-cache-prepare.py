"""CPU cache preparation through existing product model_download jobs; no generation."""
import ctypes,errno,fcntl,hashlib,importlib.util,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path

def require(ok,why):
    if not ok:raise RuntimeError('H3 owned cache: '+why)
def sha(p,check=lambda:None):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):check();h.update(b)
    return h.hexdigest()
def atom(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp')
    with t.open('w') as f:json.dump(v,f,indent=2);f.flush();os.fsync(f.fileno())
    t.replace(p)
def confined(root,p):return root.resolve() in p.resolve().parents
def no_links(p):
    for a in [p,*p.parents]:require(not a.is_symlink(),'symlink directory ancestor')
def signature(p):
    s=p.stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_mode]
def snapshot(cache,a):return cache/'hub'/('models--'+a['repo'].replace('/','--'))/'snapshots'/a['revision']/a['file']
def clone_apfs(source,target):
    require(sys.platform=='darwin' and source.stat().st_dev==target.parent.stat().st_dev,'APFS clone requires actual same volume; no full-copy fallback')
    lib=ctypes.CDLL('/usr/lib/libSystem.B.dylib',use_errno=True);fn=lib.clonefile;fn.argtypes=[ctypes.c_char_p,ctypes.c_char_p,ctypes.c_int];fn.restype=ctypes.c_int
    if fn(os.fsencode(source),os.fsencode(target),0)!=0:raise OSError(ctypes.get_errno(),'APFS clonefile refused')
    require(source.stat().st_ino!=target.stat().st_ino,'clone unexpectedly hard-links shared inode')
def prove_apfs(root):
    import plistlib
    info=plistlib.loads(subprocess.check_output(['diskutil','info','-plist',str(root)],timeout=15));require('APFS' in str(info.get('FilesystemType','')).upper()+str(info.get('FilesystemName','')).upper(),'owned model volume is not observed APFS')
    a=root/'.clone-probe-source';b=root/'.clone-probe-destination';require(not a.exists() and not b.exists(),'clone probe already exists')
    try:
        a.write_bytes(b'owned APFS proof');clone_apfs(a,b);b.write_bytes(b'private mutation');require(a.read_bytes()==b'owned APFS proof','COW source changed')
    finally:
        if b.exists():b.unlink()
        if a.exists():a.unlink()
    return {'method':'actual_same_volume_APFS_clonefile_and_private_mutation_probe','volume':info.get('VolumeUUID'),'physicalAdmission':False}
def stage_file(shared,cache,a,check,clone=clone_apfs):
    source=snapshot(shared,a);target=snapshot(cache,a);no_links(target.parent);target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        resolved=target.resolve(strict=True);require(confined(cache,resolved) and resolved.is_file() and resolved.stat().st_nlink==1,'owned cache points to foreign/shared inode');return target,None
    blob=cache/'hub'/('models--'+a['repo'].replace('/','--'))/'blobs'/(a.get('gitBlob') or a['sha256']);no_links(blob.parent);blob.parent.mkdir(parents=True,exist_ok=True)
    if blob.exists():
        require(not blob.is_symlink() and blob.stat().st_nlink==1 and blob.stat().st_size==a['bytes'] and sha(blob,check)==a['sha256'],'partial owned clone differs; retained without overwrite')
        os.chmod(blob,0o444);target.symlink_to(os.path.relpath(blob,target.parent));return target,{'method':'recovered_complete_private_blob_after_SHA','sharedMutated':False}
    if not source.exists():require(a['allowManagedFetch'],'unexpected missing source file, not approved four-file download');return target,None
    resolved=source.resolve(strict=True);require(confined(shared,resolved) and resolved.is_file() and resolved.stat().st_size==a['bytes'],'shared source path/size differs')
    before=signature(resolved);clone(resolved,blob);os.chmod(blob,0o444);check();require(signature(resolved)==before,'shared source changed during clone');target.symlink_to(os.path.relpath(blob,target.parent))
    return target,{'method':'APFS_COW_clone','sharedPath':str(resolved),'sharedStat':before,'privateBlob':str(blob),'sameVolume':True,'sharedMutated':False}
def verify_file(target,a,check):
    require(target.exists() and target.stat().st_size==a['bytes'] and sha(target,check)==a['sha256'],'frozen payload SHA/size mismatch')
    os.chmod(target.resolve(),0o444)
    return {'repo':a['repo'],'revision':a['revision'],'file':a['file'],'path':str(target),'bytes':a['bytes'],'sha256':a['sha256'],'stat':signature(target),'contentVerified':True}
def request_shape(a,key):
    # Exactly the existing normal ModelDownload executor. Predictor authority is re-derived
    # from builtin managed_file; client SHA cannot replace its catalog identity.
    return {'type':a.get('downloadType','model_download'),'requestedGpu':'cpu','payload':{'loraId':a.get('loraId'),'modelId':'minimax_h3','repo':a['repo'],'revision':a['revision'],'files':[a['file']],'expectedSha256':a['sha256'],'expectedSizeBytes':a['bytes'],'licenseAcknowledged':True,'cachePreparationKey':key}}
def verify_job(j,a,key):
    p=j.get('payload',{});require(j.get('type')==a.get('downloadType','model_download') and j.get('requestedGpu')=='cpu' and p.get('cachePreparationKey')==key and p.get('repo')==a['repo'] and p.get('revision')==a['revision'] and p.get('files')==[a['file']],'normal managed job/request identity differs')
    if a['repo'].startswith('Veda-Sparse/'):require(p.get('expectedSha256')==a['sha256'] and p.get('expectedSizeBytes')==a['bytes'],'managed predictor catalog digest not preserved')

def cleanup(services,api,active,progress):
    errors=[];closures=[]
    try:
        if active and api:
            j,a,key=active
            try:
                j=api('GET','/api/v1/jobs/'+j['id']);verify_job(j,a,key)
                if j['status'] not in ['completed','failed','canceled','interrupted']:api('POST','/api/v1/jobs/'+j['id']+'/cancel',{})
                progress['jobs'][key]['lastReadback']=j
            except Exception as e:errors.append('owned job cancellation/readback uncertainty: '+str(e))
    finally:
        if services:
            for name in ['download_worker','api']:
                try:
                    v=services.stop(name);require(not v or v['reaped'] and v['descendantsDrained'],'owned CPU tree closure incomplete');closures.append({'role':name,'closure':v})
                except Exception as e:errors.append('owned '+name+' drain uncertainty: '+str(e))
    progress['servicesClosure']=closures;progress['cleanupErrors']=errors
    progress['servicesDrained']=not services or not services.children
    return errors
def complete_files(progress,entries,cache):
    expected={a['repo']+'@'+a['revision']+'/'+a['file']:a for a in entries}
    if set(progress['files'])!=set(expected):return False
    for key,a in expected.items():
        r=progress['files'][key];p=snapshot(cache,a)
        if not p.exists() or not confined(cache,p.resolve()) or r.get('contentVerified') is not True or r.get('path')!=str(p) or r.get('bytes')!=a['bytes'] or r.get('sha256')!=a['sha256'] or r.get('stat')!=signature(p) or p.resolve().stat().st_mode&0o222:return False
    return True
def complete_jobs(progress,entries):
    allowed={hashlib.sha256((a['repo']+'@'+a['revision']+'/'+a['file']).encode()).hexdigest():a for a in entries if a['allowManagedFetch']}
    if progress.get('uncertainPost') or not set(progress['jobs']).issubset(allowed):return False
    try:
        for key,r in progress['jobs'].items():
            j=r.get('readback',{});verify_job(j,allowed[key],key)
            if j.get('id')!=r['jobId'] or j.get('status')!='completed':return False
    except (RuntimeError,KeyError,TypeError):return False
    return True

def prepare(p,controller_pid):
    root=Path(p['runDir']);cache=Path(p['cacheRoot']);shared=Path(p['sharedCache']);no_links(root);no_links(shared);root.mkdir(parents=True,exist_ok=True);cache.mkdir(exist_ok=True)
    require(not confined(shared,root) and not confined(root,shared) and cache==root/'huggingface','shared/owned cache paths overlap')
    expected={'schemaVersion':1,'templateSha256':p['template']['sha256'],'catalogSha256':p['catalog']['sha256'],'sharedCache':str(shared),'cacheRoot':str(cache)};marker=root/'cache-owner.json'
    if marker.exists():require(json.loads(marker.read_text())==expected,'owned cache belongs to different frozen closure')
    else:atom(marker,expected)
    lock=(root/'cache-prepare.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:return prepare_locked(p,controller_pid,root,cache,shared,lock)
    finally:lock.close()  # Inherited owned children retain the same lease if drainage is uncertain.

def prepare_locked(p,controller_pid,root,cache,shared,cache_lock):
    deadline=time.monotonic()+p['maxSeconds'];controller_birth=None;stop=False
    spec=importlib.util.spec_from_file_location('cache_guard',p['sourceModules']['guardian']['path']);g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
    controller_birth=g.pid_birth(controller_pid);last_probe=[0.0];require(controller_birth is not None,'live CPU controller identity required')
    def check():
        now=time.monotonic()
        if stop or now>=deadline:raise TimeoutError('bounded CPU cache phase/controller ended')
        if now-last_probe[0]>=1:
            last_probe[0]=now
            if g.pid_birth(controller_pid)!=controller_birth:raise TimeoutError('owned CPU controller identity ended')
    def cancel(signum,frame):
        nonlocal stop;stop=True
    signal.signal(signal.SIGTERM,cancel);signal.signal(signal.SIGINT,cancel)
    receipt_path=root/'cache-progress.json';progress=json.loads(receipt_path.read_text()) if receipt_path.exists() else {'schemaVersion':1,'kind':'epic25140_owned_cpu_cache_progress','files':{},'jobs':{}}
    require(progress.get('kind')=='epic25140_owned_cpu_cache_progress','foreign progress record')
    apfs=None
    services=None;api=None;active=None;error=None;startup=[]
    def observe(label):
        # Read-only actual process/device evidence. No admission or context-empty assertion
        # is invented from these raw observations; root must qualify first startup.
        outputs={}
        owned=[os.getpid(),controller_pid,*([x['child'].pid for x in services.children.values()] if services else [])]
        for name,args in [('processes',['/bin/ps','-p',','.join(map(str,owned)),'-o','pid=,ppid=,comm=']),('metalClients',['ioreg','-r','-c','IOGPU','-l','-w0']),('accelerator',['ioreg','-r','-c','AGXAccelerator','-l','-w0'])]:
            data=subprocess.check_output(args,timeout=15);file=root/(label+'-'+name+'.txt');file.write_bytes(data);outputs[name]={'path':str(file),'bytes':len(data),'sha256':sha(file)}
        return {'label':label,'observedAt':int(time.time()*1000),'evidence':outputs,'actualGpuContextVerified':False,'physicalAdmission':False}
    try:
        apfs=prove_apfs(root);require(root.stat().st_dev==shared.stat().st_dev,'shared/owned cache must be same APFS volume')
        for a in p['entries']:
            check();key=a['repo']+'@'+a['revision']+'/'+a['file'];target=snapshot(cache,a);prior=progress['files'].get(key)
            if prior and target.exists() and prior['stat']==signature(target) and target.resolve().stat().st_nlink==1 and target.resolve().stat().st_mode&0o222==0 and confined(cache,target.resolve()):continue
            target,origin=stage_file(shared,cache,a,check)
            if target.exists():progress['files'][key]={**verify_file(target,a,check),'origin':origin or {'method':'owned_reverified_private_file'}};atom(receipt_path,progress)
        missing=[a for a in p['entries'] if not snapshot(cache,a).exists()];require(len(missing)<=4 and all(a['allowManagedFetch'] for a in missing),'unexpected missing scope, no automatic widening')
        if missing or progress.get('uncertainPost') or any('readback' not in r for r in progress['jobs'].values()):
            startup.append(observe('before-startup'))
            token=os.urandom(32).hex();os.environ['SCENEWORKS_TOKEN']=token
            port=58143;data=root/'download-data';config=root/'download-config';(config/'manifests').mkdir(parents=True,exist_ok=True);data.mkdir(exist_ok=True)
            shutil.copyfile(p['catalog']['path'],config/'manifests/builtin.models.jsonc')
            builtin=Path(p['loraCatalog']['path']);require(sha(builtin)==p['loraCatalog']['sha256'],'ordinary LoRA catalog changed');shutil.copyfile(builtin,config/'manifests/builtin.loras.jsonc')
            m={'runners':{'mlx':{'guardian':{'dataDir':str(data),'configDir':str(config),'port':port},'apiUrl':f'http://127.0.0.1:{port}','workerId':'epic25140-owned-cpu-download','requestedGpu':'cpu','cacheRoot':str(cache),'apiBinary':p['binary'],'workerBinary':p['binary'],'nativeResources':{'mlxMetallib':p['metallib']}}},'sourceFreeze':{'sceneworks':{'root':str(Path(__file__).resolve().parent.parent)}}}
            class CpuServices(g.Services):
                def env(self,kind='api'):
                    env=super().env('api');env.update(SCENEWORKS_GPU_ID='cpu',SCENEWORKS_RUST_WORKER_GPU_ID='cpu',SCENEWORKS_UTILITY_WORKERS='1',SCENEWORKS_UTILITY_JOBS='1',SCENEWORKS_WORKER_ONLY='1' if kind=='download_worker' else '0');[env.pop(k,None) for k in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_TOKEN','HUGGING_FACE_HUB_TOKEN','HF_ENDPOINT']];return env
            class NoPhysicalLocks: handles=[cache_lock]
            class CpuOnly: locks=NoPhysicalLocks()
            import socket
            with socket.socket() as probe:probe.bind(('127.0.0.1',port))
            services=CpuServices(m,'mlx',root,CpuOnly());services.start('api');api=g.Product(m['runners']['mlx']['apiUrl'],token)
            until=min(deadline,time.monotonic()+30)
            while time.monotonic()<until:
                check()
                try:services.owned_pids();api('GET','/api/v1/workers');break
                except Exception:time.sleep(.2)
            else:raise RuntimeError('owned CPU API readiness timed out')
            catalog=api('GET','/api/v1/models');rows=catalog if isinstance(catalog,list) else catalog.get('models',[])
            model=next((r for r in rows if r.get('id')=='minimax_h3'),None);require(model is not None,'normal merged H3 model catalog missing')
            predictor=next(a for a in p['entries'] if a['repo'].startswith('Veda-Sparse/'))
            require(any(d.get('coRequisite') is True and d.get('repo')==predictor['repo'] and d.get('revision')==predictor['revision'] and d.get('files')==[predictor['file']] and d.get('expectedSha256')==predictor['sha256'] and d.get('expectedSizeBytes')==predictor['bytes'] for d in model.get('downloads',[])),'normal managed co-requisite catalog readback differs')
            # Reconcile a lost POST against the same private durable jobs DB BEFORE any retry.
            pending=progress.get('uncertainPost')
            if pending:
                jobs=api('GET','/api/v1/jobs?limit=1000');rows=jobs if isinstance(jobs,list) else jobs.get('jobs',[])
                known=set(pending.get('previousJobIds',[]));found=[j for j in rows if j.get('payload',{}).get('cachePreparationKey')==pending['key'] and j.get('id') not in known]
                require(len(found)<=1,'ambiguous duplicate owned managed POST')
                if found:progress['jobs'][pending['key']]={'jobId':found[0]['id'],'request':pending['request'],'previousJobIds':list(known)};progress.pop('uncertainPost');atom(receipt_path,progress)
                else:raise RuntimeError('lost POST absent/unknown; retained durable intent, never blindly resubmitted')
            services.start('download_worker');startup.append(observe('after-startup'));progress['startupEvidence']=startup;atom(receipt_path,progress)
            for a in [a for a in p['entries'] if a in missing or hashlib.sha256((a['repo']+'@'+a['revision']+'/'+a['file']).encode()).hexdigest() in progress['jobs']]:
                check();services.owned_pids();key=hashlib.sha256((a['repo']+'@'+a['revision']+'/'+a['file']).encode()).hexdigest();record=progress['jobs'].get(key)
                if record:
                    j=api('GET','/api/v1/jobs/'+record['jobId']);verify_job(j,a,key)
                    if j['status'] in ['canceled','failed','interrupted']:
                        previous=[*record.get('previousJobIds',[]),j['id']];progress['uncertainPost']={'key':key,'request':request_shape(a,key),'previousJobIds':previous};atom(receipt_path,progress)
                        j=api('POST','/api/v1/jobs/'+j['id']+'/retry',{});verify_job(j,a,key);progress['jobs'][key]={'jobId':j['id'],'request':request_shape(a,key),'previousJobIds':previous};progress.pop('uncertainPost');atom(receipt_path,progress)
                else:
                    require(not progress.get('uncertainPost'),'unknown prior POST cannot be repeated')
                    progress['uncertainPost']={'key':key,'request':request_shape(a,key)};atom(receipt_path,progress)
                    j=api('POST','/api/v1/jobs',request_shape(a,key));verify_job(j,a,key)
                    progress['jobs'][key]={'jobId':j['id'],'request':request_shape(a,key)};progress.pop('uncertainPost',None);atom(receipt_path,progress)
                active=(j,a,key)
                while j['status'] not in g.TERMINAL:
                    check();time.sleep(.5);j=api('GET','/api/v1/jobs/'+j['id']);verify_job(j,a,key)
                require(j['status']=='completed','normal managed download did not complete');progress['jobs'][key]['readback']=j;target=snapshot(cache,a);progress['files'][a['repo']+'@'+a['revision']+'/'+a['file']]={**verify_file(target,a,check),'origin':{'method':'normal_SceneWorks_'+a.get('downloadType','model_download'),'jobId':j['id'],'binarySha256':p['binary']['sha256']}};active=None;atom(receipt_path,progress)
    except Exception as e:error=type(e).__name__+': '+str(e)
    finally:
        errors=cleanup(services,api,active,progress)
        if services:
            try:startup.append(observe('after-drain'))
            except Exception as e:errors.append('after-drain device observation failed: '+str(e))
        progress['startupEvidence']=startup
        progress.update(sourceHead=p['sourceHead'],owner=p['owner'],templateSha256=p['template']['sha256'],catalogSha256=p['catalog']['sha256'],cacheRoot=str(cache),requiredFiles=len(p['entries']),verifiedFiles=len(progress['files']),requiredBytes=sum(a['bytes'] for a in p['entries']),APFS=apfs,phaseEnded=error,campaignReady=False,physicalAdmission=False,resourceQueueVerified=False,actualStartupGpuContextVerified=False,sourceCpuUtilityGuard=True,baselineIdentities=11118,optionalStressAdded=0)
        progress['contentVerified']=not error and not errors and progress['servicesDrained'] and complete_files(progress,p['entries'],cache) and complete_jobs(progress,p['entries']);atom(receipt_path,progress)
    if error or progress['cleanupErrors'] or not progress['contentVerified']:raise RuntimeError(error or '; '.join(progress['cleanupErrors']) or 'full194 cache preparation remains incomplete')
    return progress

if __name__=='__main__':
    require(sha(sys.executable)==os.environ.get('S9_CACHE_PYTHON_SHA256'),'actual owned CPU interpreter changed')
    require(sha(sys.argv[1])==sys.argv[3],'CPU plan changed after exact source assembly')
    plan=json.loads(Path(sys.argv[1]).read_text());require(plan['kind']=='epic25140_owned_managed_cache_plan' and plan['backend']=='mlx' and sys.platform=='darwin','source-owned Mac2 managed CPU plan required')
    for b in [plan['template'],plan['catalog'],plan['loraCatalog'],plan['binary'],plan['metallib'],plan['binaryAttestation'],*plan['sourceModules'].values()]:require(sha(b['path'])==b['sha256'] and Path(b['path']).stat().st_size==b['bytes'],'actual source/runtime binding changed')
    print(json.dumps(prepare(plan,int(sys.argv[2]))))
