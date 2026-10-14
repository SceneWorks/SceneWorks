"""CPU cache preparation through existing product model_download jobs; no generation."""
import ctypes,errno,fcntl,hashlib,importlib.util,json,os,pwd,re,shutil,signal,stat,subprocess,sys,time
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
    no_links(t)
    with t.open('x') as f:json.dump(v,f,indent=2);f.flush();os.fsync(f.fileno())
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
def apfs_volume_info(root,check=lambda:None):
    import plistlib
    require(sys.platform=='darwin','APFS identity requires Darwin');no_links(root);before=root.stat();check()
    rows=subprocess.check_output(['/bin/df','-P',str(root)],timeout=15).decode('utf-8').splitlines();check()
    require(len(rows)==2,'filesystem lookup did not identify exactly one volume');fields=rows[1].split(maxsplit=5)
    require(len(fields)==6 and re.fullmatch(r'/dev/disk\d+s\d+(?:s\d+)*',fields[0]) is not None,'filesystem lookup returned an unqualified device')
    device=Path(fields[0]);no_links(device);device_stat=device.stat();require(stat.S_ISBLK(device_stat.st_mode) and device_stat.st_rdev==before.st_dev,'filesystem device differs from owned path')
    info=plistlib.loads(subprocess.check_output(['/usr/sbin/diskutil','info','-plist',str(device)],timeout=15));check()
    mount=Path(fields[5]);require(mount.is_absolute() and info.get('DeviceNode')==str(device) and info.get('DeviceIdentifier')==device.name and info.get('MountPoint')==str(mount),'disk information differs from filesystem lookup');no_links(mount)
    after=root.stat();require((before.st_dev,before.st_ino)==(after.st_dev,after.st_ino) and mount.stat().st_dev==before.st_dev and device.stat()==device_stat,'owned filesystem identity changed during lookup')
    require(info.get('FilesystemType','').lower()=='apfs' and re.fullmatch(r'[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}',str(info.get('VolumeUUID',''))) is not None,'owned model volume is not observed APFS')
    return info,{'method':'df_device_authenticated_by_owned_path_st_dev_and_block_st_rdev','device':str(device),'mountPoint':str(mount),'pathDevice':before.st_dev,'volumeUuid':info['VolumeUUID']}
def prove_apfs(root,check=lambda:None):
    info,identity=apfs_volume_info(root,check)
    a=root/'.clone-probe-source';b=root/'.clone-probe-destination';require(not a.exists() and not b.exists(),'clone probe already exists')
    try:
        check();a.write_bytes(b'owned APFS proof');clone_apfs(a,b);b.write_bytes(b'private mutation');require(a.read_bytes()==b'owned APFS proof','COW source changed');check()
    finally:
        if b.exists():b.unlink()
        if a.exists():a.unlink()
    return {'method':'actual_same_volume_APFS_clonefile_and_private_mutation_probe','volume':info['VolumeUUID'],'filesystemIdentity':identity,'physicalAdmission':False}
INTERNAL_HOME=Path('/Users/MTrefry')
INTERNAL_BASE=INTERNAL_HOME/'.h3-veda-epic-25140'
MAC2_UUID='734A72AE-13DA-52A8-B0CF-F97E1D646124'
WORKSPACE_GROWTH_BYTES=64*1024*1024
FETCH_SCRATCH_BYTES=5381086712

def require_frozen_entries(entries,template):
    styles=template['styles'];rows=[a for a in template['artifacts'] if any(x['backend']=='mlx' for x in a['coverage'])]+list(template['turboArtifacts'].values())+[{**a,**a['source']} for a in styles]
    fetch={'2a8d8845c5342756a2781e8e69563940e4bb573c9a40ebb534915ff8fd76573a','b64f2289871261fdd1abbd3b78bcd66011b341de3dc8eeb2ed1a473ee7c8d95c','8d49fe5abad6195f321fc7d741cc024e006ff891a40c67844834866bcdcf9e59','0fac46be5b3498124f1bb596e9f9050338ae8028425650d5524ed40fce50f2f8'}
    expected=[]
    for a in rows:
        style=next((x for x in styles if x['source']['repo']==a['repo']),None)
        expected.append({**{k:a[k] for k in ['repo','revision','file','bytes','sha256']},'gitBlob':a.get('gitBlob'),'downloadType':'lora_download' if style else 'model_download','loraId':style['id'] if style else None,'allowManagedFetch':a['sha256'] in fetch})
    require(entries==expected and len(entries)==194 and sum(a['bytes'] for a in entries)==377576304518 and sum(a['bytes'] for a in entries if a['allowManagedFetch'])==FETCH_SCRATCH_BYTES,'full frozen194 payload/managed tuple projection differs')

def storage_policy(p):
    value=p.get('storage',{'schemaVersion':1,'mode':'same_volume_apfs_clone'})
    if value.get('mode')=='same_volume_apfs_clone':require(value=={'schemaVersion':1,'mode':'same_volume_apfs_clone'},'clone storage policy fields differ');return value
    require(value=={'schemaVersion':1,'mode':'owned_internal_cross_volume_copy','boundRoot':p['runDir'],'sourceHead':p['sourceHead'],'expectedHome':str(INTERNAL_HOME),'expectedRunner':'nax-macos-2','expectedRunnerId':5281,'expectedHostUuid':MAC2_UUID,'workspaceGrowthBytes':WORKSPACE_GROWTH_BYTES,'managedDownloadScratchBytes':FETCH_SCRATCH_BYTES},'closed internal storage policy differs')
    return value

def internal_storage_identity(p,root,shared):
    require(sys.platform=='darwin' and p['owner'].get('runnerName')=='nax-macos-2' and p['owner'].get('runnerId')==5281,'internal storage requires exact Mac2 runner')
    account=pwd.getpwuid(os.getuid());require(account.pw_name=='MTrefry' and Path(account.pw_dir)==INTERNAL_HOME and os.geteuid()==os.getuid() and os.environ.get('HOME')==str(INTERNAL_HOME),'internal storage HOME/account/UID differs')
    require(re.fullmatch('[a-f0-9]{40}',p['sourceHead']) and root==INTERNAL_BASE/('private-cache-'+p['sourceHead'][:9]) and shared==Path('/Volumes/Models/huggingface'),'exact source-owned internal/private and shared paths required')
    no_links(root);no_links(shared);ancestor=root
    while not ancestor.exists():ancestor=ancestor.parent
    require(ancestor.stat().st_uid==os.getuid() and not ancestor.stat().st_mode&0o022 and ancestor.stat().st_dev==INTERNAL_HOME.stat().st_dev and ancestor.stat().st_dev!=shared.stat().st_dev,'owned internal HOME volume differs/aliases shared volume')
    hardware=json.loads(subprocess.check_output(['system_profiler','SPHardwareDataType','-json'],timeout=15)).get('SPHardwareDataType',[])
    require(len(hardware)==1 and hardware[0].get('platform_UUID')==MAC2_UUID,'internal storage actual Mac2 UUID differs')
    return {'kind':'actual_Mac2_internal_storage_identity','uid':os.getuid(),'home':str(INTERNAL_HOME),'rootDevice':ancestor.stat().st_dev,'sharedDevice':shared.stat().st_dev,'physicalHostUuid':MAC2_UUID,'runnerId':5281,'runnerName':'nax-macos-2','sourceHead':p['sourceHead'],'physicalAdmission':False}

def workspace_allocated(root,check):
    # Existing build/tool bytes occupy filesystem space already; never read payloads.
    def walk(directory):
        total=0
        with os.scandir(directory) as entries:
            for entry in entries:
                check()
                if directory==root and entry.name=='huggingface':continue
                info=entry.stat(follow_symlinks=False);total+=getattr(info,'st_blocks',0)*512
                if stat.S_ISDIR(info.st_mode):total+=walk(Path(entry.path))
        return total
    return walk(root)

def capacity_required(entries,cache,progress):
    remaining=0
    for a in entries:
        key=a['repo']+'@'+a['revision']+'/'+a['file'];target=snapshot(cache,a);r=progress['files'].get(key)
        verified=r and target.exists() and confined(cache,target.resolve()) and target.stat().st_nlink==1 and target.stat().st_dev==cache.stat().st_dev and not target.stat().st_mode&0o222 and r.get('contentVerified') is True and r.get('bytes')==a['bytes'] and r.get('sha256')==a['sha256'] and r.get('stat')==signature(target)
        if not verified:remaining+=a['bytes']  # No discount for unverified partials.
    return remaining

def capacity_receipt(root,remaining,workspace_base,check):
    check();info=os.statvfs(root);available=info.f_bavail*info.f_frsize
    workspace=workspace_allocated(root,check);require(workspace<=workspace_base+WORKSPACE_GROWTH_BYTES,'bounded nonpayload workspace growth exceeded')
    required=remaining+FETCH_SCRATCH_BYTES+WORKSPACE_GROWTH_BYTES
    require(available>=required,'actual internal capacity insufficient for remaining full payload and declared scratch')
    return {'observedAt':int(time.time()*1000),'availableBytes':available,'remainingPayloadBytes':remaining,'managedDownloadScratchBytes':FETCH_SCRATCH_BYTES,'workspaceGrowthBudgetBytes':WORKSPACE_GROWTH_BYTES,'existingWorkspaceAllocatedBytes':workspace,'requiredFreeBytes':required,'device':root.stat().st_dev,'unverifiedPartialDiscountBytes':0}

def publish_no_replace(source,target):
    # Darwin SDK sys/stdio.h: RENAME_EXCL=0x4. Linux branch is for CPU fixtures.
    lib=ctypes.CDLL('/usr/lib/libSystem.B.dylib' if sys.platform=='darwin' else None,use_errno=True)
    if sys.platform=='darwin':
        fn=lib.renamex_np;fn.argtypes=[ctypes.c_char_p,ctypes.c_char_p,ctypes.c_uint];fn.restype=ctypes.c_int;rc=fn(os.fsencode(source),os.fsencode(target),4)
    else:
        require(sys.platform.startswith('linux') and hasattr(lib,'renameat2'),'atomic exclusive publication unavailable')
        fn=lib.renameat2;fn.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint];fn.restype=ctypes.c_int;rc=fn(-100,os.fsencode(source),-100,os.fsencode(target),1)
    if rc!=0:raise OSError(ctypes.get_errno(),'owned atomic no-replace publication refused')

def stream_private_blob(source,blob,a,check,capacity_check=lambda remaining:None):
    partial=blob.with_name(blob.name+'.partial');journal=blob.with_name(blob.name+'.partial.json');no_links(partial);no_links(journal)
    before=signature(source);expected={k:a[k] for k in ['repo','revision','file','bytes','sha256']};offset=0;h=hashlib.sha256()
    if partial.exists() or journal.exists():
        require(partial.exists() and journal.exists() and stat.S_ISREG(partial.lstat().st_mode) and stat.S_ISREG(journal.lstat().st_mode) and partial.stat().st_nlink==1 and partial.stat().st_uid==os.getuid() and journal.stat().st_nlink==1 and journal.stat().st_uid==os.getuid(),'unbound/foreign partial retained without overwrite')
        saved=json.loads(journal.read_text());require(saved.get('kind')=='owned_cross_volume_copy_prefix' and saved.get('expected')==expected and saved.get('sourcePath')==str(source) and saved.get('sourceStat')==before and saved.get('partialStat')==signature(partial),'partial source/identity changed; retained')
        offset=saved['committedBytes'];require(isinstance(offset,int) and 0<=offset<=a['bytes'] and partial.stat().st_size==offset,'partial committed length differs')
        with partial.open('rb') as f:
            for b in iter(lambda:f.read(1024*1024),b''):check();h.update(b)
        require(h.hexdigest()==saved['prefixSha256'],'partial prefix tampered; retained')
        require(not saved.get('sealed'),'sealed partial publication incomplete; reconcile rather than overwrite')
    else:
        with partial.open('xb') as f:f.flush();os.fsync(f.fileno())
    def checkpoint():
        atom(journal,{'schemaVersion':1,'kind':'owned_cross_volume_copy_prefix','expected':expected,'sourcePath':str(source),'sourceStat':before,'partialStat':signature(partial),'committedBytes':offset,'prefixSha256':h.hexdigest(),'sealed':False})
    if offset==0:checkpoint()
    with source.open('rb') as src,partial.open('r+b') as out:
        require(signature(source)==before and [os.fstat(src.fileno()).st_dev,os.fstat(src.fileno()).st_ino]==before[:2],'shared source changed before stream')
        src.seek(offset);out.seek(offset)
        while offset<a['bytes']:
            check();capacity_check(a['bytes']-offset);b=src.read(min(1024*1024,a['bytes']-offset));require(b,'source ended before declared length');out.write(b);out.flush();os.fsync(out.fileno());h.update(b);offset+=len(b);checkpoint()
        require(not src.read(1) and signature(source)==before,'shared source changed during stream')
    check();require(h.hexdigest()==a['sha256'],'copied stream SHA differs from frozen payload')
    copied=signature(partial);actual=sha(partial,check);require(actual==a['sha256'] and signature(partial)==copied and signature(source)==before,'full destination/source SHA or identity changed')
    # Publication is within the same owned volume; no replacing any existing file.
    require(not blob.exists() and not blob.is_symlink(),'private blob appeared during stream');partial.chmod(0o444);require(signature(partial)[:4]==copied[:4],'verified partial identity changed before publication');publish_no_replace(partial,blob);require(signature(blob)[:4]==copied[:4],'verified blob identity changed during publication')
    receipt={'method':'owned_cross_volume_stream_copy','sharedPath':str(source),'sharedStat':before,'privateBlob':str(blob),'sameVolume':source.stat().st_dev==blob.parent.stat().st_dev,'sharedMutated':False,'destinationVerification':{'sha256':actual,'bytes':blob.stat().st_size,'stat':signature(blob)}}
    atom(journal,{'schemaVersion':1,'kind':'owned_cross_volume_copy_prefix','expected':expected,'sourcePath':str(source),'sourceStat':before,'committedBytes':offset,'prefixSha256':h.hexdigest(),'sealed':True,'privateBlob':str(blob),'privateStat':signature(blob),'destinationSha256':actual})
    return receipt

def require_owned_device(cache,path):
    ancestor=path
    while not ancestor.exists():ancestor=ancestor.parent
    require(ancestor.stat().st_dev==cache.stat().st_dev,'owned staging path aliases a foreign volume')

def stage_file(shared,cache,a,check,clone=clone_apfs,storage_mode='same_volume_apfs_clone',capacity_check=lambda remaining:None):
    source=snapshot(shared,a);target=snapshot(cache,a);no_links(target.parent);require_owned_device(cache,target.parent);target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        resolved=target.resolve(strict=True);require(confined(cache,resolved) and resolved.is_file() and resolved.stat().st_nlink==1 and resolved.stat().st_dev==cache.stat().st_dev,'owned cache points to foreign/shared inode');return target,None
    blob=cache/'hub'/('models--'+a['repo'].replace('/','--'))/'blobs'/(a.get('gitBlob') or a['sha256']);no_links(blob.parent);require_owned_device(cache,blob.parent);blob.parent.mkdir(parents=True,exist_ok=True)
    if blob.exists() or blob.is_symlink():
        require(not blob.is_symlink() and blob.stat().st_nlink==1 and blob.stat().st_dev==cache.stat().st_dev and blob.stat().st_size==a['bytes'] and sha(blob,check)==a['sha256'],'partial owned clone differs; retained without overwrite')
        os.chmod(blob,0o444);target.symlink_to(os.path.relpath(blob,target.parent));return target,{'method':'recovered_complete_private_blob_after_SHA','sharedMutated':False}
    if not source.exists():require(a['allowManagedFetch'],'unexpected missing source file, not approved four-file download');return target,None
    resolved=source.resolve(strict=True);require(confined(shared,resolved) and resolved.is_file() and resolved.stat().st_size==a['bytes'],'shared source path/size differs')
    if storage_mode=='owned_internal_cross_volume_copy':
        require(resolved.stat().st_dev==shared.stat().st_dev and cache.stat().st_dev==blob.parent.stat().st_dev==target.parent.stat().st_dev,'staging path crosses unqualified volume alias')
        origin=stream_private_blob(resolved,blob,a,check,capacity_check);target.symlink_to(os.path.relpath(blob,target.parent));return target,origin
    require(storage_mode=='same_volume_apfs_clone','unknown storage staging mode')
    before=signature(resolved);clone(resolved,blob);os.chmod(blob,0o444);check();require(signature(resolved)==before,'shared source changed during clone');target.symlink_to(os.path.relpath(blob,target.parent))
    return target,{'method':'APFS_COW_clone','sharedPath':str(resolved),'sharedStat':before,'privateBlob':str(blob),'sameVolume':True,'sharedMutated':False}
def verify_file(target,a,check,verified=None):
    require(target.exists() and target.stat().st_size==a['bytes'] and (verified=={'sha256':a['sha256'],'bytes':a['bytes'],'stat':signature(target)} if verified else sha(target,check)==a['sha256']),'frozen payload SHA/size mismatch')
    os.chmod(target.resolve(),0o444)
    return {'repo':a['repo'],'revision':a['revision'],'file':a['file'],'path':str(target),'bytes':a['bytes'],'sha256':a['sha256'],'stat':signature(target),'contentVerified':True}
def owned_hf_environment(cache):
    cache=Path(cache)
    return {'HF_HOME':str(cache),'HUGGINGFACE_HUB_CACHE':str(cache/'hub'),'HF_HUB_CACHE':str(cache/'hub'),'HF_ASSETS_CACHE':str(cache/'assets'),'HF_XET_CACHE':str(cache/'xet'),'TRANSFORMERS_CACHE':str(cache/'transformers')}
def require_owned_hf_environment(env,cache):
    require(all(env.get(k)==v for k,v in owned_hf_environment(cache).items()),'actual CPU HF cache environment escaped owned root')
def cpu_environment(base,cache,kind='api'):
    env=dict(base);env.update(owned_hf_environment(cache));env.update(SCENEWORKS_GPU_ID='cpu',SCENEWORKS_RUST_WORKER_GPU_ID='cpu',SCENEWORKS_UTILITY_WORKERS='1',SCENEWORKS_UTILITY_JOBS='1',SCENEWORKS_WORKER_ONLY='1' if kind=='download_worker' else '0')
    for k in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_TOKEN','HUGGING_FACE_HUB_TOKEN','HF_ENDPOINT']:env.pop(k,None)
    require_owned_hf_environment(env,cache);return env

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
        if not p.exists() or p.stat().st_dev!=cache.stat().st_dev or not confined(cache,p.resolve()) or r.get('contentVerified') is not True or r.get('path')!=str(p) or r.get('bytes')!=a['bytes'] or r.get('sha256')!=a['sha256'] or r.get('stat')!=signature(p) or p.resolve().stat().st_mode&0o222:return False
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
    phase_started=time.monotonic();policy=storage_policy(p)
    spec=importlib.util.spec_from_file_location('cache_guard_prepare',p['sourceModules']['guardian']['path']);guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
    controller_birth=guard.pid_birth(controller_pid);require(controller_birth is not None,'original CPU controller birth missing')
    root=Path(p['runDir']);cache=Path(p['cacheRoot']);shared=Path(p['sharedCache']);no_links(root);no_links(shared)
    identity=internal_storage_identity(p,root,shared) if policy['mode']=='owned_internal_cross_volume_copy' else None
    require(guard.pid_birth(controller_pid)==controller_birth and time.monotonic()<phase_started+p['maxSeconds'],'controller/deadline changed during storage qualification')
    root.mkdir(parents=True,exist_ok=True,mode=0o700);no_links(cache);cache.mkdir(exist_ok=True)
    if policy['mode']=='owned_internal_cross_volume_copy':require(cache.stat().st_uid==os.getuid() and cache.stat().st_dev==root.stat().st_dev,'internal cache aliases a foreign owner/device')
    for name in ['cache-owner.json','cache-progress.json','cache-prepare.lock']:
        state=root/name;no_links(state)
        if state.exists():
            info=state.lstat();require(stat.S_ISREG(info.st_mode) and info.st_uid==os.getuid() and info.st_nlink==1,'owned state owner/type/hardlink alias differs')
    require(not confined(shared,root) and not confined(root,shared) and cache==root/'huggingface','shared/owned cache paths overlap')
    expected={'schemaVersion':1,'templateSha256':p['template']['sha256'],'catalogSha256':p['catalog']['sha256'],'sharedCache':str(shared),'cacheRoot':str(cache),**({'storage':p['storage'],'sourceHead':p['sourceHead']} if p.get('storage',{}).get('mode')=='owned_internal_cross_volume_copy' else {})};marker=root/'cache-owner.json'
    if marker.exists():require(json.loads(marker.read_text())==expected,'owned cache belongs to different frozen closure')
    else:atom(marker,expected)
    lock=(root/'cache-prepare.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:return prepare_locked(p,controller_pid,root,cache,shared,lock,phase_started,identity,guard,controller_birth)
    finally:lock.close()  # Inherited owned children retain the same lease if drainage is uncertain.

def prepare_locked(p,controller_pid,root,cache,shared,cache_lock,phase_started=None,identity=None,guard=None,original_controller_birth=None):
    deadline=(phase_started if phase_started is not None else time.monotonic())+p['maxSeconds'];controller_birth=None;stop=False
    if guard is None:
        spec=importlib.util.spec_from_file_location('cache_guard',p['sourceModules']['guardian']['path']);guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
    g=guard;controller_birth=original_controller_birth if original_controller_birth is not None else g.pid_birth(controller_pid);require(g.pid_birth(controller_pid)==controller_birth,'original controller changed before phase')
    last_probe=[0.0];quota_base=[None];quota_probe=[0.0];require(controller_birth is not None,'live CPU controller identity required')
    def check():
        now=time.monotonic()
        if p.get('storage',{}).get('mode')=='owned_internal_cross_volume_copy':
            info=os.statvfs(root);require(info.f_bavail*info.f_frsize>=FETCH_SCRATCH_BYTES+WORKSPACE_GROWTH_BYTES,'actual internal scratch capacity exhausted')
        if stop or now>=deadline:raise TimeoutError('bounded CPU cache phase/controller ended')
        if quota_base[0] is not None and now-quota_probe[0]>=1:
            quota_probe[0]=now
            def bounded_scan():
                if stop or time.monotonic()>=deadline:raise TimeoutError('bounded workspace scan deadline')
            require(workspace_allocated(root,bounded_scan)<=quota_base[0]+WORKSPACE_GROWTH_BYTES,'bounded nonpayload workspace growth exceeded')
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
        storage_mode=storage_policy(p)['mode'];check()
        apfs=prove_apfs(root,check)
        if storage_mode=='owned_internal_cross_volume_copy':
            progress['storageIdentity']=identity or internal_storage_identity(p,root,shared);check();build_root=Path(p['binaryAttestation']['path']).parent;no_links(build_root);progress['existingBuildWorkspace']={'path':str(build_root),'allocatedBytes':workspace_allocated(build_root,check),'device':build_root.stat().st_dev,'alreadyOccupiedNotAdditionalReserve':True};workspace_base=workspace_allocated(root,check);quota_base[0]=workspace_base;progress['capacity']=capacity_receipt(root,capacity_required(p['entries'],cache,progress),workspace_base,check);atom(receipt_path,progress)
        else:require(root.stat().st_dev==shared.stat().st_dev,'shared/owned cache must be same APFS volume')
        for a in p['entries']:
            check();key=a['repo']+'@'+a['revision']+'/'+a['file'];target=snapshot(cache,a);prior=progress['files'].get(key)
            if prior and target.exists() and prior['stat']==signature(target) and target.resolve().stat().st_nlink==1 and target.stat().st_dev==cache.stat().st_dev and target.resolve().stat().st_mode&0o222==0 and confined(cache,target.resolve()):continue
            if storage_mode=='owned_internal_cross_volume_copy':progress['capacity']=capacity_receipt(root,capacity_required(p['entries'],cache,progress),workspace_base,check)
            remaining=capacity_required(p['entries'],cache,progress) if storage_mode=='owned_internal_cross_volume_copy' else 0
            def capacity_tick(file_remaining):
                if storage_mode=='owned_internal_cross_volume_copy':
                    check();info=os.statvfs(root);require(info.f_bavail*info.f_frsize>=remaining-a['bytes']+file_remaining+FETCH_SCRATCH_BYTES+WORKSPACE_GROWTH_BYTES,'actual capacity changed during bounded stream')
            target,origin=stage_file(shared,cache,a,check,storage_mode=storage_mode,capacity_check=capacity_tick)
            if target.exists():progress['files'][key]={**verify_file(target,a,check,(origin or {}).get('destinationVerification')),'origin':origin or {'method':'owned_reverified_private_file'}};atom(receipt_path,progress)
        if storage_mode=='owned_internal_cross_volume_copy':progress['capacity']=capacity_receipt(root,capacity_required(p['entries'],cache,progress),workspace_base,check)
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
                    return cpu_environment(super().env('api'),cache,kind)
            class NoPhysicalLocks: handles=[cache_lock]
            class CpuOnly: locks=NoPhysicalLocks()
            import socket
            with socket.socket() as probe:probe.bind(('127.0.0.1',port))
            services=CpuServices(m,'mlx',root,CpuOnly());require_owned_hf_environment(services.env('api'),cache);progress['ownedHfEnvironment']=owned_hf_environment(cache);services.start('api');api=g.Product(m['runners']['mlx']['apiUrl'],token)
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
            require_owned_hf_environment(services.env('download_worker'),cache);services.start('download_worker');startup.append(observe('after-startup'));progress['startupEvidence']=startup;atom(receipt_path,progress)
            for a in [a for a in p['entries'] if a in missing or hashlib.sha256((a['repo']+'@'+a['revision']+'/'+a['file']).encode()).hexdigest() in progress['jobs']]:
                if storage_mode=='owned_internal_cross_volume_copy':progress['capacity']=capacity_receipt(root,capacity_required(p['entries'],cache,progress),workspace_base,check)
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
        if storage_mode=='owned_internal_cross_volume_copy':progress['capacity']=capacity_receipt(root,capacity_required(p['entries'],cache,progress),workspace_base,check)
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
    require(os.getppid()==int(sys.argv[2]),'actual CPU controller parent differs')
    require(sha(sys.executable)==os.environ.get('S9_CACHE_PYTHON_SHA256'),'actual owned CPU interpreter changed')
    require(sha(sys.argv[1])==sys.argv[3],'CPU plan changed after exact source assembly')
    plan=json.loads(Path(sys.argv[1]).read_text());require(type(plan.get('maxSeconds')) is int and 0<plan['maxSeconds']<=480,'original bounded480s phase required');storage_policy(plan);require(plan['kind']=='epic25140_owned_managed_cache_plan' and plan['backend']=='mlx' and sys.platform=='darwin','source-owned Mac2 managed CPU plan required')
    for b in [plan['template'],plan['catalog'],plan['loraCatalog'],plan['binary'],plan['metallib'],plan['binaryAttestation'],*plan['sourceModules'].values()]:require(sha(b['path'])==b['sha256'] and Path(b['path']).stat().st_size==b['bytes'],'actual source/runtime binding changed')
    require_frozen_entries(plan['entries'],json.loads(Path(plan['template']['path']).read_text()))
    print(json.dumps(prepare(plan,int(sys.argv[2]))))
