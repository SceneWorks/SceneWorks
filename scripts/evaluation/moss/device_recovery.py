"""One fresh GPU0 candidate after the exact invalidated v8; no evidence imports."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import urllib.request
from urllib.parse import urlparse
import zipfile

from ownership import physical, require, digest, read

MANIFEST_SHA256='ba0c75efe4ff8082004813d63a98f522c7bdd121f315dc76c74f0d2e2bd2a403'
MANIFEST=Path(__file__).with_name('prior-v8-failed-artifact.manifest.json')
DIRECTORY='prior-failed-attempt'
ACCOUNTING=dict(prior_calibration_ok_labels=6,prior_scored_ok_labels=51,prior_individually_invalidated_scored_ok_labels=3,prior_failed_calibration_pngs=1,prior_rows_imported=0,current_calibration_cap=6,current_scored_cap=288,later_native_probe_cap=24,current_successful_output_cap=318,lifetime_ok_label_upper_bound=375,lifetime_png_upper_bound_including_failed=376)


def authority():
    require(digest(MANIFEST)==MANIFEST_SHA256,'reviewed failed-v8 manifest changed')
    value=read(MANIFEST)
    require(value['current_rows_import_permitted']==0 and value['original_run']==39047172756 and value['artifact_id']==12106733806,'fixed failed candidate authority changed')
    return value


def validate_control(value):
    request=value['device_recovery']
    require(value['campaign']=='sc-26322-reference-v9' and set(request)=={'kind','manifest_sha256','authority','responsible_chat'},'only explicit fresh v8-to-v9 device recovery supported')
    require(request['kind']=='fresh_gpu0_v8' and request['manifest_sha256']==MANIFEST_SHA256 and all(isinstance(request[k],str) and request[k].strip() for k in ('authority','responsible_chat')),'explicit coordinator fresh-device authority required')
    reuse=value.get('reuse',{})
    require(reuse.get('native_copy',{}).get('kind')=='fresh_gpu0_v8' and reuse.get('campaign')=='sc-26322-reference-v1','fresh device requires immutable original inputs and exact v8 binary copy')
    require(not reuse.get('native_rebuild'),'fresh device may not rebuild evaluated binary')
    authority()


def download_zip(identity):
    """Fixed GitHub API artifact only; token is stripped on signed redirects."""
    base='https://api.github.com/repos/SceneWorks/SceneWorks/actions/artifacts/'+str(identity['artifact_id'])
    headers={'Accept':'application/vnd.github+json','Authorization':'Bearer '+os.environ['GH_TOKEN']}
    class NoTokenRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self,req,fp,code,msg,redirect_headers,newurl):
            parsed=urlparse(newurl)
            require(parsed.scheme=='https' and not parsed.username and not parsed.password and parsed.port in (None,443) and (parsed.hostname=='api.github.com' or parsed.hostname.endswith('.blob.core.windows.net') or parsed.hostname.endswith('.actions.githubusercontent.com')),'untrusted failed-v8 artifact redirect')
            redirected=super().redirect_request(req,fp,code,msg,redirect_headers,newurl)
            redirected.remove_header('Authorization');return redirected
    opener=urllib.request.build_opener(NoTokenRedirect())
    with opener.open(urllib.request.Request(base,headers=headers),timeout=30) as response:meta=json.loads(response.read(65537))
    require(meta['id']==identity['artifact_id'] and meta['workflow_run']['id']==identity['original_run'] and meta['workflow_run']['head_sha']==identity['head'] and meta['workflow_run']['repository_id']==1241052736 and meta['name']=='moss-control-39047172756-1' and meta['size_in_bytes']==identity['original_zip_bytes'] and meta['digest']=='sha256:'+identity['original_zip_sha256'] and not meta['expired'],'fixed failed-v8 artifact API identity changed')
    try:
        with opener.open(urllib.request.Request(base+'/zip',headers=headers),timeout=120) as response:data=response.read(identity['original_zip_bytes']+1)
    except Exception as error:
        raise ValueError('fixed failed-v8 ZIP download failed: '+type(error).__name__) from None
    require(len(data)==identity['original_zip_bytes'] and hashlib.sha256(data).hexdigest()==identity['original_zip_sha256'],'failed-v8 ZIP size/hash changed')
    return data


def install(root,request):
    import io
    value=authority();source=root.parent/'sc-26322-reference-v8'
    target=root/DIRECTORY;physical(source);physical(target)
    require(not target.exists() and not (root/'run').exists(),'fresh device candidate already has evidence; never import current rows')
    data=download_zip(value)
    require(len(data)==value['original_zip_bytes'] and hashlib.sha256(data).hexdigest()==value['original_zip_sha256'],'failed-v8 ZIP identity changed before archive')
    rows={f['path']:f for f in value['files']}
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        infos=archive.infolist()
        require(len(infos)<=1024 and sum(i.file_size for i in infos)<=256*1024**2,'failed-v8 ZIP entry/byte cap exceeded')
        members=[i for i in infos if not i.is_dir()]
        require(len(members)==len(rows)==len({i.filename for i in members}) and set(rows)=={i.filename for i in members},'failed-v8 ZIP file closure changed')
        # Validate all inputs before creating a destination or copying bytes.
        for item in members:
            path=PurePosixPath(item.filename)
            require(not path.is_absolute() and '\\' not in item.filename and ':' not in item.filename and all(p not in ('','.','..') for p in item.filename.split('/')) and ((item.external_attr>>16)&0o170000)!=0o120000,'unsafe failed-v8 ZIP member')
            expected=rows[item.filename];old=source/item.filename;physical(old)
            require(old.is_file() and item.file_size==expected['bytes']==old.stat().st_size and digest(old)==expected['sha256'],'original selected v8 file changed: '+item.filename)
            require(hashlib.sha256(archive.read(item)).hexdigest()==expected['sha256'],'failed-v8 ZIP entry changed')
        target.mkdir()
        shutil.copyfile(MANIFEST,target/'manifest.json')
        with (target/'original.zip').open('xb') as stream:stream.write(data)
        for name in rows:
            dest=target/'files'/name;physical(dest);dest.parent.mkdir(parents=True,exist_ok=True)
            with (source/name).open('rb') as old,dest.open('xb') as new:shutil.copyfileobj(old,new)
            require(digest(dest)==rows[name]['sha256'],'failed-v8 copy changed bytes')
    binding=dict(kind='fresh_gpu0_v8',path=DIRECTORY,manifest_sha256=MANIFEST_SHA256,authority=request['authority'],responsible_chat=request['responsible_chat'],current_rows_imported=0,
                 accounting=ACCOUNTING,
                 limitations='Prior six calibration successes and51 scored ok-labeled rows retain their original statuses; three scored rows are individually invalidated. All are historical and zero are imported as this candidate acceptance. One prior failed calibration PNG is retained. Replacement authorization permits fresh318, lifetime375 ok-labeled/376 including failed upper bounds. Prior artifact is an exact selected subset of the original Windows root, not the whole root. Old timings, parity, invalidations and claim chain remain historical only.')
    verify(root,binding)
    return binding


def verify(root,binding):
    value=authority();target=root/DIRECTORY
    require(binding['kind']=='fresh_gpu0_v8' and binding['path']==DIRECTORY and binding['manifest_sha256']==MANIFEST_SHA256 and binding['current_rows_imported']==0,'fresh device provenance changed')
    require(binding['accounting']==ACCOUNTING,'fresh-device output budget accounting changed')
    physical(target)
    expected={f['path']:f for f in value['files']}
    require(digest(target/'manifest.json')==MANIFEST_SHA256 and digest(target/'original.zip')==value['original_zip_sha256'],'failed-v8 archive identity changed')
    for path in target.rglob('*'):physical(path)
    actual={p.relative_to(target).as_posix() for p in target.rglob('*') if p.is_file()}
    require(actual=={'manifest.json','original.zip'}|{'files/'+n for n in expected},'failed-v8 preserved file closure changed')
    for name,item in expected.items():
        p=target/'files'/name;physical(p)
        require(p.stat().st_size==item['bytes'] and digest(p)==item['sha256'],'failed-v8 preserved bytes changed')
    return value


def verify_manifest(manifest):
    root=Path(manifest['preparation']['path']).parent
    require(root.name=='sc-26322-reference-v9' and manifest['hardware'].get('physical_gpu_index')==0 and not manifest.get('calibration_carry_forward'),'fresh GPU0 candidate cannot inherit calibration, scored or parity records')
    binding=manifest['device_recovery'];verify(root,binding)
    require(binding['responsible_chat']==manifest['ownership']['responsible_chat'],'fresh-device authority owner changed')
    require(all(not c['settings'].get('calibration_source_root') for c in manifest['configs']),'fresh device cannot import prior calibration lookups')
    require(all(c['settings'].get('prior_failed_attempt_root')==str(root/DIRECTORY) for c in manifest['configs']),'prior failed evidence is outside frozen input closure')
