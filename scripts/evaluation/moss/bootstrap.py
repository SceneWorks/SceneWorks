#!/usr/bin/env python3
"""SC-26322 Windows transport with explicit post-calibration scored authorization.

All artifacts persist outside the checkout. Control arrives as JSON in
MOSS_CONTROL_JSON (not interpolated into a shell). See bootstrap.md.
"""
import argparse
import contextlib
import csv
import datetime as dt
from email.parser import BytesParser
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from urllib.parse import unquote, urlparse
import zipfile

from ownership import (REPOS, RUNNER_GPU_UUID, ExecutionGuard, calibration_bundle, calibration_results, availability_decisions, reject_run_invalidated, terminal_scored,
    physical, gpu_rows, compute_pids, desktop_identities, idle_device, verify_staged, validate_baseline)

SOURCE = 'ba2954f47bfd75a3186002e3cef859c4e4e1d37d'
INFERENCE = '91ffb4a819aa2990c652db2d881ba0b9b10126ce'
PIPELINE = '1dac1aca9feb1bfdfc727a401e4b67d61a505d3190706d1b7d2d784f88678b31'
BASE = Path(r'E:\sceneworks-evaluation\moss')
PACKAGES = ('torch==2.9.1+cu128', 'diffusers==0.38.0', 'transformers==5.3.0',
            'accelerate==1.10.1', 'safetensors==0.8.0', 'Pillow==12.1.1',
            'psutil==7.2.2', 'huggingface-hub==1.5.0')
DEADLINE = None
COMMAND_LOG = None


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def utc():
    return dt.datetime.now(dt.timezone.utc)


def timestamp(value):
    parsed = dt.datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(parsed.tzinfo is not None, 'UTC offset required')
    return parsed


def control(mode, raw, base=BASE):
    require(len(raw) <= 8192, 'oversized control')
    value = json.loads(raw)
    required = {'campaign', 'deadline_utc'}
    allowed = required | ({'ownership', 'gpu_uuid'} if mode == 'freeze' else
                          ({'protocol_sha256', 'lease', 'calibration_receipts_sha256', 'calibration_estimate_sha256', 'authorization'} if mode == 'scored' else {'protocol_sha256', 'lease'}) if mode in ('calibration', 'scored') else set())
    if mode=='prepare' and 'reuse' in value:allowed.add('reuse')
    if mode=='scored' and 'unavailable_configs' in value:allowed.add('unavailable_configs')
    require(set(value) == allowed, f'{mode} control fields must be exactly {sorted(allowed)}')
    require(re.fullmatch(r'sc-26322-[a-z0-9][a-z0-9-]{0,48}', value['campaign']), 'invalid campaign id')
    require(utc() < timestamp(value['deadline_utc']) <= utc() + dt.timedelta(hours=6), 'deadline must be in next six hours')
    if mode=='prepare' and 'reuse' in value:
        reuse=value['reuse']
        require(set(reuse)=={'campaign','prepared_sha256','job','job_receipt_sha256'},'reuse requires exact source campaign/receipt/job bindings')
        require(re.fullmatch(r'sc-26322-[a-z0-9][a-z0-9-]{0,48}',reuse['campaign']) and reuse['campaign']!=value['campaign'],'reuse must name a different owned sibling campaign')
        require(re.fullmatch(r'[0-9]+-[0-9]+',reuse['job']),'exact completed source job/attempt required')
        require(all(re.fullmatch(r'[0-9a-f]{64}',reuse[k]) for k in ('prepared_sha256','job_receipt_sha256')),'exact source receipt digests required')
    root = base / value['campaign']
    # Refuse reparse points (including Windows directory junctions) before writes.
    physical(root)
    if mode == 'freeze':
        require(value['gpu_uuid'] == RUNNER_GPU_UUID, 'runner evaluation is restricted to the verified physical GPU1 UUID')
        expected = {'responsible_chat', 'lease', 'job', 'timeout_seconds', 'resume', 'next_action'}
        own = value['ownership']
        require(set(own) == expected and all(own.values()), 'complete ownership required')
        require(type(own['timeout_seconds']) is int and 0 < own['timeout_seconds'] <= 1800, 'per-cell timeout exceeds calibration bound')
    if mode in ('calibration', 'scored'):
        require(re.fullmatch(r'[0-9a-f]{64}', value['protocol_sha256']), 'reviewed frozen protocol SHA256 required')
        lease = value['lease']
        require(set(lease) == {'id', 'owner', 'host', 'gpu_uuid', 'expires_utc', 'external_owners_reconciled', 'desktop_baseline', 'baseline_max_memory_mib', 'baseline_max_utilization_percent'}, 'complete device lease required')
        require(all(lease[k] for k in ('id', 'owner', 'host', 'gpu_uuid', 'expires_utc')) and lease['external_owners_reconciled'] is True, 'coordinator must reconcile ALL physical host owners')
        validate_baseline(lease)
        require(lease['gpu_uuid'] == RUNNER_GPU_UUID, 'runner evaluation is restricted to the verified physical GPU1 UUID')
        require(timestamp(lease['expires_utc']) >= timestamp(value['deadline_utc']), 'lease must cover whole deadline')
    if mode == 'scored':
        for key in ('calibration_receipts_sha256', 'calibration_estimate_sha256'):
            require(re.fullmatch(r'[0-9a-f]{64}', value[key]), 'accepted calibration digest required')
        authorization = value['authorization']
        require(set(authorization) == {'id', 'coordinator', 'accepted_calibration', 'max_scored_outputs', 'acknowledge_shape_uncertainty'}
                and authorization['id'] and authorization['coordinator'] == value['lease']['owner']
                and authorization['accepted_calibration'] is True and authorization['max_scored_outputs'] == 288
                and authorization['acknowledge_shape_uncertainty'] is True, 'explicit coordinator authorization for exactly288 scored outputs required')
    return value, root


def check_deadline():
    if DEADLINE is not None:
        require(utc() < DEADLINE, 'job deadline reached; no automatic retry')


def run(argv, *, cwd=None, env=None, timeout=None):
    check_deadline()
    if DEADLINE is not None:
        timeout = min(timeout or 21600, max(1, (DEADLINE - utc()).total_seconds()))
    result = subprocess.run([str(a) for a in argv], cwd=cwd, env=env, timeout=timeout,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if COMMAND_LOG is not None:
        with COMMAND_LOG.open('a', encoding='utf-8') as stream:
            # Commands have fixed public arguments; credential-bearing environment is never logged.
            stream.write(result.stdout + '\n')
    result.check_returncode()
    return result.stdout


def fetch_json(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def selected(key, name):
    if name in ('README.md', 'LICENSE', 'LICENSE.md', 'NOTICE', 'LICENSE-CODE'):
        return True
    if key == 'moss_base':
        return name.startswith('transformer/')
    if key == 'moss_sft':
        return name == 'pipeline_moss_image.py' or name == 'model_index.json' or name.startswith(('transformer/', 'text_encoder/', 'tokenizer/', 'vae/', 'scheduler/'))
    if key == 'moss_dmd':
        return name.startswith('transformer/') or name in ('dataspace.safetensors', 'lora.safetensors', 'time_embedding.safetensors')
    return name.startswith('q4/')


def safe_name(name):
    p = PurePosixPath(name)
    require(not p.is_absolute() and all(re.fullmatch(r'[A-Za-z0-9_.-]+', v) and v not in ('.', '..') and not v.endswith(('.', ' ')) for v in p.parts), 'unsafe upstream path')
    require(not any(v.split('.')[0].upper() in {'CON','PRN','AUX','NUL',*(f'COM{i}' for i in range(10)),*(f'LPT{i}' for i in range(10))} for v in p.parts), 'reserved upstream filename')
    require(str(p) == name, 'noncanonical upstream path')
    return p


def verify_blob(path, item):
    physical(path)
    require(path.is_file() and not path.is_symlink(), f'missing regular artifact {path}')
    require(path.stat().st_size == item['size'], f'artifact size mismatch {path}')
    if item.get('lfs'):
        require(digest(path) == item['lfs']['sha256'], f'upstream LFS SHA256 mismatch {path}')
    else:
        h = hashlib.sha1(f'blob {path.stat().st_size}\0'.encode())
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                h.update(block)
        require(h.hexdigest() == item['blobId'], f'upstream Git blob identity mismatch {path}')
    return dict(path=str(path.resolve()), sha256=digest(path), bytes=path.stat().st_size,
                upstream_lfs_sha256=item.get('lfs', {}).get('sha256'), upstream_git_blob=item.get('blobId'))


def model_metadata(root, key):
    repo, revision = REPOS[key]
    metadata_path = root / 'metadata' / (key + '.json')
    if metadata_path.exists():
        metadata = read(metadata_path)
    else:
        metadata = fetch_json(f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true')
        require(metadata['sha'] == revision, 'Hub revision mismatch')
        write(metadata_path, metadata)
    require(metadata['sha'] == revision, 'stored Hub revision mismatch')
    return metadata, metadata_path


def stage(root, key):
    repo, revision = REPOS[key]
    metadata, metadata_path = model_metadata(root, key)
    snapshot = root / 'hub' / ('models--' + repo.replace('/', '--')) / 'snapshots' / revision
    files = []
    for item in metadata['siblings']:
        name = item['rfilename']
        if not selected(key, name):
            continue
        relative = safe_name(name)
        destination = snapshot / relative
        physical(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            temporary = destination.with_name(destination.name + '.partial')
            require(not temporary.is_symlink(), 'unsafe partial artifact')
            with urllib.request.urlopen(f'https://huggingface.co/{repo}/resolve/{revision}/{name}', timeout=120) as response, temporary.open('wb') as stream:
                while True:
                    check_deadline()
                    block = response.read(8 * 1024 * 1024)
                    if not block:
                        break
                    stream.write(block)
            verify_blob(temporary, item)
            temporary.replace(destination)
        files.append(verify_blob(destination, item))
    require(files and any(f['path'].endswith('.safetensors') for f in files), f'empty weight selection {key}')
    refs = snapshot.parent.parent / 'refs' / 'main'
    refs.parent.mkdir(exist_ok=True)
    if refs.exists():
        require(refs.read_text() == revision, 'cache ref drift')
    else:
        refs.write_text(revision)
    if key == 'moss_sft':
        require(digest(snapshot / 'pipeline_moss_image.py') == PIPELINE, 'unreviewed pipeline source')
    return dict(repo=repo, revision=revision, snapshot=str(snapshot), files=files,
                metadata=str(metadata_path), metadata_sha256=digest(metadata_path))


def dependency_resolution():
    resolution = read(Path(__file__).with_name('dependency-resolution.json'))
    require(resolution['primary'] == list(PACKAGES) and len(resolution['packages']) == 43, 'reviewed dependency closure drift')
    return resolution


def package_name(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def wheel_identity(wheel, expected):
    # Vendored distributions (notably setuptools/_vendor) have their own
    # METADATA. Only the wheel's single root distribution is installation authority.
    metadata_path = '-'.join(wheel.name.split('-')[:2]) + '.dist-info/METADATA'
    with zipfile.ZipFile(wheel) as archive:
        roots = [n for n in archive.namelist() if re.fullmatch(r'[^/]+\.dist-info/METADATA', n)]
        require(roots == [metadata_path], f'ambiguous or mismatched root wheel metadata: {wheel.name}')
        metadata = BytesParser().parsebytes(archive.read(metadata_path), headersonly=True)
    names, versions = metadata.get_all('Name', []), metadata.get_all('Version', [])
    require(len(names) == len(versions) == 1, f'invalid wheel identity fields: {wheel.name}')
    name, version = names[0], versions[0]
    require(package_name(name) == package_name(expected['name']) and version == expected['version'],
            f'wheel identity differs from reviewed dependency resolution: {wheel.name}')
    return name, version


def lock_wheels(root):
    resolution = dependency_resolution()
    wheels = root / 'wheels'
    lock = root / 'requirements.lock'
    expected = {item['sha256'] for item in resolution['packages']}
    if lock.exists():
        manifest = read(root / 'wheels.json')
        require(manifest['primary'] == list(PACKAGES) and {item['sha256'] for item in manifest['files']} == expected, 'retained wheel closure is from another dependency resolution')
        require(manifest['requirements_sha256'] == digest(lock), 'wheel lock drift')
        names = {unquote(urlparse(item['url']).path.rsplit('/', 1)[-1]): item['sha256'] for item in resolution['packages']}
        require(len(manifest['files']) == len(names), 'retained wheel count drift')
        require({item['name']:item['sha256'] for item in manifest['files']} == names, 'retained wheel filenames differ from reviewed URLs')
        for item in manifest['files']:
            require(re.fullmatch(r'[A-Za-z0-9_.+!-]+\.whl',item['name']) is not None, 'wheel basename required')
            physical(wheels / item['name'])
            require(digest(wheels / item['name']) == item['sha256'], 'wheel content drift')
        return
    wheels.mkdir(exist_ok=True)
    # Metadata-only Windows resolution is checked in; fetch exactly that43-wheel closure.
    links = [item['url'] + '#sha256=' + item['sha256'] for item in resolution['packages']]
    run([sys.executable, '-m', 'pip', 'download', '--no-deps', '--only-binary=:all:', '--dest', wheels, *links], timeout=7200)
    expected = {item['sha256'] for item in resolution['packages']}
    require({digest(path) for path in wheels.glob('*.whl')} == expected, 'downloaded wheel closure differs from reviewed Windows resolution')
    requirements = []
    manifest = []
    seen = set()
    for wheel in sorted(wheels.glob('*.whl')):
        sha = digest(wheel)
        expected_package = next(item for item in resolution['packages'] if item['sha256'] == sha)
        name, version = wheel_identity(wheel, expected_package)
        require(package_name(name) not in seen, 'duplicate package wheels; use a fresh campaign')
        seen.add(package_name(name))
        requirements.append(f'{name}=={version} --hash=sha256:{sha}')
        manifest.append(dict(name=wheel.name, sha256=sha))
    require(manifest, 'no resolved wheels')
    lock.write_text('\n'.join(requirements) + '\n', encoding='utf-8')
    write(root / 'wheels.json', dict(primary=list(PACKAGES), requirements_sha256=digest(lock), files=manifest))


def python(root):
    return root / 'venv' / 'Scripts' / 'python.exe'


def seed_file(source,destination,sha256,*,link=True):
    """Seed only immutable verified OWN inputs; never link new mutable refs or venvs."""
    source,destination=Path(source),Path(destination)
    physical(source);physical(destination)
    require(source.is_file() and digest(source)==sha256,'reuse source bytes changed')
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        require(link or not os.path.samefile(source,destination),'mutable source binary must be copied independently')
        require(digest(destination)==sha256,'retained reuse destination bytes changed')
        method='retained'
    else:
        temporary=destination.with_name(destination.name+'.seed-partial')
        require(not temporary.exists() and not temporary.is_symlink(),'partial reuse seed requires explicit reconciliation')
        try:
            if not link:raise OSError('mutable source requires independent copy')
            os.link(source,temporary);method='hardlink'
        except OSError:
            require(shutil.disk_usage(destination.parent).free >= source.stat().st_size+48*1024**3, 'insufficient space for verified seed copy plus48GiB headroom')
            shutil.copyfile(source,temporary);method='copy'
        require(digest(temporary)==sha256,'reuse seed byte mismatch')
        temporary.replace(destination)
    return dict(source=str(source.resolve()),destination=str(destination.resolve()),sha256=sha256,bytes=destination.stat().st_size,method=method)


def verify_native_source(native_source):
    require(run(['git','rev-parse','HEAD'],cwd=native_source).strip()==SOURCE,'native source mismatch')
    require(not run(['git','status','--porcelain'],cwd=native_source).strip(),'native source checkout has local changes')
    revisions=re.findall(r'git = "https://github.com/SceneWorks/inference", rev = "([0-9a-f]{40})"',(native_source/'Cargo.toml').read_text())
    require(revisions and set(revisions)=={INFERENCE},'native inference pin mismatch')
    locked=re.findall(r'source = "git\+https://github.com/SceneWorks/inference\?rev=([0-9a-f]{40})#([0-9a-f]{40})"',(native_source/'Cargo.lock').read_text())
    require(locked and set(locked)=={(INFERENCE,INFERENCE)},'native locked inference mismatch')


def native_env(root,offline=False):
    return dict(os.environ,CARGO_HOME=str(root/'cargo-home'),CARGO_TARGET_DIR=str(root/'target'),
        CARGO_NET_OFFLINE='true' if offline else 'false',CUDA_COMPUTE_CAP='120',RUSTC_WRAPPER='',
        CARGO_BUILD_JOBS='4',CMAKE_BUILD_PARALLEL_LEVEL='4',MAX_JOBS='4')


def native_identity(binary):
    """Read-only compiler/import evidence; not proof of GPU/model compatibility."""
    dependencies=run(['dumpbin','/DEPENDENTS',binary],timeout=60)
    imports=sorted(set(re.findall(r'^\s*([A-Za-z0-9_.-]+\.dll)\s*$',dependencies,re.M|re.I)),key=str.casefold)
    require(imports,'could not establish native PE dependency names')
    adjacent={p.name.casefold():p for p in binary.parent.iterdir() if p.is_file() and p.suffix.casefold()=='.dll'}
    required_adjacent=[str(adjacent[name.casefold()]) for name in imports if name.casefold() in adjacent]
    resolved=[]
    for name in imports:
        system=Path(os.environ.get('SystemRoot',r'C:\Windows'))/'System32'/name
        candidate=system if system.is_file() else Path(found) if (found:=shutil.which(name)) else None
        item=dict(name=name,candidate_path=str(candidate) if candidate else None,sha256=None,file_version=None,loaded_proof=False)
        if candidate:
            physical(candidate)
            item['sha256']=digest(candidate)
            item['file_version']=run(['powershell','-NoProfile','-Command','(Get-Item -LiteralPath $env:MOSS_DLL_PATH).VersionInfo.FileVersion'],env=dict(os.environ,MOSS_DLL_PATH=str(candidate)),timeout=30).strip() or None
        resolved.append(item)
    return dict(rustc=run(['rustc','-Vv'],timeout=30).strip(),cargo=run(['cargo','-V'],timeout=30).strip(),
        nvcc=run(['nvcc','--version'],timeout=30).strip(),cuda_path=os.environ.get('CUDA_PATH'),
        dependency_names=imports,dependency_report_sha256=hashlib.sha256(dependencies.encode()).hexdigest(),
        required_adjacent_dlls=required_adjacent,adjacent_dll_files=sorted(str(p) for p in adjacent.values()),dependency_candidates=resolved,
        limitations='PE import candidates are not actual loaded DLL proof. Dynamic model dependencies and CUDA/driver compatibility require execution. --help is not a CPU probe: this source starts the API for unrecognized arguments.')


def reuse_inputs(root,value):
    reuse=value['reuse'];source=root.parent/reuse['campaign']
    physical(source)
    receipt=source/('job-'+reuse['job']+'-prepare.json')
    physical(receipt);physical(source/'prepared.json')
    require(receipt.is_file() and (source/'prepared.json').is_file(),'completed source receipts missing')
    require(digest(receipt)==reuse['job_receipt_sha256'],'source job receipt changed')
    job=read(receipt)
    require(job['mode']=='prepare' and job['status']=='completed' and job['job']==reuse['job'] and Path(job['root']).resolve()==source.resolve(),'source preparation is not complete in the owned campaign')
    require(job['sceneworks_source']==SOURCE and job['inference']==INFERENCE and job['accelerator_execution_requested'] is False,'source job identity or scope changed')
    require(digest(source/'prepared.json')==reuse['prepared_sha256'],'source prepared receipt changed')
    prep=prepared(source)
    # Source preparation has ended before any old build/cache path is used.
    require(Path(prep['binary']).resolve()==(source/'target'/'release'/'sceneworks-rust-api.exe').resolve(),'source binary outside owned build')
    require((source/'requirements.lock').is_file() and (source/'wheels.json').is_file(),'source locked wheels missing')
    physical(source/'requirements.lock');physical(source/'wheels.json')
    lock_wheels(source)  # hash-only retained branch; no re-resolution/download
    expected=dependency_resolution();require({f['sha256'] for f in read(source/'wheels.json')['files']}=={p['sha256'] for p in expected['packages']},'source dependency closure changed')
    seeds=[]
    for file in read(source/'wheels.json')['files']:
        seeds.append(seed_file(source/'wheels'/file['name'],root/'wheels'/file['name'],file['sha256']))
    for name in ('requirements.lock','wheels.json'):
        seeds.append(seed_file(source/name,root/name,digest(source/name)))
    for key,item in prep['staged'].items():
        metadata=source/'metadata'/(key+'.json')
        require(Path(item['metadata']).resolve()==metadata.resolve(),'source metadata outside owned campaign')
        seeds.append(seed_file(metadata,root/'metadata'/(key+'.json'),item['metadata_sha256']))
        for file in item['files']:
            relative=Path(file['path']).relative_to(source)
            seeds.append(seed_file(Path(file['path']),root/relative,file['sha256']))
    record=dict(source_campaign=reuse['campaign'],source_root=str(source.resolve()),source_prepared_sha256=reuse['prepared_sha256'],source_job_receipt_sha256=reuse['job_receipt_sha256'],source_job=reuse['job'],seeds=seeds,downloaded_seed_bytes=0)
    return source,prep,record


def reuse_native(root,source,prep,record):
    native_source=source/'native-source'
    for path in (native_source,source/'cargo-home',source/'target',Path(prep['binary'])):physical(path)
    verify_native_source(native_source)
    env=native_env(source,offline=True)
    command=['cargo','build','--release','--locked','-p','sceneworks-rust-api','--features','backend-candle','--bin','sceneworks-rust-api']
    run(command,cwd=native_source,env=env,timeout=10800)
    binary=Path(prep['binary'])
    require(digest(binary)==prep['binary_sha256'],'locked old-cache build changed source binary; refuse reuse')
    identity=native_identity(binary)
    require(not identity['required_adjacent_dlls'] and not identity['adjacent_dll_files'],'native reuse has unrecorded adjacent DLLs; inspect dependencies or explicitly prepare a fresh locked build')
    target=root/'target'/'release'/'sceneworks-rust-api.exe'
    record['native']=dict(command=command,cwd=str(native_source),cargo_home=env['CARGO_HOME'],cargo_target_dir=env['CARGO_TARGET_DIR'],offline=True,source=SOURCE,inference=INFERENCE,binary=seed_file(binary,target,prep['binary_sha256'],link=False),identity=identity,
        prior_toolchain=prep.get('native_build',{'status':'not captured by original source prepare'}))
    return target,identity


def prepare(root, value):
    require(sys.version_info[:2] == (3, 12), 'preparation requires Python3.12')
    require(not (root / 'prepared.json').exists(), 'already prepared; use freeze or explicitly choose a new campaign')
    root.mkdir(parents=True, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES']=''
    reuse=reuse_inputs(root,value) if value.get('reuse') else None
    metadata = {key: model_metadata(root, key)[0] for key in REPOS}
    selected_bytes = sum(item['size'] for key, data in metadata.items() for item in data['siblings'] if selected(key, item['rfilename']))
    # Count the exact six chosen snapshots; additional space covers wheels, source/build and outputs.
    required_bytes=(0 if reuse else selected_bytes)+48*1024**3
    require(shutil.disk_usage(root).free>=required_bytes,f'need {required_bytes} free bytes after reuse seeding: selected artifacts plus48GiB build/wheel/output headroom')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    lock_wheels(root)
    if not python(root).exists():
        run([sys.executable, '-m', 'venv', root / 'venv'])
    run([python(root), '-m', 'pip', 'install', '--no-index', '--find-links', root / 'wheels',
         '--require-hashes', '-r', root / 'requirements.lock'], timeout=3600)
    packages = run([python(root), '-m', 'pip', 'freeze', '--all'])
    (root / 'installed.txt').write_text(packages, encoding='utf-8')
    # Import/load compatibility is real host evidence; this does not allocate CUDA or load weights.
    runtime = json.loads(run([python(root), '-c', 'import json,platform,sys,importlib.metadata,torch,diffusers,transformers,accelerate,safetensors,PIL,psutil; from diffusers import AutoencoderKLFlux2; from transformers import Qwen3_5Model; print(json.dumps(dict(python=sys.version,os=platform.platform(),cuda=torch.version.cuda,packages={d.metadata["Name"]:d.version for d in importlib.metadata.distributions()})))']))
    staged = {key: stage(root, key) for key in REPOS}
    harness = root / 'harness'
    harness.mkdir(exist_ok=True)
    for source in Path(__file__).parent.iterdir():
        if source.suffix == '.py' or source.name in ('corpus.json', 'dependency-resolution.json'):
            destination = harness / source.name
            if destination.exists():
                require(digest(destination) == digest(source), 'harness changed during preparation; fresh campaign required')
            else:
                shutil.copyfile(source, destination)
    if reuse:
        binary,native_build=reuse_native(root,*reuse)
        write(root/'reuse.json',reuse[2])
    else:
        native_source = root / 'native-source'
        if not native_source.exists():
            run(['git', 'clone', '--filter=blob:none', '--no-checkout', 'https://github.com/SceneWorks/SceneWorks.git', native_source], timeout=600)
        run(['git', 'fetch', 'origin', SOURCE], cwd=native_source, timeout=600)
        run(['git', 'checkout', '--detach', SOURCE], cwd=native_source)
        require(run(['git', 'rev-parse', 'HEAD'], cwd=native_source).strip() == SOURCE, 'native source mismatch')
        require(not run(['git', 'status', '--porcelain'], cwd=native_source).strip(), 'native source checkout has local changes')
        revisions = re.findall(r'git = "https://github.com/SceneWorks/inference", rev = "([0-9a-f]{40})"', (native_source / 'Cargo.toml').read_text())
        require(revisions and set(revisions) == {INFERENCE}, 'native inference pin mismatch')
        lock_sources = re.findall(r'source = "git\+https://github.com/SceneWorks/inference\?rev=([0-9a-f]{40})#([0-9a-f]{40})"', (native_source / 'Cargo.lock').read_text())
        require(lock_sources and set(lock_sources) == {(INFERENCE, INFERENCE)}, 'native locked inference mismatch')
        env = dict(os.environ, CARGO_HOME=str(root / 'cargo-home'), CARGO_TARGET_DIR=str(root / 'target'),
                   CARGO_NET_OFFLINE='false', CUDA_COMPUTE_CAP='120', RUSTC_WRAPPER='',
                   CARGO_BUILD_JOBS='4', CMAKE_BUILD_PARALLEL_LEVEL='4', MAX_JOBS='4')
        run(['cargo', 'fetch', '--locked'], cwd=native_source, env=env, timeout=1800)
        run(['cargo', 'build', '--release', '--locked', '-p', 'sceneworks-rust-api', '--features', 'backend-candle', '--bin', 'sceneworks-rust-api'], cwd=native_source, env=env, timeout=10800)
        binary = root / 'target' / 'release' / 'sceneworks-rust-api.exe'
        require(binary.is_file(), 'native build produced no binary')
        native_build=native_identity(binary)
    write(root / 'prepared.json', dict(staged=staged, runtime=runtime, installed=packages,
         requirements_sha256=digest(root / 'requirements.lock'), binary=str(binary), binary_sha256=digest(binary),
         sceneworks_source=SOURCE, inference=INFERENCE,native_build=native_build,
         reuse=(dict(path=str(root/'reuse.json'),sha256=digest(root/'reuse.json'),source_prepared_sha256=reuse[2]['source_prepared_sha256']) if reuse else None),
         harness={p.name: digest(p) for p in harness.iterdir() if p.is_file()}, accelerator_allocation=False))
    print(f'Prepared {root}; next: review artifact/dependency receipts then freeze with ownership/device UUID.', flush=True)


def prepared(root):
    value = read(root / 'prepared.json')
    require(value['sceneworks_source'] == SOURCE and value['inference'] == INFERENCE, 'prepared source drift')
    require(digest(value['binary']) == value['binary_sha256'], 'native binary drift')
    require(digest(root / 'requirements.lock') == value['requirements_sha256'], 'dependency lock drift')
    if value.get('reuse'):
        receipt=root/'reuse.json';physical(receipt)
        require(Path(value['reuse']['path']).resolve()==receipt.resolve() and digest(receipt)==value['reuse']['sha256'],'reuse provenance drift')
    verify_staged(root, value['staged'])
    for name, sha in value['harness'].items():
        require(digest(root / 'harness' / name) == sha, 'harness drift')
    return value


def freeze(root, value):
    prep = prepared(root)
    spec = importlib.util.spec_from_file_location('moss_freeze_evaluate', root / 'harness' / 'evaluate.py')
    evaluation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation)
    manifest = evaluation.template()
    matches = [g for g in gpu_rows() if g['uuid'] == value['gpu_uuid']]
    require(len(matches) == 1, 'device UUID not present')
    gpu = matches[0]
    require(gpu['index'] == '1' and gpu['uuid'] == RUNNER_GPU_UUID, 'physical GPU1 identity changed; review ownership before proceeding')
    import psutil
    manifest['runtime'] = dict(prep['runtime'], driver=gpu['driver'])
    manifest['hardware'] = dict(host=socket.gethostname(), gpu_uuid=gpu['uuid'], gpu_name=gpu['name'],
        gpu_memory_bytes=int(float(gpu['memory_mib']) * 1024**2), host_memory_bytes=psutil.virtual_memory().total)
    manifest['ownership'] = value['ownership']
    manifest['preparation'] = dict(path=str((root / 'prepared.json').resolve()), sha256=digest(root / 'prepared.json'))
    shared = prep['staged']['moss_sft']['snapshot']
    for index, config in enumerate(manifest['configs']):
        config['snapshot'] = prep['staged'][config['id']]['snapshot']
        config['runtime_command'] = [str(python(root)), str(root / 'harness' / 'run.py')]
        if config['id'].startswith('moss_'):
            config['settings'] = dict(shared_snapshot=shared, pipeline_code=str(Path(shared) / 'pipeline_moss_image.py'))
            if config['id'] == 'moss_dmd':
                config['settings'].update(adapter_validation=True, lora_path=str(Path(config['snapshot']) / 'lora.safetensors'),
                    replacement_path=str(Path(config['snapshot']) / 'time_embedding.safetensors'))
        else:
            config['settings'] = dict(binary=prep['binary'], binary_sha256=prep['binary_sha256'],
                hub_cache=str(root / 'hub'), state_root=str(root / 'native-state'), port=17931 + index - 3,
                worker_id='sc26322-' + config['id'], device_id=gpu['uuid'], job_timeout_seconds=manifest['ownership']['timeout_seconds'],
                runtime_versions=dict(sceneworks_source=SOURCE, inference=INFERENCE,native_build=prep.get('native_build',{'status':'not captured'})))
        config['files'] = evaluation.inventory(config)
    write(root / 'manifest.json', manifest)
    evaluation.freeze(root / 'manifest.json', root / 'protocol.lock.json')
    print(f'Frozen protocol {evaluation.read(root / "protocol.lock.json")["sha256"]}; calibration requires coordinator review and matching lease.', flush=True)


@contextlib.contextmanager
def device_lock(base, uuid, lease):
    require(re.fullmatch(r'GPU-[0-9a-fA-F-]{16,64}', uuid), 'physical UUID required')
    locks = base / 'device-locks'
    physical(locks)
    locks.mkdir(parents=True, exist_ok=True)
    path = locks / (uuid + '.lock')
    # Exclusive creation; stale locks are NEVER stolen automatically.
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, json.dumps(lease).encode())
        yield path
    finally:
        os.close(descriptor)
        path.unlink()


def remember_owned(process, known):
    import psutil
    try:
        for child in psutil.Process(process.pid).children(recursive=True):
            known[child.pid] = child.create_time()
    except psutil.NoSuchProcess:
        pass


def kill_owned(process, known=None):
    """Only stop identities observed in this explicitly spawned process tree."""
    import psutil
    known = {} if known is None else known
    remember_owned(process, known)
    children = []
    for pid, created in known.items():
        with contextlib.suppress(psutil.NoSuchProcess):
            child = psutil.Process(pid)
            if child.create_time() == created:
                children.append(child)
    for child in reversed(children):
        with contextlib.suppress(psutil.NoSuchProcess):
            child.terminate()
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    for child in children:
        with contextlib.suppress(psutil.NoSuchProcess):
            if child.is_running():
                child.kill()


def accepted_scored_authorization(root, value, manifest=None):
    require(value['calibration_receipts_sha256'] == calibration_bundle(root / 'protocol.lock.json', root / 'run'), 'coordinator accepted calibration digest differs from actual terminal attempts')
    require(value['calibration_estimate_sha256'] == digest(root / 'calibration-estimate.json'), 'coordinator accepted estimate digest differs from actual measurement')
    accepted = {key: value[key] for key in ('protocol_sha256', 'calibration_receipts_sha256', 'calibration_estimate_sha256', 'authorization')}
    if 'unavailable_configs' in value:accepted['unavailable_configs']=value['unavailable_configs']
    if manifest is None:
        from evaluate import load_lock
        manifest=load_lock(root/'protocol.lock.json')
    decisions=availability_decisions(manifest,value['protocol_sha256'],root/'run',accepted)
    path = root / 'scored-authorization.json'
    physical(path)
    if path.exists():
        require(read(path) == accepted, 'scored authorization changed; no automatic replacement')
    else:
        require(not list((root/'run'/'receipts').glob('scored.*.json')) and not list((root/'run'/'outputs').glob('scored.*.png')),'availability/scored authorization must precede every scored artifact')
        write(path, accepted)
    if decisions:
        mirror=root/'run'/'availability-authorization.json'
        physical(mirror)
        if mirror.exists():require(digest(mirror)==digest(path),'immutable availability authorization changed')
        else:shutil.copyfile(path,mirror)
    return dict(path=str(path.resolve()), sha256=digest(path))


def calibration_estimate(manifest, results, parity=None):
    require(len(results)==6 and all(r['status']=='ok' or (r['status'] in ('failed','timeout') and r.get('failure_kind')=='runtime') for r in results.values()),'six terminal guarded calibration attempts required')
    conditions = {}
    for key in REPOS:
        record = next(r for r in results.values() if r['cell']['config_id'] == key)
        width, height = record['cell']['width'], record['cell']['height']
        require((width, height) == (1024, 1024), 'calibration must use first actual1024x1024 scored fixture')
        compatible = [case for case in manifest['corpus']['cases'] if (case['width'], case['height']) == (width, height)]
        if record['status']!='ok':
            conditions[key]=dict(status=record['status'],disposition='pending_coordinator_review',calibration_canvas=[width,height],calibration_outputs=int(bool(record.get('output'))),candidate_generation_seconds=None,load_seconds=None,same_canvas_scored_outputs=len(compatible)*len(manifest['corpus']['seeds']),same_canvas_generation_estimate_seconds=None,error=record['error'],log=record['log'])
            continue
        conditions[key] = dict(calibration_canvas=[width, height], calibration_outputs=1,
            candidate_generation_seconds=record['metrics']['generation_seconds'],
            load_seconds=record['metrics'].get('load_seconds'),
            same_canvas_scored_outputs=len(compatible) * len(manifest['corpus']['seeds']),
            same_canvas_generation_estimate_seconds=record['metrics']['generation_seconds'] * len(compatible) * len(manifest['corpus']['seeds']),
            other_canvas_shapes=[list(shape) for shape in sorted({(case['width'], case['height']) for case in manifest['corpus']['cases'] if (case['width'], case['height']) != (width, height)})])
    counts = {}
    for case in manifest['corpus']['cases']:
        shape = f"{case['width']}x{case['height']}"
        counts[shape] = counts.get(shape, 0) + len(manifest['corpus']['seeds'])
    return dict(scored_outputs=288, calibration_outputs=sum(r['status']=='ok' or bool(r.get('output')) for r in results.values()),calibration_output_cap=6,calibration_attempts=6,successful_calibration_outputs=sum(r['status']=='ok' for r in results.values()), conditions=conditions,
        dmd_parity_overhead=parity,
        scored_outputs_by_canvas_per_config=counts,
        scored_outputs_by_canvas_all_configs={shape: count * 6 for shape, count in counts.items()},
        whole_campaign_estimate_seconds=None,
        limitations='One existing1024x1024 fixture per config; prompt variance and other canvas shapes unmeasured. Separate DMD parity oracle/candidate forwards are not candidate generation latency. Fresh process/worker from staged cache, not OS-cold disk; hash checks warm filesystem cache. Reviewed desktop baseline may affect timing/device-wide memory.',
        next_action='Coordinator reviews all measurements and uncertainty, then supplies accepted receipt/estimate digests and explicit scored authorization')


def execute_phase(root, value, phase):
    prep = prepared(root)
    protocol = read(root / 'protocol.lock.json')
    require(protocol['sha256'] == value['protocol_sha256'], 'protocol differs from coordinator-reviewed hash')
    sys.path.insert(0, str(root / 'harness'))
    from evaluate import load_lock, cells, parity_cells, receipts, seal
    manifest = load_lock(root / 'protocol.lock.json', verify_inputs=True)
    lease = value['lease']
    require(lease['gpu_uuid'] == manifest['hardware']['gpu_uuid'] and lease['host'].casefold() == manifest['hardware']['host'].casefold() == socket.gethostname().casefold(), 'lease/device/host mismatch')
    require(lease['id'] == manifest['ownership']['lease'] and lease['owner'] == manifest['ownership']['responsible_chat'], 'lease owner mismatch')
    require(len(list(cells(manifest, phase))) == (6 if phase == 'calibration' else 288), 'fixed execution budget changed')
    campaign = root / 'run'
    campaign.mkdir(exist_ok=True)
    expected_images = {cell['id'] + '.png' for mode in ('calibration', 'scored') for cell in cells(manifest, mode)}
    require(not (campaign / 'outputs').exists() or all(path.name in expected_images for path in (campaign / 'outputs').iterdir()), 'unexpected output outside fixed294-image plan')
    authority = accepted_scored_authorization(root, value,manifest) if phase == 'scored' else None
    lock_record = dict(lease_id=lease['id'], host=lease['host'], gpu_uuid=lease['gpu_uuid'], phase=phase,
        protocol_sha256=protocol['sha256'], deadline_epoch=timestamp(value['deadline_utc']).timestamp(),
        lease_expires_epoch=timestamp(lease['expires_utc']).timestamp(), owner_pid=os.getpid(), responsible_chat=lease['owner'],
        prepared_root=str(root.resolve()), prepared_sha=digest(root / 'prepared.json'),
        external_owners_reconciled=lease['external_owners_reconciled'],
        desktop_baseline=lease['desktop_baseline'], baseline_max_memory_mib=lease['baseline_max_memory_mib'],
        baseline_max_utilization_percent=lease['baseline_max_utilization_percent'],
        job=os.environ.get('GITHUB_RUN_ID', 'local') + '-' + os.environ.get('GITHUB_RUN_ATTEMPT', '1'),
        scored_authorization=authority)
    with device_lock(root.parent, lease['gpu_uuid'], lock_record) as held_lock:
        env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=lease['gpu_uuid'], HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', MOSS_DEVICE_LOCK=str(held_lock.resolve()), MOSS_LEASE_ID=lease['id'], MOSS_PROTOCOL_SHA256=protocol['sha256'])
        guard = ExecutionGuard(manifest, protocol['sha256'], phase=phase, allow_owner=True, environ=env)
        observation = guard.check(idle=True)
        job = lock_record['job']
        write(campaign / (phase + '-background-' + job + '.json'), observation)
        for config in manifest['configs']:
            guard.check(idle=True)
            _, completed_before = receipts(root / 'protocol.lock.json', campaign, phase, complete=False)
            reject_run_invalidated(campaign,protocol['sha256'])
            if phase=='calibration':calibration_results(manifest,protocol['sha256'],campaign,complete=False)
            else:require(all(terminal_scored(record,guard.unavailable_configs) for record in completed_before.values()),'unapproved scored failure cannot resume')
            pending = [cell['id'] for cell in cells(manifest, phase) if cell['config_id'] == config['id'] and cell['id'] not in completed_before]
            if not pending:
                continue
            log = campaign / (phase + '-transport-' + config['id'] + '-' + job + '.log')
            with log.open('x', encoding='utf-8') as stream:
                process = subprocess.Popen([str(python(root)), str(root / 'harness' / 'run.py'), str(root / 'protocol.lock.json'), str(campaign), '--config', config['id'], '--phase', phase], env=env, stdout=stream, stderr=subprocess.STDOUT)
                owned = {}
                try:
                    while process.poll() is None:
                        remember_owned(process, owned)
                        guard.check()
                        time.sleep(1)
                    require(process.returncode == 0, f'{phase} process failed; inspect {log}')
                except BaseException as exc:
                    # The outer watchdog may stop the child before its own guard can persist this.
                    # Previously completed validated cells remain immutable; this pending attempt is invalid.
                    parity_ids = [cell['id'] for cell in parity_cells(manifest, phase)] if config['id'] == 'moss_dmd' else []
                    write(campaign / 'invalidations' / (phase + '-' + config['id'] + '-' + job + '-transport.json'),
                          seal(dict(lock_sha256=protocol['sha256'], cell_ids=pending, parity_cell_ids=parity_ids,
                                    reason=f'outer owned-runtime watchdog stopped attempt: {exc}', attempt=job)))
                    raise
                finally:
                    kill_owned(process, owned)
            _, results = receipts(root / 'protocol.lock.json', campaign, phase, complete=False)
            if phase=='calibration':calibration_results(manifest,protocol['sha256'],campaign,complete=False)
            else:require(all(terminal_scored(r,guard.unavailable_configs) for r in results.values()),'execution cell failed; no automatic retry or further allocation')
            guard.check(idle=True)
        _, results = receipts(root / 'protocol.lock.json', campaign, phase)
        if phase=='calibration':calibration_results(manifest,protocol['sha256'],campaign)
        else:require(all(terminal_scored(record,guard.unavailable_configs) for record in results.values()),'phase contains unapproved failed cells')
        if phase=='scored':require(not any(r['status'] in ('failed','timeout') for r in results.values()),'scored attempts complete with runtime failures/censored cells; inspect full matrix, no retry or retrospective exclusion')
        if phase == 'calibration':
            parity_records = []
            for path in sorted((campaign / 'parity').glob('*.json')):
                record = read(path)
                evidence = record['evidence']
                parity_records.append(dict(path=str(path.resolve()), sha256=digest(path),
                    numerical_seconds=evidence['elapsed_seconds'], initialization_seconds=evidence.get('initialization_seconds'),
                    total_process_seconds=evidence.get('total_process_seconds'), forward_count=evidence['forward_count'], generated_images=evidence['generated_images']))
            estimate = calibration_estimate(manifest, results, parity_records)
            path = root / 'calibration-estimate.json'
            if path.exists():
                require(read(path) == estimate, 'calibration estimate changed')
            else:
                write(path, estimate)
            print('Calibration receipt bundle: ' + calibration_bundle(root / 'protocol.lock.json', campaign), flush=True)
            print('Calibration estimate SHA256: ' + digest(path), flush=True)


def main():
    global DEADLINE, COMMAND_LOG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'freeze', 'calibration', 'scored'))
    args = parser.parse_args()
    value, root = control(args.mode, os.environ.get('MOSS_CONTROL_JSON', '{}'))
    require(os.name == 'nt', 'Windows real-weights host only')
    if args.mode != 'prepare' and Path(sys.executable).resolve() != python(root).resolve():
        require(python(root).is_file(), 'prepare the locked environment first')
        require(digest(root / 'harness' / 'bootstrap.py') == digest(__file__), 'transport source changed since prepare; review and prepare a new campaign')
        # Validated campaign selects a fixed path; no shell or input-supplied executable.
        return subprocess.run([str(python(root)), str(root / 'harness' / 'bootstrap.py'), args.mode], check=True).returncode
    DEADLINE = timestamp(value['deadline_utc'])
    root.mkdir(parents=True, exist_ok=True)
    if os.environ.get('GITHUB_ENV'):
        with open(os.environ['GITHUB_ENV'], 'a', encoding='utf-8') as stream:
            stream.write('MOSS_RECEIPTS=' + str(root.resolve()) + '\n')
    job = os.environ.get('GITHUB_RUN_ID', str(os.getpid())) + '-' + os.environ.get('GITHUB_RUN_ATTEMPT', '1')
    receipt = root / ('job-' + job + '-' + args.mode + '.json')
    COMMAND_LOG = root / ('job-' + job + '-' + args.mode + '.log')
    require(not receipt.exists(), 'job receipt exists')
    result = dict(mode=args.mode, root=str(root.resolve()), job=job, deadline_utc=value['deadline_utc'],
        log=str(COMMAND_LOG.resolve()), resume_command=f'python {Path(__file__).resolve()} {args.mode} (reviewed MOSS_CONTROL_JSON with current lease/deadline; completed immutable cells resume; failed cells never retry)',
        next_action='Inspect preparation/protocol/calibration receipts before next explicit command', started_utc=utc().isoformat(), accelerator_execution_requested=args.mode in ('calibration', 'scored'), workflow_revision=os.environ.get('GITHUB_SHA'), transport_sha256=digest(__file__), sceneworks_source=SOURCE, inference=INFERENCE)
    try:
        if args.mode == 'prepare':
            prepare(root, value)
        else:
            # Freeze/calibration run in the exact locked environment prepared earlier.
            require(Path(sys.executable).resolve() == python(root).resolve(), 'use prepared venv Python for freeze/calibration')
            (freeze if args.mode == 'freeze' else lambda r, v: execute_phase(r, v, args.mode))(root, value)
        result['status'] = 'completed'
    except BaseException as exc:
        result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        result['finished_utc'] = utc().isoformat()
        write(receipt, result)
        print(f'Absolute job receipt: {receipt.resolve()}', flush=True)


if __name__ == '__main__':
    main()
