#!/usr/bin/env python3
"""SC-26322 Windows preparation/freeze/calibration transport. Never runs scored cells.

All artifacts persist outside the checkout. Control arrives as JSON in
MOSS_CONTROL_JSON (not interpolated into a shell). See bootstrap.md.
"""
import argparse
import contextlib
import csv
import datetime as dt
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
import zipfile

SOURCE = 'ba2954f47bfd75a3186002e3cef859c4e4e1d37d'
INFERENCE = '91ffb4a819aa2990c652db2d881ba0b9b10126ce'
RUNNER_GPU_UUID = 'GPU-e4b79931-7be6-f216-460a-f5405cfafffe'
PIPELINE = '1dac1aca9feb1bfdfc727a401e4b67d61a505d3190706d1b7d2d784f88678b31'
BASE = Path(r'E:\sceneworks-evaluation\moss')
PACKAGES = ('torch==2.9.1+cu128', 'diffusers==0.38.0', 'transformers==5.3.0',
            'accelerate==1.10.1', 'safetensors==0.6.2', 'Pillow==12.1.1',
            'psutil==7.2.2', 'huggingface-hub==1.5.0')
DEADLINE = None
COMMAND_LOG = None
REPOS = {
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


def physical(path):
    for part in [path, *path.parents]:
        require(not part.is_symlink() and not (part.exists() and getattr(part.stat(), 'st_file_attributes', 0) & 1024), 'reparse point in artifact path')


def control(mode, raw, base=BASE):
    require(len(raw) <= 8192, 'oversized control')
    value = json.loads(raw)
    required = {'campaign', 'deadline_utc'}
    allowed = required | ({'ownership', 'gpu_uuid'} if mode == 'freeze' else
                          {'protocol_sha256', 'lease'} if mode == 'calibration' else set())
    require(set(value) == allowed, f'{mode} control fields must be exactly {sorted(allowed)}')
    require(re.fullmatch(r'sc-26322-[a-z0-9][a-z0-9-]{0,48}', value['campaign']), 'invalid campaign id')
    require(utc() < timestamp(value['deadline_utc']) <= utc() + dt.timedelta(hours=6), 'deadline must be in next six hours')
    root = base / value['campaign']
    # Refuse reparse points (including Windows directory junctions) before writes.
    physical(root)
    if mode == 'freeze':
        require(value['gpu_uuid'] == RUNNER_GPU_UUID, 'runner evaluation is restricted to the verified physical GPU1 UUID')
        expected = {'responsible_chat', 'lease', 'job', 'timeout_seconds', 'resume', 'next_action'}
        own = value['ownership']
        require(set(own) == expected and all(own.values()), 'complete ownership required')
        require(type(own['timeout_seconds']) is int and 0 < own['timeout_seconds'] <= 1800, 'per-cell timeout exceeds calibration bound')
    if mode == 'calibration':
        require(re.fullmatch(r'[0-9a-f]{64}', value['protocol_sha256']), 'reviewed frozen protocol SHA256 required')
        lease = value['lease']
        require(set(lease) == {'id', 'owner', 'host', 'gpu_uuid', 'expires_utc', 'external_owners_reconciled', 'desktop_baseline', 'baseline_max_memory_mib', 'baseline_max_utilization_percent'}, 'complete device lease required')
        require(all(lease[k] for k in ('id', 'owner', 'host', 'gpu_uuid', 'expires_utc')) and lease['external_owners_reconciled'] is True, 'coordinator must reconcile ALL physical host owners')
        require(type(lease['desktop_baseline']) is list, 'reviewed desktop baseline list required (empty only on empty device)')
        for item in lease['desktop_baseline']:
            require(set(item) == {'pid', 'name', 'created_epoch'} and type(item['pid']) is int and item['pid'] > 0
                    and isinstance(item['name'], str) and item['name'] and '[' not in item['name']
                    and type(item['created_epoch']) in (float, int) and item['created_epoch'] > 0, 'complete known process identity required')
        require(len({p['pid'] for p in lease['desktop_baseline']}) == len(lease['desktop_baseline']), 'duplicate desktop process identity')
        require(type(lease['baseline_max_memory_mib']) in (float, int) and 0 <= lease['baseline_max_memory_mib'] <= 4096,
                'reviewed desktop memory ceiling must be between0 and4096MiB')
        require(type(lease['baseline_max_utilization_percent']) in (float, int) and 0 <= lease['baseline_max_utilization_percent'] <= 5,
                'reviewed idle desktop utilization ceiling must be between0 and5percent')
        require(lease['gpu_uuid'] == RUNNER_GPU_UUID, 'runner evaluation is restricted to the verified physical GPU1 UUID')
        require(timestamp(lease['expires_utc']) >= timestamp(value['deadline_utc']), 'lease must cover whole deadline')
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


def lock_wheels(root):
    wheels = root / 'wheels'
    lock = root / 'requirements.lock'
    if lock.exists():
        manifest = read(root / 'wheels.json')
        require(manifest['requirements_sha256'] == digest(lock), 'wheel lock drift')
        for item in manifest['files']:
            require(digest(wheels / item['name']) == item['sha256'], 'wheel content drift')
        return
    wheels.mkdir(exist_ok=True)
    # Resolve only binary wheels for THIS host's Python3.12; no source builds.
    run([sys.executable, '-m', 'pip', 'download', '--only-binary=:all:', '--dest', wheels,
         '--index-url', 'https://pypi.org/simple', '--extra-index-url', 'https://download.pytorch.org/whl/cu128', *PACKAGES], timeout=7200)
    requirements = []
    manifest = []
    seen = set()
    for wheel in sorted(wheels.glob('*.whl')):
        with zipfile.ZipFile(wheel) as archive:
            metadata_files = [n for n in archive.namelist() if n.endswith('.dist-info/METADATA')]
            require(len(metadata_files) == 1, 'ambiguous wheel metadata')
            metadata = archive.read(metadata_files[0]).decode()
        name = re.search(r'^Name: (.+)$', metadata, re.M).group(1)
        version = re.search(r'^Version: (.+)$', metadata, re.M).group(1)
        require(name.lower().replace('_', '-') not in seen, 'duplicate package wheels; use a fresh campaign')
        seen.add(name.lower().replace('_', '-'))
        sha = digest(wheel)
        requirements.append(f'{name}=={version} --hash=sha256:{sha}')
        manifest.append(dict(name=wheel.name, sha256=sha))
    require(manifest, 'no resolved wheels')
    lock.write_text('\n'.join(requirements) + '\n', encoding='utf-8')
    write(root / 'wheels.json', dict(primary=list(PACKAGES), requirements_sha256=digest(lock), files=manifest))


def python(root):
    return root / 'venv' / 'Scripts' / 'python.exe'


def prepare(root, value):
    require(sys.version_info[:2] == (3, 12), 'preparation requires Python3.12')
    require(not (root / 'prepared.json').exists(), 'already prepared; use freeze or explicitly choose a new campaign')
    root.mkdir(parents=True, exist_ok=True)
    metadata = {key: model_metadata(root, key)[0] for key in REPOS}
    selected_bytes = sum(item['size'] for key, data in metadata.items() for item in data['siblings'] if selected(key, item['rfilename']))
    # Count the exact six chosen snapshots; additional space covers wheels, source/build and outputs.
    require(shutil.disk_usage(root).free >= selected_bytes + 48 * 1024**3,
            f'need {selected_bytes + 48 * 1024**3} free bytes: selected artifacts plus48GiB build/wheel/output headroom')
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
        if source.suffix == '.py' or source.name == 'corpus.json':
            destination = harness / source.name
            if destination.exists():
                require(digest(destination) == digest(source), 'harness changed during preparation; fresh campaign required')
            else:
                shutil.copyfile(source, destination)
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
    write(root / 'prepared.json', dict(staged=staged, runtime=runtime, installed=packages,
         requirements_sha256=digest(root / 'requirements.lock'), binary=str(binary), binary_sha256=digest(binary),
         sceneworks_source=SOURCE, inference=INFERENCE,
         harness={p.name: digest(p) for p in harness.iterdir() if p.is_file()}, accelerator_allocation=False))
    print(f'Prepared {root}; next: review artifact/dependency receipts then freeze with ownership/device UUID.', flush=True)


def gpu_rows():
    data = run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,driver_version,memory.free,memory.used,utilization.gpu', '--format=csv,noheader,nounits'])
    return [dict(zip(('index','uuid','name','memory_mib','driver','free_mib','used_mib','utilization_percent'), (v.strip() for v in row))) for row in csv.reader(data.splitlines())]


def compute_pids(uuid):
    data = run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'])
    pids = set()
    for row in csv.reader(data.splitlines()):
        if len(row) == 2 and row[0].strip() == uuid:
            require(row[1].strip().isdigit(), 'unattributable GPU process')
            pids.add(int(row[1].strip()))
    return pids


def desktop_identities(lease):
    import psutil
    for item in lease['desktop_baseline']:
        # Permission denied, missing or reused PID is unresolved ownership, never an allow.
        process = psutil.Process(item['pid'])
        require(process.name().lower() == item['name'].lower() and abs(process.create_time() - item['created_epoch']) < .01,
                f'desktop baseline identity changed for PID{item["pid"]}')
    return {p['pid'] for p in lease['desktop_baseline']}


def idle_device(lease):
    baseline = desktop_identities(lease)
    require(compute_pids(lease['gpu_uuid']) == baseline, 'device owners differ from reviewed desktop baseline')
    matches = [g for g in gpu_rows() if g['uuid'] == lease['gpu_uuid']]
    require(len(matches) == 1, 'leased device absent')
    gpu = matches[0]
    require(gpu['index'] == '1' and gpu['uuid'] == RUNNER_GPU_UUID, 'physical GPU1 identity changed; review ownership before proceeding')
    require(float(gpu['used_mib']) <= lease['baseline_max_memory_mib'] and
            float(gpu['utilization_percent']) <= lease['baseline_max_utilization_percent'],
            'desktop device exceeds reviewed idle memory/utilization ceiling')
    return dict(gpu=gpu, desktop_baseline=lease['desktop_baseline'],
                limitation='Reviewed desktop processes share this physical device; device-wide peak memory includes their presence.')


def verify_staged(root, staged, hash_weights=True):
    for key, item in staged.items():
        require((item['repo'], item['revision']) == REPOS[key], 'staged model revision drift')
        require(digest(item['metadata']) == item['metadata_sha256'], 'upstream metadata receipt drift')
        snapshot = Path(item['snapshot'])
        repo_root = snapshot.parent.parent
        require((repo_root / 'refs' / 'main').read_text() == item['revision'], 'Hub cache ref drift')
        require({p.name for p in snapshot.parent.iterdir()} == {item['revision']}, 'unexpected snapshot in isolated cache')
        expected = {f['path'] for f in item['files']}
        actual = {str(p.resolve()) for p in snapshot.rglob('*') if p.is_file()}
        require(actual == expected, 'snapshot input closure changed')
        for file in item['files']:
            path = Path(file['path'])
            physical(path)
            if hash_weights:
                require(digest(path) == file['sha256'], 'prepared artifact bytes changed')


def prepared(root):
    value = read(root / 'prepared.json')
    require(value['sceneworks_source'] == SOURCE and value['inference'] == INFERENCE, 'prepared source drift')
    require(digest(value['binary']) == value['binary_sha256'], 'native binary drift')
    require(digest(root / 'requirements.lock') == value['requirements_sha256'], 'dependency lock drift')
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
                runtime_versions=dict(sceneworks_source=SOURCE, inference=INFERENCE))
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


def kill_owned(process):
    import psutil
    if process.poll() is not None:
        return
    try:
        children = psutil.Process(process.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        children = []
    for child in reversed(children):
        with contextlib.suppress(psutil.NoSuchProcess):
            child.terminate()
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


def calibrate(root, value):
    prep = prepared(root)
    protocol = read(root / 'protocol.lock.json')
    require(protocol['sha256'] == value['protocol_sha256'], 'protocol differs from coordinator-reviewed hash')
    sys.path.insert(0, str(root / 'harness'))
    from evaluate import load_lock, cells, receipts
    manifest = load_lock(root / 'protocol.lock.json', verify_inputs=True)
    lease = value['lease']
    require(lease['gpu_uuid'] == manifest['hardware']['gpu_uuid'] and lease['host'] == manifest['hardware']['host'] == socket.gethostname(), 'lease/device/host mismatch')
    require(lease['id'] == manifest['ownership']['lease'] and lease['owner'] == manifest['ownership']['responsible_chat'], 'lease owner mismatch')
    require(any(g['index'] == '1' and g['uuid'] == lease['gpu_uuid'] == RUNNER_GPU_UUID for g in gpu_rows()), 'leased physical GPU1 absent')
    require(len(list(cells(manifest, 'calibration'))) == 6, 'calibration must contain exactly six cells')
    campaign = root / 'calibration'
    require(not campaign.exists(), 'calibration already attempted; inspect receipt, no automatic retry')
    import psutil
    lock_record = dict(lease_id=lease['id'], host=lease['host'], gpu_uuid=lease['gpu_uuid'],
        protocol_sha256=protocol['sha256'], deadline_epoch=timestamp(value['deadline_utc']).timestamp(),
        owner_pid=os.getpid(), responsible_chat=lease['owner'],
        job=os.environ.get('GITHUB_RUN_ID', 'local') + '-' + os.environ.get('GITHUB_RUN_ATTEMPT', '1'))
    with device_lock(root.parent, lease['gpu_uuid'], lock_record) as held_lock:
        observation = idle_device(lease)
        campaign.mkdir()
        write(campaign / 'background-device.json', observation)
        env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=lease['gpu_uuid'], HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', MOSS_DEVICE_LOCK=str(held_lock.resolve()), MOSS_LEASE_ID=lease['id'], MOSS_PROTOCOL_SHA256=protocol['sha256'])
        for config in manifest['configs']:
            verify_staged(root, prep['staged'], hash_weights=False)
            require(utc() < timestamp(value['deadline_utc']), 'calibration deadline reached')
            log = campaign / ('transport-' + config['id'] + '.log')
            with log.open('x', encoding='utf-8') as stream:
                process = subprocess.Popen([str(python(root)), str(root / 'harness' / 'run.py'), str(root / 'protocol.lock.json'), str(campaign), '--config', config['id'], '--phase', 'calibration'], env=env, stdout=stream, stderr=subprocess.STDOUT)
                try:
                    while process.poll() is None:
                        require(utc() < timestamp(value['deadline_utc']), 'calibration deadline reached')
                        try:
                            owned = {process.pid} | {p.pid for p in psutil.Process(process.pid).children(recursive=True)}
                        except psutil.NoSuchProcess:
                            owned = {process.pid}
                        require(compute_pids(lease['gpu_uuid']) <= owned | desktop_identities(lease), 'foreign process appeared on leased device; aborting owned runtime')
                        time.sleep(1)
                    require(process.returncode == 0, f'calibration process failed; inspect {log}')
                finally:
                    kill_owned(process)
            # A completed command can still have failure receipts; do not advance silently.
            _, results = receipts(root / 'protocol.lock.json', campaign, 'calibration', complete=False)
            require(all(r['status'] == 'ok' for r in results.values()), 'calibration cell failed; review receipt before any further allocation')
            verify_staged(root, prep['staged'], hash_weights=False)
            idle_device(lease)  # Owned model processes must release before the next configuration.
        _, results = receipts(root / 'protocol.lock.json', campaign, 'calibration')
        estimate = {key: sum(r['metrics']['generation_seconds'] for r in results.values() if r['cell']['config_id'] == key) * 48 for key in REPOS}
        write(root / 'calibration-estimate.json', dict(scored_outputs=288, calibration_outputs=6, estimate_generation_seconds=estimate,
            limitations='512x512 calibration does not estimate full resolution scaling; measured load time is separate. Reviewed desktop baseline is recorded in calibration/background-device.json and can affect device-wide memory/timing. Review before scored authorization.', next_action='Coordinator reviews measured calibration and gives separate scored command. Transport cannot run scored cells.'))


def main():
    global DEADLINE, COMMAND_LOG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'freeze', 'calibration'))
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
        log=str(COMMAND_LOG.resolve()), resume_command=f'python {Path(__file__).resolve()} {args.mode} (same reviewed MOSS_CONTROL_JSON; failed calibration requires reviewed fresh campaign)',
        next_action='Inspect preparation/protocol/calibration receipts before next explicit command', started_utc=utc().isoformat(), accelerator_execution_requested=args.mode == 'calibration', workflow_revision=os.environ.get('GITHUB_SHA'), transport_sha256=digest(__file__), sceneworks_source=SOURCE, inference=INFERENCE)
    try:
        if args.mode == 'prepare':
            prepare(root, value)
        else:
            # Freeze/calibration run in the exact locked environment prepared earlier.
            require(Path(sys.executable).resolve() == python(root).resolve(), 'use prepared venv Python for freeze/calibration')
            (freeze if args.mode == 'freeze' else calibrate)(root, value)
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
