"""Shared physical GPU1, lease, cache and scored-authorization policy.

Used by the workflow transport and direct runner. Queries are read-only. This
module never terminates a process; callers own cleanup of their spawned trees.
"""
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def run(argv):
    return subprocess.run(argv, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, timeout=15).stdout

RUNNER_GPU_UUID = 'GPU-e4b79931-7be6-f216-460a-f5405cfafffe'

REPOS = {
 'moss_base': ('OpenMOSS-Team/MOSS-Image-Base', '1ceede80badddd6a7812069619edaa5633587192'),
 'moss_sft': ('OpenMOSS-Team/MOSS-Image-SFT', 'fc5de0d56fc910cbca6f6625a70f5f582b26deb1'),
 'moss_dmd': ('lodestones/MOSS-Image-SFT-8-steps-dmd', '5ff314b4c1470876b114b810146085db8a181a6e'),
 'qwen_image_2_1': ('SceneWorks/qwen-image-2-1-mlx', 'ef03535eac756dd21feb7ac47c4f2df3e9b5b3c0'),
 'flux2_klein_9b': ('SceneWorks/flux2-klein-9b-mlx', '1902693279fcfb828919370dfac2b8922d99499a'),
 'z_image_turbo': ('SceneWorks/z-image-turbo-mlx', 'bb2bc9893b3c49ae96c813350775f791a2e8bc80'),
}

def physical(path):
    for part in [path, *path.parents]:
        require(not part.is_symlink() and not (part.exists() and getattr(part.stat(), 'st_file_attributes', 0) & 1024), 'reparse point in artifact path')

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
    require(set(staged) == set(REPOS), 'all six staged snapshot identities required')
    hub = root / 'hub'
    require({p.name for p in hub.iterdir()} == {'models--' + repo.replace('/', '--') for repo, _ in REPOS.values()}, 'unexpected resolver repository in isolated Hub cache')
    for key, item in staged.items():
        require((item['repo'], item['revision']) == REPOS[key], 'staged model revision drift')
        require(digest(item['metadata']) == item['metadata_sha256'], 'upstream metadata receipt drift')
        snapshot = Path(item['snapshot'])
        require(snapshot == hub / ('models--' + item['repo'].replace('/', '--')) / 'snapshots' / item['revision'], 'snapshot escaped prepared resolver authority')
        physical(snapshot)
        repo_root = snapshot.parent.parent
        require({p.name for p in repo_root.iterdir()} == {'refs', 'snapshots'}, 'unexpected repository resolver metadata')
        require({p.name for p in (repo_root / 'refs').iterdir()} == {'main'}, 'unexpected Hub resolver ref')
        physical(repo_root / 'refs' / 'main')
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


def validate_baseline(lease):
    baseline = lease['desktop_baseline']
    require(type(baseline) is list, 'reviewed desktop baseline list required')
    for item in baseline:
        require(set(item) == {'pid', 'name', 'created_epoch'} and type(item['pid']) is int and item['pid'] > 0
                and isinstance(item['name'], str) and item['name'] and '[' not in item['name']
                and type(item['created_epoch']) in (float, int) and item['created_epoch'] > 0, 'complete known process identity required')
    require(len({p['pid'] for p in baseline}) == len(baseline), 'duplicate desktop process identity')
    require(type(lease['baseline_max_memory_mib']) in (float, int) and 0 <= lease['baseline_max_memory_mib'] <= 4096,
            'reviewed desktop memory ceiling must be between0 and4096MiB')
    require(type(lease['baseline_max_utilization_percent']) in (float, int) and 0 <= lease['baseline_max_utilization_percent'] <= 5,
            'reviewed idle desktop utilization ceiling must be between0 and5percent')


def reject_run_invalidated(run_root, lock_hash):
    from evaluate import unseal
    for path in (Path(run_root)/'invalidations').glob('*.json'):
        require(unseal(read(path))['lock_sha256'] != lock_hash,
                'execution/input invalidations cannot be approved as unavailability')


def calibration_results(manifest, lock_hash, run_root, *, complete=True):
    from evaluate import cells, validate_receipt
    reject_run_invalidated(run_root,lock_hash)
    results={}
    for cell in cells(manifest,'calibration'):
        path=Path(run_root)/'receipts'/(cell['id']+'.json')
        if not path.exists():
            require(not complete,'six terminal validated calibration attempts required')
            continue
        row=validate_receipt(run_root,lock_hash,cell,read(path),manifest=manifest)
        require(row['status']=='ok' or (row['status'] in ('failed','timeout') and row.get('failure_kind')=='runtime'),
                'calibration failure is not an actual guarded runtime attempt')
        results[cell['id']]=row
    return results


def terminal_scored(row, approved):
    return (row['status']=='ok' or
        (row['status']=='unavailable' and row['cell']['config_id'] in approved) or
        (row['status'] in ('failed','timeout') and row.get('failure_kind') in ('runtime','runtime_censored')))


def calibration_bundle(lock, run_root):
    """Identity of six immutable attempts; failure never becomes a success."""
    from evaluate import load_lock, canonical
    results=calibration_results(load_lock(lock),read(lock)['sha256'],run_root)
    return hashlib.sha256(canonical(results)).hexdigest()


def availability_decisions(manifest, lock_hash, run_root, accepted, *, pending_cell=None):
    """Explicit pre-score coordinator disposition of actual runtime failures only."""
    from evaluate import canonical
    results=calibration_results(manifest,lock_hash,run_root)
    reject_run_invalidated(run_root,lock_hash)
    require(accepted['protocol_sha256']==lock_hash,'availability protocol changed')
    authority=accepted['authorization']
    require(authority['id'] and authority['coordinator']==manifest['ownership']['responsible_chat']
        and authority['accepted_calibration'] is True and authority['max_scored_outputs']==288
        and authority['acknowledge_shape_uncertainty'] is True,'explicit coordinator calibration authorization required')
    require(accepted['calibration_receipts_sha256']==hashlib.sha256(canonical(results)).hexdigest(),'accepted calibration attempt bundle changed')
    failed={r['cell']['config_id']:r for r in results.values() if r['status']!='ok'}
    decisions=accepted.get('unavailable_configs',{})
    require(isinstance(decisions,dict) and set(decisions)==set(failed),'every failed calibration requires explicit disposition; healthy configs cannot be excluded')
    for key,decision in decisions.items():
        require(set(decision)=={'calibration_receipt_sha256','calibration_log_sha256','reason','reproduce','code_and_acquisition_defects_excluded'},'exact unavailable evidence fields required')
        row=failed[key]
        require(row['status'] in ('failed','timeout') and row.get('failure_kind')=='runtime','only actual guarded runtime failures can be approved')
        require(decision['code_and_acquisition_defects_excluded'] is True and all(isinstance(decision[k],str) and decision[k].strip() for k in ('reason','reproduce')),'code/acquisition defects must be fixed; explicit reason/reproduction required')
        require(decision['calibration_receipt_sha256']==digest(Path(run_root)/'receipts'/(row['cell']['id']+'.json'))
            and decision['calibration_log_sha256']==row['log']['sha256'],'approved calibration receipt/log evidence changed')
    if decisions:
        scored=list((Path(run_root)/'receipts').glob('scored.*.json'))
        images=list((Path(run_root)/'outputs').glob('scored.*.png'))
        if scored or images:
            mirror=Path(run_root)/'availability-authorization.json'
            require(mirror.is_file() and read(mirror)==accepted,'availability decision must precede scored execution')
            binding=dict(path='availability-authorization.json',sha256=digest(mirror))
            from evaluate import unseal
            require(all(unseal(read(path)).get('availability_authorization')==binding for path in scored),
                    'post-scored exclusion forbidden: existing scores lack original availability binding')
            require(all(path.stem==pending_cell or (Path(run_root)/'receipts'/(path.stem+'.json')).is_file() for path in images),
                    'orphan scored image prevents availability approval')
    return decisions


class ExecutionGuard:
    """One policy for workflow and direct CLI, checked before and during execution."""
    def __init__(self, manifest, protocol_sha256, *, phase='calibration', allow_owner=False,
                 environ=None, now=None, ancestor_pids=None):
        require(phase in ('calibration', 'scored', 'parity'), 'unknown execution phase')
        self.manifest = manifest
        self.protocol_sha256 = protocol_sha256
        self.phase = phase
        self.environ = os.environ if environ is None else environ
        self.now = time.time if now is None else now
        require(self.environ.get('MOSS_DEVICE_LOCK'), 'coordinator-held device lease required')
        self.path = Path(self.environ['MOSS_DEVICE_LOCK'])
        physical(self.path)
        require(self.path.is_absolute() and self.path.is_file(), 'invalid device lease file')
        self.record = read(self.path)
        self.record_sha256 = digest(self.path)
        lease = self.record
        required = {'gpu_uuid', 'lease_id', 'protocol_sha256', 'host', 'responsible_chat', 'phase', 'owner_pid', 'deadline_epoch', 'lease_expires_epoch', 'prepared_root', 'prepared_sha', 'desktop_baseline', 'baseline_max_memory_mib', 'baseline_max_utilization_percent', 'external_owners_reconciled'}
        require(required <= set(lease), 'incomplete held device lease')
        validate_baseline(lease)
        require(lease['external_owners_reconciled'] is True, 'shared host ownership is not reconciled')
        require(manifest['hardware']['gpu_uuid'] == lease['gpu_uuid'] == RUNNER_GPU_UUID,
                'execution is restricted to physical GPU1 UUID')
        require(lease['lease_id'] == manifest['ownership']['lease'] == self.environ.get('MOSS_LEASE_ID'), 'device lease identity mismatch')
        require(lease['protocol_sha256'] == protocol_sha256 == self.environ.get('MOSS_PROTOCOL_SHA256'), 'device lease freeze mismatch')
        require(lease['host'].casefold() == manifest['hardware']['host'].casefold() == socket.gethostname().casefold(), 'device lease host mismatch')
        require(lease['responsible_chat'] == manifest['ownership']['responsible_chat'], 'device lease owner chat mismatch')
        require(lease['phase'] == phase or (lease['phase'] == 'calibration' and phase == 'parity'), 'lease execution phase mismatch')
        import psutil
        ancestors = {p.pid for p in psutil.Process().parents()} if ancestor_pids is None else ancestor_pids
        if allow_owner:
            ancestors = set(ancestors) | {os.getpid()}
        require(type(lease['owner_pid']) is int and lease['owner_pid'] in ancestors, 'live coordinator ancestor does not hold lease')
        self.owner_pid = lease['owner_pid']
        self.owner_created = psutil.Process(self.owner_pid).create_time()
        self.deadline_epoch = min(lease['deadline_epoch'], lease['lease_expires_epoch'])
        preparation = manifest['preparation']
        self.prepared_root = Path(lease['prepared_root'])
        self.prepared_path = self.prepared_root / 'prepared.json'
        require(self.prepared_root.is_absolute() and str(self.prepared_path) == preparation['path'], 'prepared campaign identity mismatch')
        require(lease['prepared_sha'] == preparation['sha256'] == digest(self.prepared_path), 'prepared receipt changed')
        require(self.path == self.prepared_root.parent / 'device-locks' / (RUNNER_GPU_UUID + '.lock'), 'lease file is outside physical device lock authority')
        self.preparation = read(self.prepared_path)
        for config in manifest['configs']:
            item = self.preparation['staged'][config['id']]
            require((config['repo'], config['revision'], config['snapshot']) ==
                    (item['repo'], item['revision'], item['snapshot']), 'prepared configuration snapshot identity mismatch')
            if not config['id'].startswith('moss_'):
                require(Path(config['settings']['hub_cache']) == self.prepared_root / 'hub', 'native resolver is outside isolated prepared cache')
        verify_staged(self.prepared_root, self.preparation['staged'], hash_weights=False)
        self.input_fingerprints = {item['path']: self.fingerprint(Path(item['path']))
                                   for config in manifest['configs'] for item in config['files']}
        self.authorization = None
        self.unavailable_configs = {}
        self.availability_fingerprints = {}
        if phase == 'scored':
            authority = lease.get('scored_authorization')
            require(isinstance(authority, dict) and set(authority) == {'path', 'sha256'}, 'accepted scored authorization missing from held lease')
            path = Path(authority['path'])
            require(path == self.prepared_root / 'scored-authorization.json', 'scored authorization is outside campaign')
            require(digest(path) == authority['sha256'], 'scored authorization receipt changed')
            self.authorization = (path, authority['sha256'])
            accepted = read(path)
            require(accepted['protocol_sha256'] == protocol_sha256 and accepted['authorization']['max_scored_outputs'] == 288
                    and accepted['authorization']['accepted_calibration'] is True
                    and accepted['authorization']['acknowledge_shape_uncertainty'] is True
                    and accepted['authorization']['coordinator'] == lease['responsible_chat'], 'explicit accepted scored authorization required')
            require(accepted['calibration_receipts_sha256'] == calibration_bundle(self.prepared_root / 'protocol.lock.json', self.prepared_root / 'run'), 'accepted calibration receipt bundle changed')
            require(accepted['calibration_estimate_sha256'] == digest(self.prepared_root / 'calibration-estimate.json'), 'accepted calibration estimate changed')
            self.unavailable_configs=availability_decisions(manifest,protocol_sha256,self.prepared_root/'run',accepted)
            if self.unavailable_configs:
                mirror=self.prepared_root/'run'/'availability-authorization.json'
                require(digest(mirror)==authority['sha256'],'availability authorization mirror changed')
                from evaluate import cells, artifact
                evidence=[mirror]
                for cell in cells(manifest,'calibration'):
                    path=self.prepared_root/'run'/'receipts'/(cell['id']+'.json')
                    evidence.extend([path,artifact(self.prepared_root/'run',read(path)['log'])])
                self.availability_fingerprints={str(p):self.fingerprint(p) for p in evidence}

        self.check()

    @staticmethod
    def fingerprint(path):
        physical(path)
        info = path.stat()
        return (info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_ino)

    def check(self, allowed_pids=None, idle=False):
        import psutil
        require(self.now() < self.deadline_epoch, 'device lease expired or execution deadline reached')
        reject_run_invalidated(self.prepared_root/'run',self.protocol_sha256)
        physical(self.path)
        require(digest(self.path) == self.record_sha256, 'held lease changed during execution')
        require(psutil.Process(self.owner_pid).create_time() == self.owner_created, 'coordinator lease owner PID changed')
        require(digest(self.prepared_path) == self.record['prepared_sha'], 'prepared receipt changed during execution')
        verify_staged(self.prepared_root, self.preparation['staged'], hash_weights=False)
        require(all(self.fingerprint(Path(path)) == before for path, before in self.input_fingerprints.items()),
                'frozen input metadata changed during execution; evidence must be invalidated')
        matches = [g for g in gpu_rows() if g['uuid'] == RUNNER_GPU_UUID]
        require(len(matches) == 1 and matches[0]['index'] == '1', 'physical GPU1 index/UUID identity changed')
        baseline = desktop_identities(self.record)
        if allowed_pids is None:
            allowed_pids = {self.owner_pid} | {p.pid for p in psutil.Process(self.owner_pid).children(recursive=True)}
        require(compute_pids(RUNNER_GPU_UUID) <= set(allowed_pids) | baseline,
                'foreign process appeared on leased physical GPU1')
        require(all(self.fingerprint(Path(p))==before for p,before in self.availability_fingerprints.items()),'availability calibration evidence changed during execution')
        if self.authorization:
            require(digest(self.authorization[0]) == self.authorization[1], 'scored authorization changed during execution')
        require(self.now() < self.deadline_epoch, 'device lease expired during ownership check')
        return idle_device(self.record) if idle else self.record
