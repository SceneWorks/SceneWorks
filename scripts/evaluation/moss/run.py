#!/usr/bin/env python3
"""Explicit bounded serial execution. Never downloads, retries, or expands the plan."""
import argparse
import contextlib
import multiprocessing
import os
from pathlib import Path
import queue
import socket
import sys
import time

from evaluate import CONFIG_IDS, cells, file_hash, load_lock, read, receipts, require, seal, validate_manifest, validate_receipt, write


def worker(config, requests, replies, log, gpu_uuid):
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID';os.environ['CUDA_VISIBLE_DEVICES']=gpu_uuid
    with open(log,'a',encoding='utf-8') as stream, contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
        runtime=None
        try:
            start=time.monotonic()
            if config['id'].startswith('moss_'):
                from reference import Reference
                runtime=Reference(config)
            else:
                from native import Native
                runtime=Native(config)
            load_seconds=time.monotonic()-start
            replies.put(('ready',load_seconds))
            while True:
                request=requests.get()
                if request is None:return
                cell,output=request
                metrics=runtime.render(cell,Path(output))
                replies.put(('result',metrics))
        except BaseException as exc:
            import traceback
            traceback.print_exc()
            replies.put(('error',f'{type(exc).__name__}: {exc}'))
        finally:
            if runtime is not None and hasattr(runtime,'close'):runtime.close()


def process_rss(pid,extra_pid=None):
    import psutil
    total=0
    for root_pid in {pid,extra_pid}-{None}:
        try:
            root=psutil.Process(root_pid)
            total+=sum(p.memory_info().rss for p in [root]+root.children(recursive=True) if p.is_running())
        except psutil.NoSuchProcess:pass
    return total


def receive(process,replies,timeout,extra_pid=None):
    deadline=time.monotonic()+timeout;peak=0
    while time.monotonic()<deadline:
        peak=max(peak,process_rss(process.pid,extra_pid))
        try:return replies.get(timeout=.1),peak
        except queue.Empty:
            if not process.is_alive():raise RuntimeError(f'owned runtime exited {process.exitcode}')
    raise TimeoutError(f'owned runtime exceeded {timeout}s; no automatic retry')


def stop(process):
    import psutil
    if process and process.is_alive():
        # Only the explicitly spawned reference runtime tree, never native service PID.
        root=psutil.Process(process.pid)
        for child in root.children(recursive=True):
            with contextlib.suppress(psutil.NoSuchProcess):child.kill()
        process.terminate();process.join(timeout=10)
        if process.is_alive():process.kill();process.join(timeout=5)


def validate_lease(manifest,lock_hash,environ=None,now=None,ancestor_pids=None):
    environ=os.environ if environ is None else environ
    require(environ.get('MOSS_DEVICE_LOCK'),'coordinator-held device lease required')
    path=Path(environ['MOSS_DEVICE_LOCK'])
    require(path.is_absolute() and path.is_file() and not path.is_symlink(),'invalid device lease file')
    lease=read(path)
    require(lease['lease_id']==manifest['ownership']['lease']==environ.get('MOSS_LEASE_ID'),'device lease identity mismatch')
    require(lease['protocol_sha256']==lock_hash==environ.get('MOSS_PROTOCOL_SHA256'),'device lease freeze mismatch')
    require(lease['host'].casefold()==manifest['hardware']['host'].casefold()==socket.gethostname().casefold(),'device lease host mismatch')
    require(lease['gpu_uuid']==manifest['hardware']['gpu_uuid'],'device lease GPU mismatch')
    require(lease['responsible_chat']==manifest['ownership']['responsible_chat'],'device lease owner chat mismatch')
    require(lease['deadline_epoch']>(time.time() if now is None else now),'device lease expired')
    if ancestor_pids is None:
        import psutil
        ancestor_pids={p.pid for p in psutil.Process().parents()}
    require(type(lease['owner_pid']) is int and lease['owner_pid'] in ancestor_pids,'live coordinator ancestor does not hold lease')
    return lease


def execute(lock,run,config_id,phase):
    require(config_id in CONFIG_IDS,'unknown configuration')
    manifest=load_lock(lock,verify_inputs=True);lock_hash=read(lock)['sha256'];run=Path(run).resolve()
    config=next(c for c in manifest['configs'] if c['id']==config_id)
    if not config['unavailable']:validate_lease(manifest,lock_hash)
    selected=[c for c in cells(manifest,phase) if c['config_id']==config_id]
    if phase=='scored':
        _,calibration=receipts(lock,run,'calibration')
        calibration_cell=next(c for c in cells(manifest,'calibration') if c['config_id']==config_id)
        if calibration[calibration_cell['id']]['status']!='ok':
            require(config['unavailable'],'failed calibration requires a new freeze with explicit unavailable reason before scored phase')
    run.mkdir(parents=True,exist_ok=True);(run/'receipts').mkdir(exist_ok=True);(run/'outputs').mkdir(exist_ok=True)
    log=run/f'{phase}-{config_id}.log'
    require(not log.exists(),'configuration already attempted; inspect receipts and create explicitly reviewed replacement run, never silently retry')
    log.write_text(f'lock={lock_hash} phase={phase} config={config_id}\n',encoding='utf-8')
    ctx=multiprocessing.get_context('spawn');requests=ctx.Queue();replies=ctx.Queue();process=None
    timeout=manifest['ownership']['timeout_seconds'];failure=config['unavailable'];load_seconds=None;load_peak=0
    start=time.monotonic()
    try:
        if not failure:
            process=ctx.Process(target=worker,args=(config,requests,replies,str(log),manifest['hardware']['gpu_uuid']));process.start()
            try:
                (state,value),load_peak=receive(process,replies,timeout,config['settings'].get('worker_pid'))
                require(state=='ready',str(value));load_seconds=value
            except (ValueError,RuntimeError,TimeoutError) as exc:failure={'reason':str(exc),'initial_status':'timeout' if isinstance(exc,TimeoutError) else 'failed'}
        for index,cell in enumerate(selected):
            if not failure:validate_lease(manifest,lock_hash)
            receipt_path=run/'receipts'/(cell['id']+'.json')
            require(not receipt_path.exists(),'cell already has receipt')
            output=run/'outputs'/(cell['id']+'.png');cell_start=time.monotonic();metrics=None;error=None
            status='unavailable' if config['unavailable'] else 'failed'
            if failure:
                error=failure['reason']
                if not config['unavailable']:status=failure.get('initial_status','unavailable') if index==0 else 'unavailable'
            else:
                try:
                    requests.put((cell,str(output)))
                    (state,value),peak=receive(process,replies,timeout,config['settings'].get('worker_pid'))
                    require(state=='result',str(value));metrics=value
                    metrics.update(peak_host_rss_bytes=max(peak,load_peak if index==0 else 0),download_bytes=0,
                        cold_seconds=(load_seconds+metrics['generation_seconds']) if index==0 else None,
                        warm_seconds=metrics['generation_seconds'] if index>0 else None,
                        latency_condition='cold_process' if index==0 else 'resident_model',
                        load_seconds=load_seconds if config_id.startswith('moss_') and index==0 else metrics.get('load_seconds'))
                    if config_id.startswith('moss_'):
                        metrics['peak_gpu_bytes']=metrics['peak_gpu_reserved_bytes'];metrics['gpu_memory_method']='PyTorch reserved allocator peak; allocated peak also retained'
                    status='ok'
                except (ValueError,RuntimeError,TimeoutError) as exc:
                    error=str(exc);status='timeout' if isinstance(exc,TimeoutError) else 'failed';failure={'reason':error};stop(process)
            with log.open('a',encoding='utf-8') as stream:stream.write(f"{cell['id']}: {status} {error or ''}\n")
            record=dict(lock_sha256=lock_hash,cell=cell,status=status,elapsed_seconds=time.monotonic()-cell_start,
                ownership=dict(manifest['ownership'],revision=lock_hash),metrics=metrics,
                reproduce=f'python scripts/evaluation/moss/run.py {lock} {run} --config {config_id} --phase {phase} (use a fresh explicitly approved output directory)',
                error=error,output=dict(path=str(output.relative_to(run)),sha256=file_hash(output)) if output.exists() else None,
                log=dict(path=log.name,sha256=file_hash(log)))
            # Per-cell log snapshots remain immutable while shared runtime logging continues.
            snapshot=run/'receipts'/(cell['id']+'.log');snapshot.write_bytes(log.read_bytes())
            record['log']=dict(path=str(snapshot.relative_to(run)),sha256=file_hash(snapshot))
            if status=='ok':
                try:validate_receipt(run,lock_hash,cell,seal(record))
                except (ValueError,KeyError,TypeError) as exc:
                    record['status']='failed';record['error']=f'incomplete measurements: {exc}'
                    failure={'reason':record['error']};stop(process)
            write(receipt_path,seal(record))
            print(f"{cell['id']}: {status}",flush=True)
            load_peak=0
        validate_manifest(manifest)  # post-run byte identity check; never update frozen hashes
    finally:
        if process:
            with contextlib.suppress(Exception):requests.put(None)
            process.join(timeout=3);stop(process)
    return time.monotonic()-start


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('lock');p.add_argument('run');p.add_argument('--config',required=True,choices=CONFIG_IDS);p.add_argument('--phase',required=True,choices=('calibration','scored'))
    a=p.parse_args();execute(a.lock,a.run,a.config,a.phase)


if __name__=='__main__':
    try:main()
    except (ValueError,RuntimeError,KeyError,OSError) as exc:
        print(f'run rejected: {exc}',file=sys.stderr);sys.exit(1)
