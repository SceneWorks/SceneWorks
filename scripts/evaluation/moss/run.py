#!/usr/bin/env python3
"""Bounded serial execution under the same live device/cache guard as transport."""
import argparse
import contextlib
import multiprocessing
import os
from pathlib import Path
import queue
import sys
import time
import uuid

from evaluate import (CONFIG_IDS, cells, file_hash, load_lock, parity_cells, read,
    receipts, require, seal, validate_manifest, validate_parity, validate_receipt, write)


def worker(config, requests, replies, log, gpu_uuid, mode='candidate'):
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=gpu_uuid)
    with open(log,'a',encoding='utf-8') as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
        runtime=None
        try:
            start=time.monotonic()
            if config['id'].startswith('moss_'):
                from reference import Reference
                runtime=Reference(config,mode=mode)
            else:
                from native import Native
                runtime=Native(config)
            replies.put(('ready',time.monotonic()-start))
            while True:
                request=requests.get()
                if request is None:return
                cell,output=request
                value=runtime.predictions(cell) if mode=='parity' else runtime.render(cell,Path(output))
                replies.put(('result',value))
        except BaseException as exc:
            import traceback
            traceback.print_exc()
            replies.put(('error',f'{type(exc).__name__}: {exc}'))
        finally:
            if runtime is not None and hasattr(runtime,'close'):runtime.close()


def process_rss(pid):
    import psutil
    try:
        root=psutil.Process(pid)
        return sum(p.memory_info().rss for p in [root]+root.children(recursive=True) if p.is_running())
    except psutil.NoSuchProcess:return 0


def receive(process,replies,timeout,guard):
    # The authorization deadline always caps the cell timeout, including initialization.
    deadline=min(time.time()+timeout,guard.deadline_epoch);peak=0;next_check=0
    while time.time()<deadline:
        if time.monotonic()>=next_check:
            guard.check();next_check=time.monotonic()+1
        peak=max(peak,process_rss(process.pid))
        try:
            reply=replies.get(timeout=min(.1,max(.001,deadline-time.time())))
            guard.check()  # no result is accepted after lease/cache/owner changes
            return reply,peak
        except queue.Empty:
            if not process.is_alive():raise RuntimeError(f'owned runtime exited {process.exitcode}')
    guard.check()
    raise TimeoutError(f'owned runtime exceeded bounded {timeout}s; no automatic retry')


def stop(process):
    import psutil
    if process and process.is_alive():
        root=psutil.Process(process.pid)
        for child in root.children(recursive=True):
            with contextlib.suppress(psutil.NoSuchProcess):child.kill()
        process.terminate();process.join(timeout=10)
        if process.is_alive():process.kill();process.join(timeout=5)


def finish(process,requests):
    if process:
        with contextlib.suppress(Exception):requests.put(None)
        process.join(timeout=10)
        stop(process)
        require(not process.is_alive(),'owned runtime failed to stop')


def launch(config,manifest,log,guard,mode='candidate'):
    guard.check()
    ctx=multiprocessing.get_context('spawn');requests=ctx.Queue();replies=ctx.Queue()
    process=ctx.Process(target=worker,args=(config,requests,replies,str(log),manifest['hardware']['gpu_uuid'],mode))
    process.start()
    return process,requests,replies


def snapshot_log(run,log,destination):
    destination.write_bytes(log.read_bytes())
    return dict(path=destination.relative_to(run).as_posix(),sha256=file_hash(destination))


def ensure_parity(manifest,lock_hash,run,phase,guard,attempt):
    """Separate oracle process, joined before any measured candidate process exists."""
    config=next(c for c in manifest['configs'] if c['id']=='moss_dmd')
    root=run/'parity';root.mkdir(exist_ok=True);pending=[];records=[]
    for cell in parity_cells(manifest,phase):
        path=root/(cell['id']+'.json')
        if path.exists():records.append(validate_parity(run,lock_hash,cell,read(path)))
        else:pending.append(cell)
    if not pending:return records
    log=root/(attempt+'.log');log.write_text('Separate DMD oracle/candidate prediction fixture process; no decoded images\n')
    process=requests=None
    try:
        process,requests,replies=launch(config,manifest,log,guard,mode='parity')
        (state,value),_=receive(process,replies,manifest['ownership']['timeout_seconds'],guard)
        require(state=='ready',str(value))
        initialization_seconds=value
        for index,cell in enumerate(pending):
            guard.check();requests.put((cell,None))
            (state,evidence),_=receive(process,replies,manifest['ownership']['timeout_seconds'],guard)
            require(state=='result',str(evidence))
            evidence['initialization_seconds']=initialization_seconds if index==0 else 0.0
            evidence['total_process_seconds']=evidence['initialization_seconds']+evidence['elapsed_seconds']
            row=seal(dict(lock_sha256=lock_hash,cell=cell,status='ok',attempt=attempt,evidence=evidence,ownership=dict(manifest['ownership'],revision=lock_hash),
                log=snapshot_log(run,log,root/(cell['id']+'.'+attempt+'.log'))))
            validate_parity(run,lock_hash,cell,row)
            write(root/(cell['id']+'.json'),row);records.append({k:v for k,v in row.items() if k!='sha256'})
    except BaseException as exc:
        created=[read(p)['cell']['id'] for p in root.glob('*.json') if read(p).get('attempt')==attempt]
        write(run/'invalidations'/(attempt+'-parity.json'),seal(dict(lock_sha256=lock_hash,cell_ids=[],parity_cell_ids=created,reason=str(exc),attempt=attempt)))
        raise
    finally:
        finish(process,requests)
    guard.check()
    return records


def existing_successes(manifest,lock_hash,run,config_id,phase):
    selected=[c for c in cells(manifest,phase) if c['config_id']==config_id]
    pending=[]
    for cell in selected:
        path=run/'receipts'/(cell['id']+'.json')
        if path.exists():
            row=validate_receipt(run,lock_hash,cell,read(path))
            require(row['status']=='ok','failed attempts cannot be retried automatically')
        else:
            require(not (run/'outputs'/(cell['id']+'.png')).exists(),'orphan image blocks retry; retain and reconcile image budget')
            pending.append(cell)
    return pending


def execute(lock,run,config_id,phase):
    require(config_id in CONFIG_IDS,'unknown configuration')
    manifest=load_lock(lock);lock_hash=read(lock)['sha256'];run=Path(run).resolve()
    # Direct CLI shares the complete cache, physical GPU1, owner and scored-authorization policy.
    from ownership import ExecutionGuard
    guard=ExecutionGuard(manifest,lock_hash,phase=phase)
    guard.check()
    validate_manifest(manifest,config_id=config_id)
    guard.check()
    config=next(c for c in manifest['configs'] if c['id']==config_id)
    selected=existing_successes(manifest,lock_hash,run,config_id,phase)
    if phase=='scored':
        _,calibration=receipts(lock,run,'calibration')
        own=next(c for c in cells(manifest,'calibration') if c['config_id']==config_id)
        require(calibration[own['id']]['status']=='ok' or config['unavailable'],'failed calibration needs explicitly reviewed unavailable freeze')
    if not selected:return 0.0
    for directory in ('receipts','outputs','invalidations'):(run/directory).mkdir(parents=True,exist_ok=True)
    attempt=f'{phase}-{config_id}-{uuid.uuid4().hex}'
    log=run/(attempt+'.log');log.write_text(f'lock={lock_hash} phase={phase} config={config_id}\n')
    timeout=manifest['ownership']['timeout_seconds'];process=requests=None
    failure=config['unavailable'];load_seconds=None;load_peak=0;parity=[];touched=[];start=time.monotonic()
    try:
        if not failure and config_id=='moss_dmd':parity=ensure_parity(manifest,lock_hash,run,phase,guard,attempt)
        if not failure:
            process,requests,replies=launch(config,manifest,log,guard)
            (state,value),load_peak=receive(process,replies,timeout,guard)
            require(state=='ready',str(value));load_seconds=value
        for index,cell in enumerate(selected):
            guard.check();touched.append(cell['id'])
            output=run/'outputs'/(cell['id']+'.png');cell_start=time.monotonic();metrics=None;error=None
            status='unavailable' if failure else 'failed'
            if failure:error=failure['reason']
            else:
                try:
                    requests.put((cell,str(output)))
                    (state,value),peak=receive(process,replies,timeout,guard)
                    require(state=='result',str(value));metrics=value
                    metrics.update(peak_host_rss_bytes=max(peak,load_peak if index==0 else 0),download_bytes=0,
                        cold_seconds=(load_seconds+metrics['generation_seconds']) if index==0 else None,
                        warm_seconds=metrics['generation_seconds'] if index>0 else None,
                        latency_condition='cold_process' if index==0 else 'resident_model',
                        load_seconds=load_seconds if config_id.startswith('moss_') and index==0 else metrics.get('load_seconds'))
                    if config_id.startswith('moss_'):
                        metrics['peak_gpu_bytes']=metrics['peak_gpu_reserved_bytes'];metrics['gpu_memory_method']='PyTorch reserved allocator peak; allocated peak also retained'
                    if config_id=='moss_dmd':
                        metrics['parity_fixture_digests']=[seal(r)['sha256'] for r in parity]
                    status='ok'
                except (ValueError,RuntimeError,TimeoutError) as exc:
                    # Authorization/cache changes invalidate the attempt rather than becoming mere render failures.
                    guard.check()
                    error=str(exc);status='timeout' if isinstance(exc,TimeoutError) else 'failed';failure={'reason':error};stop(process)
            with log.open('a') as stream:stream.write(f"{cell['id']}: {status} {error or ''}\n")
            record=dict(lock_sha256=lock_hash,cell=cell,status=status,elapsed_seconds=time.monotonic()-cell_start,
                ownership=dict(manifest['ownership'],revision=lock_hash),metrics=metrics,
                reproduce=f'guarded bootstrap {phase} resume at lock {lock_hash}; never repeat an existing image',error=error,
                output=dict(path=output.relative_to(run).as_posix(),sha256=file_hash(output)) if output.exists() else None,
                log=snapshot_log(run,log,run/'receipts'/(cell['id']+'.'+attempt+'.log')))
            if status=='ok':validate_receipt(run,lock_hash,cell,seal(record))
            write(run/'receipts'/(cell['id']+'.json'),seal(record))
            print(f"{cell['id']}: {status}",flush=True);load_peak=0
        finish(process,requests);process=None
        guard.check()
        validate_manifest(manifest,config_id=config_id)
        guard.check()
    except BaseException as exc:
        created_parity=[read(p)['cell']['id'] for p in (run/'parity').glob('*.json') if read(p).get('attempt')==attempt]
        write(run/'invalidations'/(attempt+'.json'),seal(dict(lock_sha256=lock_hash,config_id=config_id,cell_ids=touched,parity_cell_ids=created_parity,reason=str(exc),attempt=attempt)))
        raise
    finally:finish(process,requests)
    return time.monotonic()-start


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('lock');p.add_argument('run');p.add_argument('--config',required=True,choices=CONFIG_IDS);p.add_argument('--phase',required=True,choices=('calibration','scored'))
    a=p.parse_args();execute(a.lock,a.run,a.config,a.phase)


if __name__=='__main__':
    try:main()
    except (ValueError,RuntimeError,KeyError,OSError) as exc:
        print(f'run rejected: {exc}',file=sys.stderr);sys.exit(1)
