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


class RuntimeFailure(RuntimeError):
    """Observed owned worker failure after the live guard passed."""


class RuntimeTimeout(TimeoutError,RuntimeFailure):
    pass


def expect_runtime(state,value,expected):
    if state=='error':raise RuntimeFailure(str(value))
    if state=='timeout':raise RuntimeTimeout(str(value))
    require(state==expected,str(value))


def worker_failure(exc):
    # Unknown/programming/input/dependency errors fail closed as integrity errors.
    # Even a runtime error needs coordinator diagnosis before unavailability.
    if isinstance(exc,TimeoutError):return 'timeout'
    return 'error' if isinstance(exc,RuntimeError) else 'integrity_error'


def worker(config, requests, replies, log, gpu_uuid, mode='candidate'):
    # Verified pipeline source lives inside the immutable snapshot closure.
    # importlib must not create a .pyc there, including direct CLI spawned workers.
    sys.dont_write_bytecode=True
    from ownership import device_identity
    index,_=device_identity(dict(gpu_uuid=gpu_uuid,**({'physical_gpu_index':config['settings']['physical_gpu_index']} if 'physical_gpu_index' in config['settings'] else {})))
    os.environ['MOSS_PHYSICAL_GPU_INDEX']=str(index)
    os.environ.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=gpu_uuid)
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
            replies.put((worker_failure(exc),f'{type(exc).__name__}: {exc}'))
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
            if not process.is_alive():raise RuntimeFailure(f'owned runtime exited {process.exitcode}')
    guard.check()
    raise RuntimeTimeout(f'owned runtime exceeded bounded {timeout}s; no automatic retry')


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
    from calibration_source import carried
    _,prior=carried(manifest,run)
    for cell in parity_cells(manifest,phase):
        if cell['id'] in prior:
            records.append(prior[cell['id']]);continue
        path=root/(cell['id']+'.json')
        if path.exists():records.append(validate_parity(run,lock_hash,cell,read(path)))
        else:pending.append(cell)
    if not pending:return records
    log=root/(attempt+'.log');log.write_text('Separate DMD oracle/candidate prediction fixture process; no decoded images\n')
    process=requests=None
    try:
        process,requests,replies=launch(config,manifest,log,guard,mode='parity')
        (state,value),_=receive(process,replies,manifest['ownership']['timeout_seconds'],guard)
        expect_runtime(state,value,'ready')
        initialization_seconds=value
        for index,cell in enumerate(pending):
            guard.check();requests.put((cell,None))
            (state,evidence),_=receive(process,replies,manifest['ownership']['timeout_seconds'],guard)
            expect_runtime(state,evidence,'result')
            evidence['initialization_seconds']=initialization_seconds if index==0 else 0.0
            evidence['total_process_seconds']=evidence['initialization_seconds']+evidence['elapsed_seconds']
            row=seal(dict(lock_sha256=lock_hash,cell=cell,status='ok',attempt=attempt,evidence=evidence,ownership=dict(manifest['ownership'],revision=lock_hash),
                log=snapshot_log(run,log,root/(cell['id']+'.'+attempt+'.log'))))
            validate_parity(run,lock_hash,cell,row)
            write(root/(cell['id']+'.json'),row);records.append({k:v for k,v in row.items() if k!='sha256'})
    except RuntimeFailure:
        guard.check()
        raise
    except BaseException as exc:
        created=[read(p)['cell']['id'] for p in root.glob('*.json') if read(p).get('attempt')==attempt]
        write(run/'invalidations'/(attempt+'-parity.json'),seal(dict(lock_sha256=lock_hash,cell_ids=[],parity_cell_ids=created,reason=str(exc),attempt=attempt)))
        raise
    finally:
        finish(process,requests)
    guard.check()
    return records


def existing_successes(manifest,lock_hash,run,config_id,phase,approved=None):
    from calibration_source import carried
    carried_rows,_=carried(manifest,run)
    selected=[c for c in cells(manifest,phase) if c['config_id']==config_id]
    pending=[]
    for cell in selected:
        if cell['id'] in carried_rows:continue
        path=run/'receipts'/(cell['id']+'.json')
        if path.exists():
            row=validate_receipt(run,lock_hash,cell,read(path),manifest=manifest)
            terminal_calibration=phase=='calibration' and row['status'] in ('failed','timeout') and row.get('failure_kind')=='runtime'
            terminal_unavailable=phase=='scored' and ((config_id in (approved or {}) and row['status']=='unavailable') or (row['status'] in ('failed','timeout') and row.get('failure_kind') in ('runtime','runtime_censored')))
            require(row['status']=='ok' or terminal_calibration or terminal_unavailable,'failed attempts cannot be retried automatically')
        else:
            require(not (run/'outputs'/(cell['id']+'.png')).exists(),'orphan image blocks retry; retain and reconcile image budget')
            pending.append(cell)
    return pending


def execute(lock,run,config_id,phase):
    require(config_id in CONFIG_IDS,'unknown configuration')
    manifest=load_lock(lock);lock_hash=read(lock)['sha256'];run=Path(run).resolve()
    # Direct CLI shares the complete cache, frozen physical GPU, owner and scored-authorization policy.
    from ownership import ExecutionGuard
    guard=ExecutionGuard(manifest,lock_hash,phase=phase)
    guard.check()
    validate_manifest(manifest,config_id=config_id)
    guard.check()
    require(run==(guard.prepared_root/'run').resolve() and Path(lock).resolve()==(guard.prepared_root/'protocol.lock.json').resolve(),'run/lock must use original prepared campaign authority')
    config=next(c for c in manifest['configs'] if c['id']==config_id)
    approved=getattr(guard,'unavailable_configs',{})
    selected=existing_successes(manifest,lock_hash,run,config_id,phase,approved)
    if phase=='scored':
        _,calibration=receipts(lock,run,'calibration')
        own=next(c for c in cells(manifest,'calibration') if c['config_id']==config_id)
        require(calibration[own['id']]['status']=='ok' or config_id in approved,'failed calibration requires explicit approved availability decision')
    if not selected:return 0.0
    stopped=None
    if phase=='scored':
        ordered=[c for c in cells(manifest) if c['config_id']==config_id]
        order={c['id']:i for i,c in enumerate(ordered)}
        for cell in ordered:
            path=run/'receipts'/(cell['id']+'.json')
            if path.exists() and read(path).get('failure_kind')=='runtime':
                require(stopped is None,'multiple actual failures in stopped config history')
                stopped=validate_receipt(run,lock_hash,cell,read(path),manifest=manifest)
        if stopped:
            require(all(order[c['id']]>order[stopped['cell']['id']] for c in selected),'missing receipt before actual runtime failure; cannot rerender or censor earlier cells')
    for directory in ('receipts','outputs','invalidations'):(run/directory).mkdir(parents=True,exist_ok=True)
    attempt=f'{phase}-{config_id}-{uuid.uuid4().hex}'
    log=run/(attempt+'.log');log.write_text(f'lock={lock_hash} phase={phase} config={config_id}\n')
    timeout=manifest['ownership']['timeout_seconds'];process=requests=None
    decision=approved.get(config_id) if phase=='scored' else None
    failure=decision or ({'reason':stopped['error']} if stopped else config['unavailable']);load_seconds=None;load_peak=0;parity=[];touched=[];start=time.monotonic();initial_error=None;failure_cell=stopped['cell']['id'] if stopped else None
    try:
        try:
            if not failure and config_id=='moss_dmd':parity=ensure_parity(manifest,lock_hash,run,phase,guard,attempt)
            if not failure:
                process,requests,replies=launch(config,manifest,log,guard)
                (state,value),load_peak=receive(process,replies,timeout,guard)
                expect_runtime(state,value,'ready');load_seconds=value
        except RuntimeFailure as exc:
            guard.check()
            initial_error=exc;failure={'reason':str(exc)};stop(process)
            parity_log=run/'parity'/(attempt+'.log')
            if config_id=='moss_dmd' and parity_log.exists():
                with log.open('ab') as stream:stream.write(parity_log.read_bytes())
        for index,cell in enumerate(selected):
            guard.check();touched.append(cell['id'])
            output=run/'outputs'/(cell['id']+'.png');cell_start=start if index==0 else time.monotonic();metrics=None;error=None
            attempted=not failure or (initial_error is not None and index==0)
            if failure and not decision and not config['unavailable'] and failure_cell is None:failure_cell=cell['id']
            status='unavailable' if decision or config['unavailable'] else ('timeout' if attempted and isinstance(initial_error,TimeoutError) else 'failed')
            if failure:error=failure['reason']
            else:
                try:
                    requests.put((cell,str(output)))
                    (state,value),peak=receive(process,replies,timeout,guard)
                    expect_runtime(state,value,'result');metrics=value
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
                except RuntimeFailure as exc:
                    # Authorization/cache changes invalidate the attempt rather than becoming mere render failures.
                    guard.check()
                    error=str(exc);status='timeout' if isinstance(exc,TimeoutError) else 'failed';failure={'reason':error};failure_cell=cell['id'];stop(process)
            with log.open('a',encoding='utf-8') as stream:stream.write(f"{cell['id']}: {status} {error or ''}\n")
            record=dict(lock_sha256=lock_hash,cell=cell,status=status,elapsed_seconds=time.monotonic()-cell_start,
                ownership=dict(manifest['ownership'],revision=lock_hash),metrics=metrics,
                reproduce=decision['reproduce'] if decision else f'guarded bootstrap {phase} at lock {lock_hash}; failed attempts never retry',error=error,
                output=dict(path=output.relative_to(run).as_posix(),sha256=file_hash(output)) if output.exists() else None,
                log=snapshot_log(run,log,run/'receipts'/(cell['id']+'.'+attempt+'.log')))
            record['attempted']=attempted
            if status in ('failed','timeout'):
                record.update(failure_kind='runtime' if attempted else 'runtime_censored',runtime_failure_cell=failure_cell)
                if not attempted:record['censored_reason']=f'Not attempted after {failure_cell} stopped this owned runtime; no retry'
            if approved:record['availability_authorization']=dict(path='availability-authorization.json',sha256=file_hash(run/'availability-authorization.json'))
            validate_receipt(run,lock_hash,cell,seal(record),manifest=manifest)
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
