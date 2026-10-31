"""Exact SceneWorks bounded product CUDA claim, using inference's low-level protocol.

This is a source-owned consumer selector, not an inference registry entry. No
registry substitution or environment opt-out is accepted. This process does no
model work; Node owns the native PID and sends release only after child/stdio close.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import stat
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

SELECTOR = 'tests::moss_training_acceleration::moss_bounded_product_acceptance'
UUID = 'GPU-e4b79931-7be6-f216-460a-f5405cfafffe'
CAP = 80 * 1024**3
RESERVE = 2 * 1024**3
PROFILE = 'non-isolated-correctness-headroom'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def selected_primary_metadata_valid(device):
    """Fresh identity/counters, independent of admitting another capped workload."""
    total, used, free = (device.get(key) for key in ('total_mib', 'used_mib', 'free_mib'))
    utilization = device.get('utilization_percent')
    return (device.get('uuid') == UUID and type(device.get('index')) is int
            and device['index'] == 1
            and all(type(value) is int for value in (total, used, free))
            and total > 0 and 0 <= used <= total and 0 <= free <= total
            and used + free <= total
            and (utilization is None or type(utilization) is int and 0 <= utilization <= 100))


def source_modules(root, pin):
    # The supplied checkout may advance on main. Only load matching pinned bytes.
    names = ('scripts/release/veda_cuda_guard.py', 'scripts/ci/model_proof_resources.py',
             'scripts/ci/model_proof_cuda_primitives.py', 'scripts/ci/windows_cuda_metadata.py')
    seals = {}
    for name in names:
        data = (root / name).read_bytes()
        pinned = subprocess.check_output(['git', 'show', f'{pin}:{name}'], cwd=root, timeout=30)
        require(data == pinned, f'actual inference helper differs from permanent pin: {name}')
        seals[name] = hashlib.sha256(data).hexdigest()
    sys.path.insert(0, str(root))
    from scripts.release import veda_cuda_guard as guard
    from scripts.ci import model_proof_resources as resources
    from scripts.ci import model_proof_cuda_primitives as primitives
    return guard, resources, primitives, seals


def main(args):
    from datetime import datetime
    require(os.name == 'nt' and args.selector == SELECTOR, 'exact Windows bounded product selector required')
    require(os.environ.get('MOSS_ACCEPTANCE_PHASE') == 'bounded-product', 'bounded phase required')
    require(os.environ.get('MODEL_INFERENCE_CASE_FQN') == SELECTOR, 'reader selector binding required')
    visibility = {name: os.environ.get(name) for name in ('CUDA_DEVICE_ORDER','CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES')}
    require(visibility == {'CUDA_DEVICE_ORDER':'PCI_BUS_ID','CUDA_VISIBLE_DEVICES':'1','NVIDIA_VISIBLE_DEVICES':'1'}, 'source-owned physical1 visibility required')
    require(os.environ.get('MOSS_ACCEPTANCE_RESOLVED_CAP_BYTES') == str(CAP), 'fixed source cap required')
    require(os.environ.get('MOSS_ACCEPTANCE_WORKFLOW_SOURCE') == os.environ.get('GITHUB_SHA'), 'actual dispatched source required')
    require(os.environ.get('GITHUB_EVENT_NAME') == 'workflow_dispatch', 'manual dispatch required')
    remaining = datetime.fromisoformat(args.deadline.replace('Z','+00:00')).timestamp() - time.time()
    require(0 < remaining <= 3600, 'original bounded case deadline required')
    deadline = time.monotonic() + remaining
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=True)
    guard, resources, primitives, seals = source_modules(Path(args.inference_root).resolve(), os.environ['MOSS_ACCEPTANCE_INFERENCE_PIN'])
    import msvcrt
    public = Path(os.environ['PUBLIC'])
    require(public.is_absolute(), 'host-global PUBLIC namespace required')
    directory = public / 'SceneWorks' / 'gpu-locks'
    for candidate in (public, public/'SceneWorks', directory):
        require(not candidate.is_symlink() and not getattr(candidate,'is_junction',lambda:False)(), 'linked lock namespace refused')
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / (UUID + '.lock')
    require(not lock.is_symlink() and not getattr(lock,'is_junction',lambda:False)(), 'linked UUID lock refused')
    record = dict(selector=SELECTOR, deviceUuid=UUID, physicalIndex=1, activityProfile=PROFILE,
                  childGpuCapBytes=CAP, gpuFreeReserveBytes=RESERVE, visibility=visibility,
                  workflowSource=os.environ['MOSS_ACCEPTANCE_WORKFLOW_SOURCE'], inferencePin=os.environ['MOSS_ACCEPTANCE_INFERENCE_PIN'],
                  runId=os.environ['GITHUB_RUN_ID'], runAttempt=os.environ['GITHUB_RUN_ATTEMPT'],
                  sourceSeals=seals, state='claiming', leaseReleased=False, postExecutionVerified=False,
                  isolatedBenchmarkClaim=False, backgroundActivityIsDiagnostic=True)
    file = output / 'physical-claim.json'
    def save():
        temporary = output / 'physical-claim.json.tmp'
        temporary.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
        temporary.replace(file)
    claimed = False
    process_handle = None
    kernel = None
    with lock.open('a+b') as handle:
        info = os.fstat(handle.fileno())
        require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, 'unique regular lock required')
        if handle.seek(0,2) == 0:
            handle.write(b'0'); handle.flush()
        try:
            while not claimed:
                require(time.monotonic() < deadline, 'UUID lease deadline exceeded')
                handle.seek(0)
                try:
                    msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1); claimed=True
                except OSError:
                    time.sleep(min(.5,max(0,deadline-time.monotonic())))
            # Shared-headroom correctness requires fresh exact-card counters, not
            # a host-wide process census. Preserve the original case deadline
            # even when the sampling helper supplies its own five-second window.
            def selected_inventory(out=None, label='snapshot', probe_deadline=None):
                bounded_deadline = deadline if probe_deadline is None else min(deadline, probe_deadline)
                require(time.monotonic() < bounded_deadline, 'selected GPU metadata deadline exhausted')
                return guard.inventory(out,label,bounded_deadline,selected_uuid=UUID)
            devices, processes = selected_inventory(output,'bounded-claimed-device',deadline)
            selected = [d for d in devices if d['uuid']==UUID and d['index']==1]
            require(len(selected)==1, 'physical1 UUID changed; no fallback')
            policy = SimpleNamespace(child_gpu_cap_bytes=CAP,gpu_free_reserve_bytes=RESERVE)
            idle_check = lambda device, rows: selected_primary_metadata_valid(device) and resources.proof_cuda_idle(device,rows,policy,activity_profile=PROFILE)
            record['admissionDiagnostic'] = resources.cuda_claim_diagnostic(UUID,devices,processes,policy,activity_profile=PROFILE)
            require(idle_check(selected[0],processes), 'selected GPU lacks cap plus reserve headroom')
            identity_key = lambda uuid, rows: resources.proof_cuda_contexts(uuid,rows,activity_profile=PROFILE)
            device, baseline = guard.sample_baseline(selected[0],output,'bounded-admission',idle_check=idle_check,identity_key=identity_key,permit_bounded_growth=True,require_stable_contexts=False,inventory_reader=selected_inventory)
            binding = primitives.resolve_binding(SimpleNamespace(inventory=selected_inventory),output,deadline,dict(os.environ),cuda_ordinals='0')
            require(binding.get('mapping') == [{'cuda_ordinal':0,'uuid':UUID,'nvml_index':1}], 'actual CUDA ordinal0 does not resolve to physical1 leased UUID')
            record.update(state='admitted', ordinalBinding=binding, admissionBaseline=baseline)
            save(); print(json.dumps(record),flush=True)
            commands = queue.Queue()
            def read_commands():
                for line in sys.stdin:
                    commands.put(line)
                commands.put('')
            threading.Thread(target=read_commands,daemon=True).start()
            raw = commands.get(timeout=max(.001,deadline-time.monotonic()))
            first = json.loads(raw)
            # A preparation refusal may close a claim before creating a native PID.
            if first != {'release':True}:
                root = first.get('root',{})
                require(set(first)=={'root'} and root.get('scope')=='native-api-worker-root'
                        and type(root.get('pid')) is int and root['pid']>0
                        and Path(root.get('executable','')).is_absolute()
                        and len(root.get('binarySha256',''))==64, 'owned native root binding required')
                import ctypes.wintypes
                kernel = ctypes.WinDLL('kernel32',use_last_error=True)
                kernel.OpenProcess.argtypes = [ctypes.wintypes.DWORD,ctypes.wintypes.BOOL,ctypes.wintypes.DWORD]
                kernel.OpenProcess.restype = ctypes.wintypes.HANDLE
                kernel.WaitForSingleObject.argtypes = [ctypes.wintypes.HANDLE,ctypes.wintypes.DWORD]
                kernel.WaitForSingleObject.restype = ctypes.wintypes.DWORD
                kernel.CloseHandle.argtypes = [ctypes.wintypes.HANDLE]
                kernel.CloseHandle.restype = ctypes.wintypes.BOOL
                process_handle = kernel.OpenProcess(0x00100000,False,root['pid']) # SYNCHRONIZE only.
                require(bool(process_handle), 'cannot retain synchronization handle for owned native root')
                record['nativeRoot'] = root; save()
                raw = commands.get(timeout=max(.001,deadline+30-time.monotonic()))
                require(json.loads(raw)=={'release':True}, 'native supervisor did not authenticate release')
                require(kernel.WaitForSingleObject(process_handle,0)==0, 'native process still alive at release')
                record['nativeProcessExited']=True
            else:
                record['nativeProcessStarted']=False
            # Native ownership is settled above (and rechecked before unlocking below).
            # Shared-card release is not admission of another 80+2 GiB workload: fresh
            # exact-card primary metadata must remain valid, but free memory, utilization
            # and uncollected foreign contexts cannot prove or disprove owned-process exit.
            release_check = lambda current, rows: selected_primary_metadata_valid(current)
            record['releasePolicy'] = 'owned-native-exit-and-selected-primary-metadata'
            guard.verify_idle(device,output,'bounded-release',dict(baseline),idle_check=release_check,identity_key=identity_key,permit_bounded_growth=True,require_stable_contexts=False,inventory_reader=selected_inventory)
            record.update(state='released',postExecutionVerified=True)
        except BaseException as error:
            record.update(state='failed',error=str(error)); raise
        finally:
            # Never free the shared UUID lease while our authenticated native PID
            # is alive, including a parent crash or failed cooperative deadline.
            # This is a failed-run safety hold, never a success or force-kill.
            try:
                if process_handle:
                    while kernel.WaitForSingleObject(process_handle,1000) == 258:
                        record.update(state='failed',error=record.get('error','native root did not exit'))
                        save()
                    require(kernel.WaitForSingleObject(process_handle,0)==0, 'native handle wait failed; lease release cannot be certified')
                    record['nativeProcessExited']=True
                    require(kernel.CloseHandle(process_handle), 'owned native handle close failed')
                if claimed:
                    handle.seek(0); msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
                    record['leaseReleased']=True
            except BaseException as error:
                record.update(state='failed',error=str(error)); raise
            finally:
                save()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('inference-root','output','selector','deadline'):
        parser.add_argument('--'+name,required=True)
    main(parser.parse_args())
