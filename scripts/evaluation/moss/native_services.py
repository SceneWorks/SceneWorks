#!/usr/bin/env python3
"""Start task-owned offline API/worker processes; never builds or acquires weights.

Keep this command running while freezing and executing. Services end on Ctrl-C.
Use one distinct port/data directory per native configuration. The generated
settings are inputs to the freeze, not evidence that generation has happened.
"""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.request import urlopen
from evaluate import file_hash, write


@contextmanager
def services(settings):
    class Args:pass
    a=Args();a.state=str(Path(settings['state_root'])/f'owned-{os.getpid()}');a.binary=settings['binary'];a.hub=settings['hub_cache'];a.gpu=settings['device_id'];a.port=settings['port'];a.worker_id=settings['worker_id']
    state=Path(a.state).resolve();binary=Path(a.binary).resolve();hub=Path(a.hub).resolve()
    if state.exists() or not binary.is_file() or not hub.is_dir():raise ValueError('fresh state, existing binary and prepared hub required')
    if file_hash(binary)!=settings['binary_sha256']:raise ValueError('native binary hash mismatch')
    state.mkdir(parents=True);env={k:v for k,v in os.environ.items() if not k.startswith('SCENEWORKS_') and not any(word in k.upper() for word in ('TOKEN','SECRET','PASSWORD','CREDENTIAL','API_KEY'))}
    url=f'http://127.0.0.1:{a.port}'
    env.update(SCENEWORKS_DATA_DIR=str(state/'data'),SCENEWORKS_CONFIG_DIR=str(state/'config'),SCENEWORKS_JOBS_DB_PATH=str(state/'data'/'cache'/'jobs.db'),SCENEWORKS_CREDENTIALS_DIR=str(state/'credentials'),SCENEWORKS_API_HOST='127.0.0.1',SCENEWORKS_API_PORT=str(a.port),SCENEWORKS_API_URL=url,SCENEWORKS_WORKER_ID=a.worker_id,SCENEWORKS_WORKER_CHILD='1',SCENEWORKS_GPU_ID='0',SCENEWORKS_POLL_SECONDS='1',SCENEWORKS_HEARTBEAT_SECONDS='2',SCENEWORKS_BACKEND_MLX_ENABLED='false',SCENEWORKS_BACKEND_CANDLE_ENABLED='true',SCENEWORKS_CANDLE_REQUIRED='1',SCENEWORKS_CANDLE_UNSUPPORTED_MODE='enforce',SCENEWORKS_TRUST_LOOPBACK='true',SCENEWORKS_UTILITY_JOBS='0',CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=a.gpu,HF_HOME=str(hub.parent),HF_HUB_CACHE=str(hub),HUGGINGFACE_HUB_CACHE=str(hub),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    children=[];streams=[]
    try:
        for name,extra in (('api',{}),('worker',{'SCENEWORKS_WORKER_ONLY':'1'})):
            stream=(state/f'{name}.log').open('wb');streams.append(stream)
            children.append(subprocess.Popen([str(binary)],env=dict(env,**extra),stdout=stream,stderr=subprocess.STDOUT))
        deadline=time.monotonic()+180
        while time.monotonic()<deadline:
            if any(c.poll() is not None for c in children):raise RuntimeError('owned service exited; inspect logs')
            try:
                with urlopen(url+'/api/v1/workers',timeout=2) as response:workers=json.load(response)
                if any(w['id']==a.worker_id and w['status']=='idle' for w in workers):break
            except OSError:pass
            time.sleep(1)
        else:raise TimeoutError('worker registration timed out; no jobs submitted')
        context=dict(api=url,binary=str(binary),binary_sha256=file_hash(binary),worker_id=a.worker_id,worker_pid=children[1].pid,worker_log=str(state/'worker.log'),owned_state=str(state),hub_cache=str(hub),runtime_versions=settings['runtime_versions'])
        yield context
    finally:
        for child in reversed(children):
            if child.poll() is None:child.terminate()
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
        for stream in streams:stream.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('settings');p.add_argument('output')
    a=p.parse_args()
    with services(json.loads(Path(a.settings).read_text())) as context:
        write(a.output,context)
        print('Owned idle services started; Ctrl-C closes only this process tree.',flush=True)
        while True:time.sleep(1)


if __name__=='__main__':main()
