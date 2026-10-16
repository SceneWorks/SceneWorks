"""Use the existing native product route; packed snapshots never enter Diffusers."""
import json
from pathlib import PurePosixPath
import time
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen


def request_body(config, cell, project_id):
    return dict(projectId=project_id,mode='text_to_image',model=config['id'],prompt=cell['prompt'],
        count=1,width=cell['width'],height=cell['height'],seed=cell['seed'],
        advanced=dict(mlxQuantize={'bf16':0,'q8':8,'q4':4}[config['precision']],
            steps=config['steps'],guidanceScale=config['guidance'],enhancePrompt=False,mlxQuantizeExplicit=True))


class Native:
    def __init__(self,config):
        from native_services import services
        self.config=config;settings=config['settings']
        self.services_context=services(settings)
        self.context=self.services_context.__enter__()
        try:self._connect()
        except BaseException:
            self.close()
            raise

    def _connect(self):
        config=self.config;settings=config['settings']
        self.api=self.context['api'].rstrip('/')
        parsed=urlparse(self.api)
        if parsed.scheme!='http' or parsed.hostname not in ('127.0.0.1','localhost'):
            raise ValueError('owned loopback API required for process memory attribution')
        workers=self.call('GET','/api/v1/workers')
        self.worker=next((w for w in workers if w['id']==settings['worker_id']),None)
        if not self.worker or self.worker['status']!='idle' or self.worker.get('currentJobId') or self.worker.get('gpuId') in ('cpu','mlx',''):
            raise ValueError('expected exclusive idle Candle worker not registered')
        import psutil
        from evaluate import file_hash
        worker=psutil.Process(self.context['worker_pid'])
        if file_hash(worker.exe())!=settings['binary_sha256']:
            raise ValueError('native PID does not execute frozen binary')
        self.route_events=[]
        self.project=self.call('POST','/api/v1/projects',{'name':'sc-26322 frozen reference comparison'})

    def close(self):
        self.services_context.__exit__(None,None,None)

    def call(self,method,path,body=None,raw=False):
        req=Request(self.api+path,data=json.dumps(body).encode() if body is not None else None,method=method,headers={'Content-Type':'application/json'})
        with urlopen(req,timeout=30) as response:data=response.read()
        return data if raw else json.loads(data)

    def render(self,cell,output):
        config=self.config;start=time.monotonic();job=None
        body=request_body(config,cell,self.project['id'])
        accepted=self.call('POST','/api/v1/image/jobs',body)
        job_id=accepted['id'];deadline=start+config['settings']['job_timeout_seconds']
        try:
            while time.monotonic()<deadline:
                job=self.call('GET',f'/api/v1/jobs/{job_id}')
                if job['status'] in ('failed','canceled'):raise ValueError(f'native job {job_id} {job["status"]}: {job.get("error")}')
                if job['status']=='completed' and job.get('result',{}).get('assets'):
                    metrics=self.call('GET',f'/api/v1/jobs/{job_id}/metrics')
                    # Completion can precede the final coalesced metrics write.
                    if metrics and all(k in metrics for k in ('model','quantLabel','steps','peakMemoryBytes','totalMs','loadMs','sampleMs','decodeMs')):break
                time.sleep(.5)
            else:raise TimeoutError(f'native job {job_id} timed out')
        except BaseException:
            # Cancel only this harness-created job; never terminate a shared worker.
            self.call('POST',f'/api/v1/jobs/{job_id}/cancel',{})
            raise
        expected=dict(model=config['id'],quantLabel=config['precision'],steps=config['steps'],seed=cell['seed'],width=cell['width'],height=cell['height'],imageCount=1,backend='cuda')
        if job.get('workerId')!=config['settings']['worker_id'] or any(metrics.get(k)!=v for k,v in expected.items()):
            raise ValueError('native actual worker/model/tier/settings mismatch')
        assets=job['result']['assets']
        if len(assets)!=1:raise ValueError('native output count mismatch')
        recipe=assets[0]['recipe'];raw=recipe['rawAdapterSettings']
        if recipe.get('model')!=config['id'] or raw.get('realModelInference') is not True or not raw.get('repo') or raw.get('numInferenceSteps')!=config['steps'] or raw.get('mlxQuantize')!={'bf16':None,'q8':8,'q4':4}[config['precision']]:
            raise ValueError('native actual asset route/repo/tier mismatch')
        path=assets[0]['file']['path'];parts=PurePosixPath(path)
        if parts.is_absolute() or '..' in parts.parts or '\\' in path:raise ValueError('unsafe native asset path')
        from pathlib import Path
        events=[]
        for line in Path(self.context['worker_log']).read_text(encoding='utf-8',errors='replace').splitlines():
            try:event=json.loads(line)
            except json.JSONDecodeError:continue
            if event.get('jobId')==job_id:events.append(event)
        routes=[event for event in events if event.get('event')=='image_pipeline_load_complete']
        if routes:self.route_events=routes
        if not self.route_events or not all(e.get('engine')==config['id'] for e in self.route_events):
            raise ValueError('missing or mismatched actual native engine load event')
        if not any(e.get('event')=='image_inference_complete' and e.get('backend')=='candle' for e in events):
            raise ValueError('missing completed native Candle inference event')
        data=self.call('GET',f'/api/v1/projects/{self.project["id"]}/files/{quote(path,safe="/")}',raw=True)
        from PIL import Image
        import io
        image=Image.open(io.BytesIO(data));image.load()
        if image.format!='PNG' or image.size!=(cell['width'],cell['height']):raise ValueError('native invalid PNG/canvas')
        output.write_bytes(data)
        return dict(generation_seconds=time.monotonic()-start,load_seconds=metrics['loadMs']/1000,
            encode_seconds=None,denoise_seconds=metrics['sampleMs']/1000,decode_seconds=metrics['decodeMs']/1000,
            peak_gpu_bytes=metrics['peakMemoryBytes'],gpu_memory_method='worker nvidia-smi sampled device memory, 1s cadence; not allocator peak',
            component_dtypes={'native_tier':metrics['quantLabel'],'component_dtypes':'not exposed by native metrics'},
            pixels_finite=True,sampler={'name':metrics.get('sampler'),'scheduler':metrics.get('scheduler'),'native_metrics':metrics},
            runtime_versions=config['settings']['runtime_versions'],native_proof={'submitted_request':body,'job':job,'metrics':metrics,'asset':assets[0],'route_events':self.route_events,'job_events':events,'runtime_context':self.context},
            limitations=['native loadMs includes encoding until first denoise step','device-wide sampled GPU peak is not PyTorch allocator peak','native execution representation is missing if no runtime load receipt; do not infer from tier'])
