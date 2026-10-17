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


KLEIN_TRANSFORMER_CONFIG = '1ef42c2f91ff0060b147f7ff6790b5dc6f2e4c14d31dbeae3005ff339cea7922'
KLEIN_TEXT_ENCODER_CONFIG = '4b8a90efaacbc0eaa1f3f3a2006335bbbbad2683fbd0688ada2c436e71342134'
FROZEN_SOURCE = 'ba2954f47bfd75a3186002e3cef859c4e4e1d37d'
FROZEN_INFERENCE = '91ffb4a819aa2990c652db2d881ba0b9b10126ce'


def klein_tier_proof(config, job_id, metrics, recipe, worker_log):
    """Independent resolved-route proof for the frozen Windows metrics defect.

    A requested tier is insufficient. Bind the closed artifact inventory/configs,
    frozen binary/source, job-correlated resolved strategy and Candle completion.
    Raw metrics remain unchanged; this is indirect source/bytes proof, not an
    executed tensor dtype receipt.
    """
    from pathlib import Path
    from evaluate import file_hash, read
    def need(ok, message):
        if not ok: raise ValueError(message)
    settings=config['settings'];versions=settings['runtime_versions']
    need(config['id']=='flux2_klein_9b' and config['precision']=='q4' and
         versions['sceneworks_source']==FROZEN_SOURCE and versions['inference']==FROZEN_INFERENCE,
         'Klein defect proof requires exact frozen source/pin/q4')
    snapshot=Path(config['snapshot'])
    need(config['revision']=='1902693279fcfb828919370dfac2b8922d99499a', 'Klein proof revision mismatch')
    inventory={f['path']:f['sha256'] for f in config['files']}
    configs={}
    for name,sha in (('transformer',KLEIN_TRANSFORMER_CONFIG),('text_encoder',KLEIN_TEXT_ENCODER_CONFIG)):
        path=snapshot/'q4'/name/'config.json'
        need(inventory.get(str(path))==sha and file_hash(path)==sha,'Klein actual artifact config/hash mismatch')
        configs[name]=read(path)
    need(configs['transformer'].get('quantization',{}).get('bits')==4 and not configs['text_encoder'].get('quantization') and configs['text_encoder'].get('dtype')=='bfloat16','Klein packed DiT/dense TE contract mismatch')
    need(inventory.get(settings['binary'])==settings['binary_sha256'] and file_hash(settings['binary'])==settings['binary_sha256'],'Klein frozen binary proof mismatch')
    events=[]
    for line in Path(worker_log).read_text(encoding='utf-8',errors='strict').splitlines():
        try: event=json.loads(line)
        except json.JSONDecodeError: continue
        need(not ('jobId' in event and 'job_id' in event) or event['jobId']==event['job_id'],'conflicting native event job identities')
        if event.get('jobId',event.get('job_id'))==job_id:events.append(event)
    strategy=[e for e in events if e.get('event')=='image_memory_strategy_selected' and e.get('job_id')==job_id]
    need(len(strategy)==1 and strategy[0].get('model')==config['id'] and strategy[0].get('route')==config['id'] and strategy[0].get('actual_tier')=='q4','missing exact-job resolved Klein q4 event')
    need(any(e.get('event')=='image_inference_complete' and e.get('backend')=='candle' for e in events),'missing exact-job completed Candle inference')
    need(recipe.get('model')==config['id'] and recipe['rawAdapterSettings'].get('mlxQuantize')==4 and recipe['rawAdapterSettings'].get('realModelInference') is True,'Klein actual recipe tier mismatch')
    need((metrics.get('quantLabel'),metrics.get('quantBits'))==('bf16',None),'unexpected frozen Klein metrics defect signature')
    return dict(resolved_tier='q4',reported_quant_label=metrics['quantLabel'],reported_quant_bits=metrics.get('quantBits'),
        transformer_config_sha256=KLEIN_TRANSFORMER_CONFIG,text_encoder_config_sha256=KLEIN_TEXT_ENCODER_CONFIG,
        binary_sha256=settings['binary_sha256'],revision=config['revision'],source=FROZEN_SOURCE,inference=FROZEN_INFERENCE,
        strategy_event=strategy[0],job_id=job_id,method='closed staged bytes + pinned packed loader + actual job/route/backend; runtime component dtypes not exposed')


Z_TRANSFORMER_CONFIG='3a634d2dd4c84aa9ae511ad88a61fa2a427c05f61940fda45a15aa97cb20a7c6'
Z_TEXT_ENCODER_CONFIG='6237fccdd922663bd3d8bce89bc390ca073ca22c10e8b3e1fe81a58ccf2169db'


def z_tier_proof(config,job_id,metrics,recipe,worker_log):
    """Frozen packed Z recipe-null defect; require actual bytes/route, never request."""
    from pathlib import Path
    from evaluate import file_hash,read
    def need(ok,message):
        if not ok:raise ValueError(message)
    settings=config['settings'];versions=settings['runtime_versions']
    need(config['id']=='z_image_turbo' and config['precision']=='q4' and config['revision']=='bb2bc9893b3c49ae96c813350775f791a2e8bc80' and versions['sceneworks_source']==FROZEN_SOURCE and versions['inference']==FROZEN_INFERENCE,'Z packed proof requires exact frozen source/pin/artifact')
    inventory={f['path']:f['sha256'] for f in config['files']}
    for part,sha in (('transformer',Z_TRANSFORMER_CONFIG),('text_encoder',Z_TEXT_ENCODER_CONFIG)):
        path=Path(config['snapshot'])/'q4'/part/'config.json'
        need(inventory.get(str(path))==sha and file_hash(path)==sha and read(path).get('quantization',{}).get('bits')==4,'Z packed component config/hash mismatch')
    need(inventory.get(settings['binary'])==settings['binary_sha256'] and file_hash(settings['binary'])==settings['binary_sha256'],'Z frozen binary mismatch')
    events=[]
    for line in Path(worker_log).read_text(encoding='utf-8').splitlines():
        try:row=json.loads(line)
        except json.JSONDecodeError:continue
        need(not ('jobId' in row and 'job_id' in row) or row['jobId']==row['job_id'],'conflicting native event job identities')
        if row.get('jobId',row.get('job_id'))==job_id:events.append(row)
    strategy=[e for e in events if e.get('event')=='image_memory_strategy_selected' and e.get('job_id')==job_id]
    need(len(strategy)==1 and strategy[0].get('model')==config['id'] and strategy[0].get('route')==config['id'] and strategy[0].get('actual_tier')=='q4','missing exact-job resolved Z q4 event')
    need(any(e.get('event')=='image_inference_complete' and e.get('backend')=='candle' for e in events),'missing completed exact-job Candle Z inference')
    need(metrics.get('quantLabel')=='q4' and metrics.get('quantBits')==4 and recipe.get('model')==config['id'] and recipe['rawAdapterSettings'].get('mlxQuantize') is None and recipe['rawAdapterSettings'].get('realModelInference') is True,'unexpected frozen Z recipe-null defect signature')
    return dict(resolved_tier='q4',reported_raw_mlx_quantize=None,transformer_config_sha256=Z_TRANSFORMER_CONFIG,text_encoder_config_sha256=Z_TEXT_ENCODER_CONFIG,binary_sha256=settings['binary_sha256'],revision=config['revision'],source=FROZEN_SOURCE,inference=FROZEN_INFERENCE,job_id=job_id,strategy_event=strategy[0],method='closed staged packed configs + pinned loader + exact job route/backend; runtime component dtypes not exposed')


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
                if job['status']=='failed':raise RuntimeError(f'native job {job_id} failed: {job.get("error")}')
                if job['status']=='canceled':raise ValueError(f'owned native job {job_id} was canceled; review execution integrity')
                if job['status']=='completed' and job.get('result',{}).get('assets'):
                    metrics=self.call('GET',f'/api/v1/jobs/{job_id}/metrics')
                    # Completion can precede the final coalesced metrics write.
                    if metrics and all(k in metrics for k in ('model','quantLabel','steps','peakMemoryBytes','totalMs','loadMs','sampleMs','decodeMs')):break
                time.sleep(.5)
            else:raise TimeoutError(f'native job {job_id} timed out')
        except BaseException:
            # Cancel only this harness-created job; never terminate a shared worker.
            if job is None or job['status'] not in ('failed','canceled','completed'):
                self.call('POST',f'/api/v1/jobs/{job_id}/cancel',{})
            raise
        expected=dict(model=config['id'],quantLabel=config['precision'],steps=config['steps'],seed=cell['seed'],width=cell['width'],height=cell['height'],imageCount=1,backend='cuda')
        tier_proof=None
        if config['id']=='flux2_klein_9b':
            tier_proof=klein_tier_proof(config,job_id,metrics,job['result']['assets'][0]['recipe'],self.context['worker_log'])
            expected.pop('quantLabel')
        if config['id']=='z_image_turbo':
            tier_proof=z_tier_proof(config,job_id,metrics,job['result']['assets'][0]['recipe'],self.context['worker_log'])
        if job.get('workerId')!=config['settings']['worker_id'] or any(metrics.get(k)!=v for k,v in expected.items()):
            raise ValueError('native actual worker/model/tier/settings mismatch')
        assets=job['result']['assets']
        if len(assets)!=1:raise ValueError('native output count mismatch')
        recipe=assets[0]['recipe'];raw=recipe['rawAdapterSettings']
        if recipe.get('model')!=config['id'] or raw.get('realModelInference') is not True or not raw.get('repo') or raw.get('numInferenceSteps')!=config['steps'] or raw.get('mlxQuantize')!=(None if config['id']=='z_image_turbo' else {'bf16':None,'q8':8,'q4':4}[config['precision']]):
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
        return dict(device=dict(physical_gpu_index=int(self.context['physical_gpu_index']),visible_uuid=self.context['gpu_uuid']),generation_seconds=time.monotonic()-start,load_seconds=metrics['loadMs']/1000,
            encode_seconds=None,denoise_seconds=metrics['sampleMs']/1000,decode_seconds=metrics['decodeMs']/1000,
            peak_gpu_bytes=metrics['peakMemoryBytes'],gpu_memory_method='worker nvidia-smi sampled device memory, 1s cadence; not allocator peak',
            component_dtypes={'native_tier':metrics['quantLabel'],'component_dtypes':'not exposed by native metrics'},
            pixels_finite=True,sampler={'name':metrics.get('sampler'),'scheduler':metrics.get('scheduler'),'native_metrics':metrics},
            runtime_versions=config['settings']['runtime_versions'],native_proof={'submitted_request':body,'job':job,'metrics':metrics,'asset':assets[0],'route_events':self.route_events,'job_events':events,'runtime_context':self.context,'resolved_tier_proof':tier_proof},
            limitations=['native loadMs includes encoding until first denoise step','device-wide sampled GPU peak is not PyTorch allocator peak','native execution representation is missing if no runtime load receipt; do not infer from tier'])
