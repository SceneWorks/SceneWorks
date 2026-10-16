#!/usr/bin/env python3
"""Offline, single-output MOSS reference runtime. No acquisition or remote app imports.

Requires a frozen local snapshot and reviewed pipeline source. Imports torch only
inside execution functions so protocol tests remain CPU/dependency independent.
"""
import hashlib
import importlib.util
from collections import Counter
import os
import sys
import time
from pathlib import Path

PIPELINE_SHA256 = '1dac1aca9feb1bfdfc727a401e4b67d61a505d3190706d1b7d2d784f88678b31'
REPLACEMENT_KEYS = {'time_embedding.0.weight','time_embedding.0.bias','time_embedding.2.weight','time_embedding.2.bias','shared_block_modulation.bias','video_head_modulation.bias'}
PROJECTIONS = ('q','k','v','o','ffn_up_gate','ffn_down')


def dmd_loop(z, q, codes, predict, steps=8, observer=None):
    """Exact fixed-noise recursion; injected operations permit tiny CPU fixtures."""
    if not 1 <= steps <= len(codes):
        raise ValueError('DMD steps must be a nonempty trained prefix')
    x0 = z * 0
    for index in range(steps):
        x = z + q(x0)
        velocity = predict(x, codes[index])
        x0 = z - velocity
        if observer:
            observer(index, x, velocity, x0)
    return x0


def import_pipeline(path):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != PIPELINE_SHA256:
        raise ValueError('unreviewed pipeline bytes')
    spec=importlib.util.spec_from_file_location('sc26322_verified_pipeline',path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module


def install_adapter(transformer, lora, replacements):
    """Author unmerged FP32 low-rank path; no BF16 merge is permitted.

    The FP32 residual is returned directly. That arithmetic choice is recorded and
    must be compared against full FP32 predictions; quality parity is not assumed.
    """
    import torch
    from torch import nn
    if set(replacements) != REPLACEMENT_KEYS:
        raise ValueError('replacement key coverage mismatch (both modulation biases required)')
    for value in replacements.values():
        if value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError('replacement must be finite FP32')
    for name in ('time_embedding','shared_block_modulation','video_head_modulation'):
        getattr(transformer,name).float()
    params=dict(transformer.named_parameters())
    with torch.no_grad():
        for key,value in replacements.items():
            if params[key].shape != value.shape:
                raise ValueError(f'replacement shape mismatch {key}')
            params[key].copy_(value)
    expected=set()
    class Unmerged(nn.Module):
        def __init__(self, backbone, down, up, alpha):
            super().__init__();self.backbone=backbone
            self.register_buffer('down',down);self.register_buffer('up',up)
            self.scale=alpha/down.shape[0]
        @property
        def weight(self):return self.backbone.weight
        def forward(self,x):
            from torch.nn.functional import linear
            return self.backbone(x.to(self.backbone.weight.dtype)).float()+self.scale*linear(linear(x.float(),self.down),self.up)
    for index,layer in enumerate(transformer.layers):
        for name in PROJECTIONS:
            prefix=f'diffusion_model.layers.{index}.proj.{name}'
            keys=[prefix+suffix for suffix in ('.lora_down.weight','.lora_up.weight','.alpha')]
            expected.update(keys)
            if any(k not in lora for k in keys):raise ValueError(f'missing LoRA module {prefix}')
            down,up,alpha=(lora[k] for k in keys)
            target=layer.proj[name]
            if down.dtype!=torch.float32 or up.dtype!=torch.float32 or down.shape!=(128,target.in_features) or up.shape!=(target.out_features,128):
                raise ValueError(f'invalid rank/dtype/shape {prefix}')
            if not all(torch.isfinite(v).all() for v in (down,up,alpha)) or alpha.numel()!=1 or alpha.item()!=128:
                raise ValueError(f'invalid LoRA alpha or values {prefix}')
            layer.proj[name]=Unmerged(target,down,up,float(alpha.item()))
    if set(lora)!=expected:raise ValueError('extra/unconsumed LoRA tensors')
    return transformer


def candidate_transformer(module,config,load_file):
    import torch
    settings=config['settings'];is_dmd=config['id']=='moss_dmd'
    root=settings['shared_snapshot'] if is_dmd else config['snapshot']
    model=module.MOSSImageTransformer.from_pretrained(root,subfolder='transformer',torch_dtype=torch.bfloat16,local_files_only=True,use_safetensors=True).eval()
    if is_dmd:
        if not settings.get('adapter_validation'):raise ValueError('unmerged DMD adapter and parity are mandatory')
        model=install_adapter(model,load_file(settings['lora_path']),load_file(settings['replacement_path']))
    return model


def candidate_latents(transformer,z,q,codes,embeds,mask,steps):
    import torch
    return dmd_loop(z,lambda x:torch.einsum('oc,bchw->bohw',q,x),codes,
        lambda x,code:transformer(x,code,embeds,mask).sample.float(),steps)


class Reference:
    def __init__(self, config, mode="candidate"):
        import torch
        from diffusers import AutoencoderKLFlux2
        from transformers import AutoModel, AutoTokenizer
        from safetensors.torch import load_file
        self.torch=torch;self.config=config;self.mode=mode
        if not torch.cuda.is_available():raise ValueError('CUDA reference required; CPU fallback forbidden')
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest')
        torch.cuda.reset_peak_memory_stats()
        self.first=True
        settings=config['settings'];m=import_pipeline(settings['pipeline_code'])
        common=dict(local_files_only=True,use_safetensors=True)
        root=config['snapshot'];shared=settings.get('shared_snapshot',root)
        tok=AutoTokenizer.from_pretrained(shared,subfolder='tokenizer',local_files_only=True,trust_remote_code=False)
        enc=AutoModel.from_pretrained(shared,subfolder='text_encoder',dtype=torch.bfloat16,trust_remote_code=False,**common).eval()
        is_dmd=config['id']=='moss_dmd'
        if mode not in ('candidate','parity') or (mode=='parity' and not is_dmd):
            raise ValueError('invalid reference execution arm')
        tr=candidate_transformer(m,config,load_file)
        vae=None if mode=='parity' else AutoencoderKLFlux2.from_pretrained(shared,subfolder='vae',torch_dtype=torch.float32,**common).eval()
        sch=m.MOSSImageScheduler.from_pretrained(shared,subfolder='scheduler',local_files_only=True)
        self.pipe=m.MOSSImagePipeline(tr,sch,enc,tok,vae).to('cuda')
        # Full FP32 weights are never loaded in the measured candidate process.
        self.oracle=None
        if mode=='parity':
            self.oracle=m.MOSSImageTransformer.from_pretrained(root,subfolder='transformer',torch_dtype=torch.float32,**common).eval().to('cuda')
        self.q=None;self.codes=None
        if is_dmd:
            ds=load_file(str(Path(root)/'dataspace.safetensors'))
            if set(ds)!={'latent_mix.Q','step_codes'}:raise ValueError('unexpected dataspace keys')
            self.q=ds['latent_mix.Q'].to('cuda');self.codes=ds['step_codes'].to('cuda')
            if self.q.shape!=(128,128) or self.codes.shape!=(8,) or any(v.dtype!=torch.float32 or not torch.isfinite(v).all() for v in (self.q,self.codes)):
                raise ValueError('DMD dataspace shape/dtype/finite mismatch')
        self.initial_allocated=torch.cuda.max_memory_allocated();self.initial_reserved=torch.cuda.max_memory_reserved()

    def sync(self):self.torch.cuda.synchronize()

    def render(self,cell,output):
        torch=self.torch;pipe=self.pipe;config=self.config
        if self.mode!='candidate' or self.oracle is not None:raise ValueError('measured generation forbids oracle residency')
        self.sync();torch.cuda.reset_peak_memory_stats()
        phases=dict(encode_seconds=0.0,denoise_seconds=0.0,decode_seconds=0.0)
        def timed(name,fn,*args,**kwargs):
            self.sync();start=time.monotonic();value=fn(*args,**kwargs);self.sync();phases[name]+=time.monotonic()-start;return value
        gen=torch.Generator('cpu').manual_seed(cell['seed'])
        z_cpu=torch.randn((1,128,cell['height']//16,cell['width']//16),generator=gen,dtype=torch.float32)
        noise_hash=hashlib.sha256(z_cpu.numpy().tobytes()).hexdigest();z=z_cpu.to('cuda')
        start=time.monotonic()
        with torch.inference_mode(),torch.autocast('cuda',enabled=False):
            embeds,mask=timed('encode_seconds',pipe.encode_prompt,cell['prompt'])
            if config['id']=='moss_dmd':
                latents=timed('denoise_seconds',candidate_latents,pipe.transformer,z,self.q,self.codes,embeds,mask,config['steps'])
                sampler=dict(name='published_dmd_fixed_noise',Q_sha256=hashlib.sha256(self.q.cpu().numpy().tobytes()).hexdigest(),step_codes=self.codes.cpu().tolist(),steps=config['steps'],cfg=False,autocast=False,execution_arm='sft_bf16_unmerged_fp32_adapter',oracle_resident=False,adapter_arithmetic='bf16 backbone + unmerged fp32 low rank residual; fp32 time and both modulation modules')

            else:
                negatives,negative_mask=timed('encode_seconds',pipe.encode_prompt,'')
                latents=timed('denoise_seconds',pipe,prompt_embeds=embeds,prompt_attention_mask=mask,negative_prompt_embeds=negatives,negative_prompt_attention_mask=negative_mask,
                    height=cell['height'],width=cell['width'],num_inference_steps=config['steps'],guidance_scale=config['guidance'],latents=z,uncond_skip_layers=[22],output_type='latent').images
                sampler=dict(name='published_training_grid_euler',timesteps=pipe.scheduler.timesteps.cpu().tolist(),sigmas=pipe.scheduler.sigmas.cpu().tolist(),uncond_skip_layers=[22],shift=pipe.scheduler.config.shift,autocast=False)
            def decode():
                vae=pipe.vae;bn=vae.bn
                packed=latents.float()*(bn.running_var+vae.config.batch_norm_eps).sqrt().view(1,-1,1,1)+bn.running_mean.view(1,-1,1,1)
                b,c,h,w=packed.shape
                packed=packed.reshape(b,c//4,2,2,h,w).permute(0,1,4,2,5,3).reshape(b,c//4,2*h,2*w)
                pixels=vae.decode(packed,return_dict=False)[0]
                if not torch.isfinite(pixels).all():raise ValueError('nonfinite decoded pixels')
                pipe.image_processor.postprocess(pixels.clamp(-1,1),output_type='pil')[0].save(output)
            timed('decode_seconds',decode)
        self.sync()
        import importlib.metadata
        allocated=max(torch.cuda.max_memory_allocated(),self.initial_allocated if self.first else 0)
        reserved=max(torch.cuda.max_memory_reserved(),self.initial_reserved if self.first else 0)
        self.first=False
        return dict(phases, generation_seconds=time.monotonic()-start,noise_sha256=noise_hash,pixels_finite=True,
            peak_gpu_allocated_bytes=allocated,peak_gpu_reserved_bytes=reserved,
            component_dtypes=dict(transformer=str(pipe.transformer.dtype),encoder=str(pipe.text_encoder.dtype),vae=str(pipe.vae.dtype),transformer_parameter_numel_by_dtype=dict(Counter({str(dtype):sum(p.numel() for p in pipe.transformer.parameters() if p.dtype==dtype) for dtype in {p.dtype for p in pipe.transformer.parameters()}}))),sampler=sampler,
            device=dict(visible_uuid=os.environ['CUDA_VISIBLE_DEVICES'],name=torch.cuda.get_device_name(0),logical_index=torch.cuda.current_device(),allow_tf32_matmul=torch.backends.cuda.matmul.allow_tf32,allow_tf32_cudnn=torch.backends.cudnn.allow_tf32,float32_matmul_precision=torch.get_float32_matmul_precision()),
            runtime_versions={name:importlib.metadata.version(name) for name in ('torch','transformers','diffusers','safetensors','accelerate','Pillow')})


    def predictions(self,cell):
        """No decode/output image: compare both arms on identical candidate-path inputs."""
        if self.mode!='parity' or self.oracle is None:raise ValueError('parity requires separate oracle process')
        torch=self.torch;pipe=self.pipe
        tensor_hash=lambda value:hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        gen=torch.Generator('cpu').manual_seed(cell['seed'])
        z=torch.randn((1,128,cell['height']//16,cell['width']//16),generator=gen,dtype=torch.float32).to('cuda')
        steps=[];start=time.monotonic()
        with torch.inference_mode(),torch.autocast('cuda',enabled=False):
            embeds,mask=pipe.encode_prompt(cell['prompt'])
            def predict(x,code):
                candidate=pipe.transformer(x,code,embeds,mask).sample.float()
                oracle=self.oracle(x,code,embeds,mask).sample.float()
                difference=(candidate-oracle).abs()
                if not torch.isfinite(candidate).all() or not torch.isfinite(oracle).all():raise ValueError('nonfinite parity prediction')
                steps.append(dict(code=float(code),input_sha256=tensor_hash(x),candidate_sha256=tensor_hash(candidate),oracle_sha256=tensor_hash(oracle),max_abs=float(difference.max()),mean_abs=float(difference.mean()),reference_rms=float(oracle.square().mean().sqrt()),candidate_finite=True))
                return candidate
            dmd_loop(z,lambda x:torch.einsum('oc,bchw->bohw',self.q,x),self.codes,predict,8)
        self.sync()
        return dict(steps=steps,forward_count=16,noise_sha256=tensor_hash(z),Q_sha256=tensor_hash(self.q),step_codes=self.codes.cpu().tolist(),context_sha256=tensor_hash(embeds),mask_sha256=tensor_hash(mask),candidate_arm='sft_bf16_unmerged_fp32_adapter',oracle_arm='published_full_fp32_transformer',oracle_autocast=False,generated_images=0,elapsed_seconds=time.monotonic()-start)
