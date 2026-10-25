"""Three fixed MPS quality comparisons using the unchanged frozen upstream learned source."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ExactCaseNoise:
    def __init__(self, path: Path, expected_sha: str, shape: tuple[int, ...]):
        import torch
        payload = path.read_bytes()
        if len(payload) != 786432 or hashlib.sha256(payload).hexdigest() != expected_sha:
            raise ValueError('Frozen case initial noise identity mismatch')
        self.shape = shape
        self.tensor = torch.frombuffer(bytearray(payload), dtype=torch.float32).clone().reshape(shape)
        self.calls = 0

    def randn(self, *args, **kwargs):
        import torch
        if self.calls or len(args) != 1 or tuple(args[0]) != self.shape:
            raise ValueError('Unexpected or repeated initial-noise allocation')
        generator = kwargs.get('generator')
        if (set(kwargs) != {'generator', 'device', 'dtype'} or kwargs['dtype'] is not torch.float32
                or torch.device(kwargs['device']) != torch.device('cpu')
                or not isinstance(generator, torch.Generator) or generator.device != torch.device('cpu')
                or generator.initial_seed() != 173):
            raise ValueError('Expected exactly one CPU f32 seed173 initial noise allocation')
        self.calls += 1
        return self.tensor.clone()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior-payload', type=Path, required=True)
    parser.add_argument('--case-payload', type=Path, required=True)
    parser.add_argument('--case', choices=['base-four', 'sft-single', 'sft-four'], required=True)
    parser.add_argument('--snapshot-view', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['mps'], required=True)
    parser.add_argument('--execute-reviewed', action='store_true')
    args = parser.parse_args()
    if not args.execute_reviewed:
        parser.error('Reviewed fixed case execution is required')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUDA_VISIBLE_DEVICES='', PYTORCH_ENABLE_MPS_FALLBACK='0')
    sys.dont_write_bytecode = True
    prior = args.prior_payload.resolve(strict=True)
    spec = importlib.util.spec_from_file_location('frozen_reference_diagnostic', prior / 'reference_diagnostic.py')
    if not spec or not spec.loader:
        raise ValueError('Frozen reference diagnostic unavailable')
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    if frozen.verify_frozen_sources() != 24:
        raise ValueError('Frozen learned source closure changed')
    cases = json.loads((args.case_payload / 'cases.json').read_text())
    case = next(item for item in cases['cases'] if item['id'] == args.case)
    refs = []
    for item in case['refs']:
        path = (args.case_payload / item['file']).resolve(strict=True)
        if digest(path) != item['sha256']:
            raise ValueError('Ordered frozen reference bytes changed')
        refs.append(path)
    noise_info = case['noise']
    noise = ExactCaseNoise(args.case_payload / noise_info['file'], noise_info['sha256'], tuple(noise_info['shape']))
    config = json.loads((args.case_payload / 'reference-config.plan.json').read_text())
    config.update(snapshot_dir=str(args.snapshot_view.resolve(strict=True)), device=args.device, encoder_device=args.device)
    if (config['text_quantization'], config['dit_quantization'], config['attention_backend'],
            config['training_shift'], config['inference_shift']) != ('none', 'none', 'standard', 1.0, 5.0):
        raise ValueError('Frozen upstream model configuration changed')
    if args.output.exists() or args.output.with_suffix('.diagnostic.json').exists():
        raise ValueError('Refusing to overwrite an upstream quality result')
    sys.path.insert(0, str(prior / 'sources/Moss-Image-Edit'))
    import infer
    from moss_edit.model import MossEditModel
    from moss_edit.sampling import sample_latents
    scoped_sampler = frozen.sampler_with_local_noise(sample_latents, noise)

    def sample_once(*positional, **keywords):
        result = scoped_sampler(*positional, **keywords)
        if noise.calls != 1:
            raise RuntimeError('Frozen sampler did not consume supplied noise exactly once')
        return result

    adapter = Path('/Volumes/Models/sceneworks-terminal-sc-26752/moss-upstream-reference-r15-retained-adapter')
    command = ['--adapter', str(adapter), '--references', *map(str, refs), '--prompt', case['prompt'],
               '--negative-prompt', case['negativePrompt'], '--output', str(args.output),
               '--height', str(case['height']), '--width', str(case['width']), '--steps', str(case['steps']),
               '--cfg', str(case['cfg']), '--seed', str(case['seed']), '--device', args.device]
    frozen_args = infer.build_parser().parse_args(command)
    infer.validate_arguments(frozen_args, config)
    original_time, original_sampler = MossEditModel._time, infer.sample_latents
    try:
        MossEditModel._time, infer.sample_latents = frozen.placed_time, sample_once
        result = infer.run(frozen_args, config)
        if noise.calls != 1:
            raise RuntimeError('Frozen sampler did not consume supplied noise exactly once')
    finally:
        MossEditModel._time, infer.sample_latents = original_time, original_sampler
    receipt = {'caseId': args.case, 'variant': case['variant'], 'result': result,
               'noiseSha256': noise_info['sha256'], 'noiseShape': noise_info['shape'],
               'noiseAllocations': noise.calls, 'wrapperSha256': digest(Path(__file__)),
               'placementAdaptation': 'same frozen CPU F32/F64 sinusoid and learned-dtype transfer as Base/single',
               'noQuantization': True, 'frozenFilesVerifiedAfterRun': frozen.verify_frozen_sources()}
    args.output.with_suffix('.diagnostic.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
