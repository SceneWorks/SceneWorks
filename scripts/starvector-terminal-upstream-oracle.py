#!/usr/bin/env python3
"""Validation-only, offline execution of the audited upstream StarVector implementation.

No remote Python is executed. Constructor adapters replace downloads with local
config/tokenizer initialization; upstream vision, projection, transformer and
SVG generation remain in use. A parent process enforces RSS/runtime bounds.

Provision a separate Python 3.11/3.12 environment using required_packages in
release/starvector-terminal-upstream-lock-v1.json; do not install this in the
product/metrics interpreter. Materialize an exact git checkout of the locked
upstream source and the existing native model snapshots. No git/HF download is
performed by this command. --components-root must contain components.json:
  {"starcoder1": {"repository": "bigcode/starcoderbase-1b",
    "revision": "182f0165fdf8da9c9935901eec65c94337f01c11",
    "config_path": "starcoder1/config.json", "config_sha256": "<sha256>"}, ...}
The analogous starcoder2 and siglip entries must match the public revisions and
config digests in the source lock. The 1B base config is gated: provision only
an authorized exact revision, never infer unspecified architecture defaults.
Use the already configured official HF client authentication to obtain it after
access is granted, without embedding credentials in command arguments:
  hf_hub_download('bigcode/starcoderbase-1b', 'config.json',
                  revision='182f0165fdf8da9c9935901eec65c94337f01c11')
The CLI validates all local prerequisites before prepare can allocate CUDA:
  python scripts/starvector-terminal-upstream-oracle.py validate \
    --upstream-root SOURCE --weights-root WEIGHTS --assets-root ASSETS \
    --components-root COMPONENTS --output FRESH_OUTPUT --tier 1b \
    --sanitizer ABSOLUTE_PRODUCTION_SANITIZER_BINARY
Replace validate with prepare only after coordinator hardware admission. Use a
fresh output root per failed attempt; successful per-tier manifests share one
output root and are reused unchanged for MLX and Candle. The constructor uses
eager attention instead of optional Flash Attention and initializes the final
tokenizer-sized embeddings directly before strictly loading checkpoint tensors;
those validation-only adapters are recorded in the generation transcript.

"""
import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import struct
import sys
import time
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
LOCK = HERE.parent / 'release/starvector-terminal-upstream-lock-v1.json'
SOURCE_INDICES = [base + i for base in (0, 30, 60, 90) for i in range(5)]


def fail(message):
    raise ValueError('upstream oracle: ' + message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def local_file(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or '\\' in relative or any(x in ('..', '.') for x in relative.split('/')):
        fail('invalid relative file path: ' + str(relative))
    item = Path(root)
    for part in relative.split('/'):
        item = item / part
        if item.is_symlink():
            fail('symlink is forbidden: ' + str(item))
    if not item.is_file() or not item.resolve().is_relative_to(Path(root).resolve()):
        fail('missing local file: ' + str(item))
    return item


def verified_file(root, relative, expected):
    path = local_file(root, relative)
    if not re.fullmatch('[a-f0-9]{64}', str(expected)) or digest(path) != expected:
        fail('file identity mismatch: ' + str(path))
    return path


def inventory(root):
    entries = []
    for item in sorted(Path(root).rglob('*'), key=lambda p: p.relative_to(root).as_posix()):
        if item.is_symlink():
            fail('model inventory contains symlink: ' + str(item))
        if item.is_file():
            entries.append({'path': item.relative_to(root).as_posix(), 'byte_size': item.stat().st_size, 'sha256': digest(item)})
    if not entries:
        fail('empty model inventory')
    return hashlib.sha256(canonical(entries)).hexdigest()


def source_identity(root, lock):
    root = Path(root)
    revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != lock['implementation_revision']:
        fail('upstream source revision mismatch')
    paths = subprocess.check_output(['git', '-C', str(root), 'ls-tree', '-r', '--name-only', revision], text=True).splitlines()
    paths = sorted(p for p in paths if p.startswith('starvector/') and p.endswith('.py'))
    entries = [{'path': p, 'sha256': digest(local_file(root, p))} for p in paths]
    observed = hashlib.sha256(canonical(entries)).hexdigest()
    if observed != lock['python_source_sha256']:
        fail(f'audited upstream Python source changed: expected={lock["python_source_sha256"]} actual={observed} files={len(entries)}')
    actual = {p.relative_to(root).as_posix() for p in (root / 'starvector').rglob('*.py')}
    if actual != set(paths):
        fail('untracked upstream Python source is forbidden')
    return observed


def select_rows(assets_root):
    index = json.loads(local_file(assets_root, 'starvector-terminal-row-index-v1.json').read_text())
    rows = index.get('rows', [])
    if len(rows) != 120 or [r.get('case_index') for r in rows] != list(range(120)):
        fail('expected ordered 120-case immutable input index')
    selected, seen = [], set()
    for case_index, source_index in enumerate(SOURCE_INDICES):
        row = rows[source_index]
        png = verified_file(assets_root, row['input_png_path'], row['png_sha256'])
        if row['png_sha256'] in seen:
            fail('upstream parity requires twenty distinct input images')
        seen.add(row['png_sha256'])
        sampling = row['sampling']
        if sampling.get('temperature') != 0:
            fail('upstream parity requires greedy native sampling (temperature=0)')
        budget = row['detail_budget']['maxNewTokens']
        if sampling.get('topP') != 1.0 or sampling.get('topK') != 1 or sampling.get('repetitionPenalty') != 1.0:
            fail('parity requires the declared greedy sampling contract')
        if not 1 <= row['detail_budget'].get('maxWallTimeMs', 0) <= 3600000 or not 1 <= row['detail_budget'].get('maxSvgBytes', 0) <= 1048576:
            fail('invalid native wall-time or SVG byte budget')
        if isinstance(budget, bool) or not isinstance(budget, int) or not 1 <= budget <= 16384:
            fail('invalid new-token budget')
        selected.append({'case_index': case_index, 'source_case_index': source_index, 'seed': case_index,
                         'input_png': str(png), 'input_png_sha256': row['png_sha256'],
                         'sampling': sampling, 'detail_budget': row['detail_budget']})
    return selected


def checkpoint_map(model_root):
    index = json.loads(local_file(model_root, 'model.safetensors.index.json').read_text())
    mapping = index.get('weight_map')
    if not isinstance(mapping, dict) or not mapping:
        fail('checkpoint requires nonempty safetensors shard index')
    for shard in set(mapping.values()):
        local_file(model_root, shard)
    observed = {}
    for shard in set(mapping.values()):
        file = local_file(model_root, shard)
        with file.open('rb') as stream:
            size_bytes = stream.read(8)
            if len(size_bytes) != 8:
                fail('truncated safetensors header')
            header_size = struct.unpack('<Q', size_bytes)[0]
            if not 2 <= header_size <= min(16 * 1024**2, file.stat().st_size - 8):
                fail('invalid safetensors header size')
            header = json.loads(stream.read(header_size))
        for key, value in header.items():
            if key == '__metadata__':
                continue
            if key in observed or mapping.get(key) != shard:
                fail('duplicate or misindexed safetensors tensor: ' + key)
            observed[key] = value['shape']
            start, end = value['data_offsets']
            if not 0 <= start <= end <= file.stat().st_size - 8 - header_size:
                fail('invalid safetensors data range')
    if set(observed) != set(mapping):
        fail('checkpoint index refers to missing tensors')
    return mapping


def check_tensor_coverage(expected, mapping, observed):
    # HF safetensors intentionally removes this shared weight in both pinned
    # checkpoints. No other missing parameter or buffer is accepted.
    head = 'model.svg_transformer.transformer.lm_head.weight'
    candidates = ['model.svg_transformer.transformer.transformer.wte.weight',
                  'model.svg_transformer.transformer.model.embed_tokens.weight']
    aliases = {}
    if head in expected and head not in mapping:
        targets = [key for key in candidates if key in mapping]
        if len(targets) != 1 or expected[head] != expected[targets[0]]:
            fail('unverifiable tied language-model head')
        aliases[head] = targets[0]
    if set(expected) - set(aliases) != set(mapping):
        fail('checkpoint/model tensor key coverage mismatch')
    if set(observed) != set(mapping):
        fail('safetensors/index key coverage mismatch')
    for key, shape in observed.items():
        if list(shape) != list(expected[key]):
            fail('checkpoint tensor shape mismatch: ' + key)
    return aliases


def validate(args, packages=True):
    lock = json.loads(LOCK.read_text())
    source_hash = source_identity(args.upstream_root, lock)
    if packages:
        for name, version in lock['required_packages'].items():
            try:
                actual = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                fail('missing validation-only package: ' + name)
            if actual != version:
                fail('oracle package version mismatch: ' + name + '=' + actual)
    manifest = json.loads(local_file(args.weights_root, 'starvector-terminal-weights-v1.json').read_text())
    model_entry = manifest['models']['starvector-' + args.tier]
    expected = lock['checkpoints'][args.tier]
    if model_entry['revision'] != expected['revision']:
        fail('checkpoint revision mismatch')
    model_root = Path(args.weights_root) / model_entry['relative_path']
    # Check every path component, including the model directory.
    config_path = local_file(args.weights_root, model_entry['relative_path'] + '/config.json')
    model_hash = inventory(model_root)
    if model_hash != model_entry['inventory_sha256']:
        fail('checkpoint inventory differs from native model inventory')
    if digest(config_path) != expected['config_sha256']:
        fail('pinned checkpoint config mismatch')
    processor_path = verified_file(model_root, expected['processor_file'], expected['processor_sha256'])
    for name in ['tokenizer_config.json', 'vocab.json', 'merges.txt', 'special_tokens_map.json']:
        local_file(model_root, name)
    mapping = checkpoint_map(model_root)
    if len(mapping) != expected['tensor_count']:
        fail('checkpoint tensor count differs from pinned snapshot')
    components = json.loads(local_file(args.components_root, 'components.json').read_text())
    needed = ['starcoder1'] if args.tier == '1b' else ['starcoder2', 'siglip']
    configs = {}
    for key in needed:
        component = components[key]
        if component['repository'] != lock['components'][key]['repository'] or not re.fullmatch('[a-f0-9]{40}', component['revision']):
            fail('invalid component provenance: ' + key)
        pinned = lock['components'][key]
        if (pinned.get('revision') and component['revision'] != pinned['revision']) or (pinned.get('config_sha256') and component['config_sha256'] != pinned['config_sha256']):
            fail('component identity mismatch: ' + key)
        configs[key] = str(verified_file(args.components_root, component['config_path'], component['config_sha256']))
    if not Path(args.sanitizer).is_file():
        fail('production sanitizer binary is missing')
    rows = select_rows(args.assets_root)
    return {'lock': lock, 'source_sha256': source_hash, 'model_root': str(model_root), 'model_inventory_sha256': model_hash,
            'config_path': str(config_path), 'processor_path': str(processor_path), 'components': components,
            'component_configs': configs, 'rows': rows, 'weight_map': mapping}


@contextlib.contextmanager
def offline_initialization(model_root, component_configs):
    """Retain upstream constructors but replace only their remote I/O operations."""
    import torch
    import transformers as tr
    from starvector.model.starvector_arch import SimpleStarVectorProcessor
    tokenizer_loader = tr.AutoTokenizer.from_pretrained
    processor_loader = tr.AutoImageProcessor.from_pretrained
    tokenizer = tokenizer_loader(model_root, local_files_only=True, use_fast=False, trust_remote_code=False)
    config_objects = {}
    for key, file in component_configs.items():
        raw = json.loads(Path(file).read_text())
        config_objects[key] = tr.AutoConfig.for_model(raw.pop('model_type'), **raw)
    def config_loader(name, **kwargs):
        key = 'starcoder2' if 'starcoder2' in str(name) else 'starcoder1'
        return config_objects[key]
    def causal_loader(name, config, **kwargs):
        # Eager attention uses the same HF attention computation on CPU/CUDA;
        # flash-attn is an optional kernel dependency, not a model parameter.
        config._attn_implementation = 'eager'
        # Upstream immediately resizes to this tokenizer; initialize directly at
        # that shape because mean-resizing uninitialized meta weights is invalid.
        config.vocab_size = len(tokenizer)
        return tr.AutoModelForCausalLM.from_config(config, torch_dtype=kwargs.get('torch_dtype'), trust_remote_code=False, attn_implementation='eager')
    def vision_loader(name, **kwargs):
        return tr.AutoModel.from_config(config_objects['siglip'], torch_dtype=kwargs.get('torch_dtype'), attn_implementation='eager')
    def processor(*unused, **kwargs):
        if 'starcoder1' in component_configs:
            raw = json.loads((Path(model_root) / 'processor_config.json').read_text())
            return SimpleStarVectorProcessor(tokenizer=tokenizer, size=raw['size'], mean=raw['mean'], std=raw['std'])
        return processor_loader(model_root, local_files_only=True, trust_remote_code=False)
    with contextlib.ExitStack() as stack:
        for cls, attr, replacement in [(tr.AutoConfig, 'from_pretrained', config_loader),
                                      (tr.AutoTokenizer, 'from_pretrained', lambda *a, **k: tokenizer),
                                      (tr.AutoModelForCausalLM, 'from_pretrained', causal_loader),
                                      (tr.AutoModel, 'from_pretrained', vision_loader),
                                      (tr.AutoProcessor, 'from_pretrained', processor),
                                      (tr.AutoImageProcessor, 'from_pretrained', processor),
                                      (tr.utils, 'is_flash_attn_2_available', lambda: False)]:
            stack.enter_context(patch.object(cls, attr, replacement))
        yield


def load_model(facts, device):
    import torch
    from safetensors import safe_open
    from starvector.model.starvector_arch import StarVectorConfig, StarVectorForCausalLM
    raw = json.loads(Path(facts['config_path']).read_text())
    raw['_name_or_path'] = facts['model_root']
    config = StarVectorConfig(**raw)
    # Construction allocates no real model tensors; each verified shard is then
    # assigned once on CPU, avoiding duplicate 8B resident copies.
    from accelerate import init_empty_weights
    with init_empty_weights(include_buffers=False), offline_initialization(facts['model_root'], facts['component_configs']):
        model = StarVectorForCausalLM(config)
    expected = {k: list(v.shape) for k, v in model.state_dict().items()}
    observed, shards = {}, {}
    for shard in sorted(set(facts['weight_map'].values())):
        file = local_file(facts['model_root'], shard)
        with safe_open(file, framework='pt', device='cpu') as handle:
            for key in handle.keys():
                if key in observed or facts['weight_map'].get(key) != shard:
                    fail('duplicate or misindexed checkpoint tensor: ' + key)
                observed[key] = list(handle.get_slice(key).get_shape())
        shards[shard] = file
    aliases = check_tensor_coverage(expected, facts['weight_map'], observed)
    for shard, file in shards.items():
        with safe_open(file, framework='pt', device='cpu') as handle:
            state = {key: handle.get_tensor(key) for key in handle.keys()}
        result = model.load_state_dict(state, strict=False, assign=True)
        if result.unexpected_keys:
            fail('unexpected tensor after verified load')
        del state
    lm = model.model.svg_transformer.transformer
    if aliases:
        if not lm.config.tie_word_embeddings:
            fail('checkpoint omitted a head but component config does not tie embeddings')
        lm.tie_weights()
        if lm.get_input_embeddings().weight is not lm.get_output_embeddings().weight:
            fail('language-model head did not bind to exact checkpoint embeddings')
    if any(t.is_meta for t in list(model.parameters()) + list(model.buffers())):
        fail('uninitialized meta tensor remains after full checkpoint load')
    return model.eval().to(device), {'checkpoint_tensor_count': len(observed), 'tied_aliases': aliases}


def generate(model, row, device):
    import torch
    from PIL import Image
    torch.manual_seed(row['seed'])
    torch.cuda.manual_seed_all(row['seed'])
    with Image.open(row['input_png']) as source:
        # Corpus PNGs are canonical opaque rasters; preserve the exact upstream
        # processor resize/padding/normalization rather than duplicating it.
        image = source.convert('RGB')
        pixels = model.model.processor(images=image, return_tensors='pt')['pixel_values']
    if pixels.ndim == 3:
        pixels = pixels.unsqueeze(0)
    pixels = pixels.to(device)
    lm = model.model.svg_transformer.transformer
    original = lm.generate
    observation = {}
    budget = row['detail_budget']['maxNewTokens']
    deadline = time.monotonic() + row['detail_budget']['maxWallTimeMs'] / 1000
    from transformers import StoppingCriteria, StoppingCriteriaList
    class Deadline(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            if time.monotonic() > deadline:
                fail('upstream exceeded the native case wall-time budget')
            return False
    def bounded_generate(**kwargs):
        if kwargs.get('do_sample') is not False or kwargs.get('num_beams') != 1:
            fail('upstream wrapper did not select greedy decoding')
        observation['prefix_length'] = int(kwargs['inputs_embeds'].shape[1])
        kwargs.pop('max_length', None)
        kwargs['max_new_tokens'] = budget
        kwargs['stopping_criteria'] = StoppingCriteriaList([*kwargs.get('stopping_criteria', []), Deadline()])
        output = original(**kwargs)
        observation['generated_tokens'] = int(output.shape[1])
        if observation['generated_tokens'] > budget:
            fail('upstream exceeded the native new-token budget')
        return output
    with torch.inference_mode(), patch.object(lm, 'generate', bounded_generate):
        result = model.generate_im2svg({'image': pixels}, use_nucleus_sampling=False, num_beams=1,
                                      max_length=budget, repetition_penalty=row['sampling'].get('repetitionPenalty', 1.0))
    if not isinstance(result, list) or len(result) != 1 or not isinstance(result[0], str):
        fail('upstream generation returned an invalid batch')
    if len(result[0].encode()) > row['detail_budget']['maxSvgBytes']:
        fail('upstream exceeded the native SVG byte budget')
    return result[0], observation


def durable_json(path, value):
    # Exclusive creates preserve original attempts. Final manifests appear only
    # after every case has succeeded; interrupted runs retain their transcript.
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def worker(args, facts):
    import torch
    sys.path.insert(0, str(Path(args.upstream_root).resolve()))
    if not args.device.startswith('cuda:') or not torch.cuda.is_available():
        fail('reference production requires the coordinator-admitted CUDA device')
    device = torch.device(args.device)
    free, total = torch.cuda.mem_get_info(device)
    if free < args.min_free_vram_gib * 1024**3:
        fail('insufficient free CUDA memory for admitted oracle bound')
    if args.max_vram_gib * 1024**3 > total:
        fail('CUDA allocation cap exceeds device memory')
    torch.cuda.set_per_process_memory_fraction(args.max_vram_gib * 1024**3 / total, device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    output = Path(args.output)
    tier_root = output / ('upstream-' + args.tier)
    tier_root.mkdir(parents=True, exist_ok=False)
    transcript_path = tier_root / 'transcript.jsonl'
    cases = []
    with transcript_path.open('x') as transcript:
        def record(event):
            transcript.write(json.dumps(event, separators=(',', ':')) + '\n'); transcript.flush(); os.fsync(transcript.fileno())
        record({'event': 'start', 'implementation_revision': facts['lock']['implementation_revision'],
                'source_sha256': facts['source_sha256'], 'checkpoint_inventory_sha256': facts['model_inventory_sha256'],
                'components': facts['components'], 'attention_implementation': 'eager', 'embedding_initialization': 'exact-local-tokenizer-size-before-strict-checkpoint-load',
                'max_rss_gib': args.max_rss_gib, 'max_vram_gib': args.max_vram_gib, 'timeout_seconds': args.timeout_seconds})
        try:
            model, coverage = load_model(facts, device)
            record({'event': 'model_loaded', **coverage})
            for row in facts['rows']:
                case_root = tier_root / ('case-%02d' % row['case_index']); case_root.mkdir()
                record({'event': 'case_started', 'started_at': time.time(), **row})
                raw, generation = generate(model, row, device)
                raw_path = case_root / 'raw.svg'; raw_path.write_text(raw)
                rendered = case_root / 'rendered'
                result = subprocess.run([args.sanitizer, 'run', str(raw_path), str(rendered)], capture_output=True, text=True, timeout=60, check=True)
                event = json.loads(result.stdout)
                if event.get('outcome') != 'sanitized_inert':
                    fail('upstream SVG rejected by canonical renderer: ' + str(row['case_index']))
                svg = local_file(rendered, 'canonical.svg'); preview = local_file(rendered, 'preview.png')
                from PIL import Image
                with Image.open(preview) as image:
                    if image.size != (512, 512):
                        fail('upstream canonical preview is not 512x512')
                case = {key: row[key] for key in ['case_index', 'source_case_index', 'seed', 'input_png_sha256']}
                case.update(upstream_svg=svg.relative_to(output).as_posix(), upstream_svg_sha256=digest(svg),
                            upstream_preview_png=preview.relative_to(output).as_posix(), upstream_preview_png_sha256=digest(preview))
                cases.append(case)
                record({'event': 'case_completed', **case, **generation, 'raw_svg_sha256': digest(raw_path)})
            del model
            torch.cuda.empty_cache()
            record({'event': 'completed', 'cases': len(cases), 'peak_cuda_bytes': torch.cuda.max_memory_allocated(device)})
        except BaseException as exc:
            record({'event': 'failed', 'error': str(exc), 'completed_cases': len(cases)})
            raise
    config_copy = tier_root / 'config.json'; shutil.copyfile(facts['config_path'], config_copy)
    processor_copy = tier_root / 'processor.json'; shutil.copyfile(facts['processor_path'], processor_copy)
    value = {'schema_version': 1, 'upstream_reference': reference_metadata(facts, args.tier, config_copy, processor_copy, transcript_path),
        'config_path': config_copy.relative_to(output).as_posix(), 'processor_path': processor_copy.relative_to(output).as_posix(),
        'transcript_path': transcript_path.relative_to(output).as_posix(), 'cases': cases}
    durable_json(output / ('upstream-reference-' + args.tier + '.json'), value)


def reference_metadata(facts, tier, config_path, processor_path, transcript_path):
    checkpoint = facts['lock']['checkpoints'][tier]
    return {'implementation_repository': facts['lock']['implementation_repository'],
            'implementation_revision': facts['lock']['implementation_revision'],
            'checkpoint_repository': checkpoint['repository'], 'checkpoint_revision': checkpoint['revision'],
            'checkpoint_inventory_sha256': facts['model_inventory_sha256'], 'config_sha256': digest(config_path),
            'processor_sha256': digest(processor_path), 'transcript_sha256': digest(transcript_path)}


def supervise(args):
    import psutil
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    if (output / ('upstream-' + args.tier)).exists() or (output / ('upstream-reference-' + args.tier + '.json')).exists():
        fail('output already contains this tier; preserve the attempt and select a fresh output directory')
    command = [sys.executable, str(Path(__file__).resolve()), '_worker', *sys.argv[2:]]
    start = time.monotonic()
    with (output / ('upstream-' + args.tier + '-process.log')).open('x') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                try:
                    root = psutil.Process(process.pid)
                    rss = 0
                    for child in [root, *root.children(recursive=True)]:
                        try:
                            rss += child.memory_info().rss
                        except psutil.NoSuchProcess:
                            pass
                except psutil.NoSuchProcess:
                    process.wait()
                    break
                if rss > args.max_rss_gib * 1024**3:
                    fail('host RSS limit exceeded')
                transcript_path = output / ('upstream-' + args.tier) / 'transcript.jsonl'
                if transcript_path.is_file():
                    lines = transcript_path.read_text().splitlines()
                    try:
                        event = json.loads(lines[-1]) if lines else {}
                    except json.JSONDecodeError:
                        event = {}  # writer may be between write and flush
                    if event.get('event') == 'case_started' and time.time() > event['started_at'] + event['detail_budget']['maxWallTimeMs'] / 1000 + 5:
                        fail('hard per-case runtime deadline exceeded')
                if time.monotonic() - start > args.timeout_seconds:
                    fail('hard runtime deadline exceeded')
                time.sleep(1)
            if process.returncode:
                fail('upstream worker failed; preserved process log and partial transcript')
        finally:
            if process.poll() is None:
                root = psutil.Process(process.pid)
                children = root.children(recursive=True)
                for child in children:
                    child.kill()
                process.kill(); process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['validate', 'prepare', '_worker'])
    for key in ['upstream-root', 'weights-root', 'assets-root', 'output', 'components-root', 'sanitizer']:
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--tier', choices=['1b', '8b'], required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--timeout-seconds', type=int, default=3600)
    parser.add_argument('--max-rss-gib', type=float, default=40)
    parser.add_argument('--max-vram-gib', type=float, default=30)
    parser.add_argument('--min-free-vram-gib', type=float, default=24)
    args = parser.parse_args()
    if not (1 <= args.timeout_seconds <= 14400 and 1 <= args.max_rss_gib <= 64 and 1 <= args.max_vram_gib <= 80 and 1 <= args.min_free_vram_gib <= args.max_vram_gib):
        fail('invalid execution resource bounds')
    for name in ['HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE']:
        os.environ[name] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    facts = validate(args)
    if args.command == 'validate':
        print(json.dumps({'status': 'validated', 'tier': args.tier, 'model_inventory_sha256': facts['model_inventory_sha256'], 'source_sha256': facts['source_sha256'], 'cases': len(facts['rows'])}))
    elif args.command == '_worker':
        worker(args, facts)
    else:
        supervise(args)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, subprocess.SubprocessError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
