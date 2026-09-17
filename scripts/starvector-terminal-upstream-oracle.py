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
import tarfile
import tempfile
import time
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
LOCK = HERE.parent / 'release/starvector-terminal-upstream-lock-v1.json'
SOURCE_INDICES = [base + i for base in (0, 30, 60, 90) for i in range(5)]
DIAGNOSTIC_CASE_INDEX = 9
DIAGNOSTIC_CORPUS_SHA256 = 'dbfd6b6ef972f3104f7a20c2033af6157b720e49adf46c506e0b34ef612d3a59'
DIAGNOSTIC_ROWS_SHA256 = 'f9529c2e5a86bef6644054c909c4f621991f6384d9b33a029ad46ff2e6cd3b88'
DIAGNOSTIC_BUDGET = {'maxNewTokens': 7933, 'maxSvgBytes': 262144, 'maxWallTimeMs': 120000}
DIAGNOSTIC_SAMPLING = {'temperature': 0.0, 'topP': 1.0, 'topK': 1,
                       'repetitionPenalty': 1.0, 'seed': 7}
_CAIRO_HANDLES = []


class SvgCaseRejected(ValueError):
    """A generated SVG was explicitly rejected by the canonical sanitizer."""

    def __init__(self, case_index, event):
        self.case_index = case_index
        self.event = event
        super().__init__('upstream SVG rejected by canonical renderer: case '
                         + str(case_index) + ': ' + event['error_code'])


class CollectedSvgRejections(ValueError):
    """All planned cases ran, but at least one SVG was rejected."""


def native_cairo_files(lock):
    files = {}
    for package in lock['windows_cairo']['packages']:
        for entry in package['files']:
            name = entry['path']
            if not re.fullmatch(r'ucrt64/(?:bin/[^/]+\.dll|share/licenses/[A-Za-z0-9._/+\-]+)', name) or '..' in name.split('/') or name in files:
                fail('invalid native Cairo file path')
            files[name] = entry
    return files


def native_directory(root):
    root = Path(root)
    if not root.is_absolute():
        fail('native Cairo root must be absolute')
    for item in [root, *root.parents]:
        if item.is_symlink() or getattr(item, 'is_junction', lambda: False)():
            fail('native Cairo directory must not traverse links')
        if item.exists() and not item.is_dir():
            fail('native Cairo directory is not regular')
    return root


def verify_native_cairo(root, lock):
    root = native_directory(root)
    files = native_cairo_files(lock)
    observed = set()
    for item in root.rglob('*'):
        if item.is_symlink() or getattr(item, 'is_junction', lambda: False)():
            fail('native Cairo runtime contains links')
        if item.is_file():
            observed.add(item.relative_to(root).as_posix())
    if observed != set(files):
        fail('native Cairo runtime file inventory differs')
    for name, entry in files.items():
        file = verified_file(root, name, entry['sha256'])
        if file.stat().st_size != entry['byte_size']:
            fail('native Cairo runtime file size differs')
    return root


def provision_native_cairo(root, archives, lock):
    """Extract only pinned DLLs/licenses from authenticated MSYS2 archives."""
    import zstandard
    root = native_directory(root)
    native_directory(archives)
    if root.exists():
        verify_native_cairo(root, lock)
        return {'status': 'reused', 'version': lock['windows_cairo']['version']}
    files = native_cairo_files(lock)
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=root.name + '.staging-', dir=root.parent))
    try:
        for package in lock['windows_cairo']['packages']:
            archive = verified_file(archives, package['sha256'] + '.pkg.tar.zst', package['sha256'])
            if archive.stat().st_size != package['byte_size']:
                fail('native Cairo archive size differs')
            selected = {entry['path'] for entry in package['files']}
            seen = set()
            with archive.open('rb') as source, zstandard.ZstdDecompressor().stream_reader(source) as stream, tarfile.open(fileobj=stream, mode='r|') as bundle:
                for member in bundle:
                    if member.name not in selected:
                        continue
                    entry = files[member.name]
                    if member.name in seen or not member.isfile() or member.size != entry['byte_size']:
                        fail('native Cairo archive entry differs')
                    data = bundle.extractfile(member).read(member.size + 1)
                    if len(data) != member.size or hashlib.sha256(data).hexdigest() != entry['sha256']:
                        fail('native Cairo archive entry digest differs')
                    output = staging / member.name
                    output.parent.mkdir(parents=True, exist_ok=True)
                    with output.open('xb') as target:
                        target.write(data)
                    seen.add(member.name)
            if seen != selected:
                fail('native Cairo archive is incomplete')
        verify_native_cairo(staging, lock)
        # Never replace an existing runtime; concurrent or corrupt setup fails closed.
        if root.exists():
            fail('native Cairo runtime appeared during provisioning')
        staging.rename(root)
        return {'status': 'provisioned', 'version': lock['windows_cairo']['version']}
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def import_upstream_runtime(upstream_root, lock):
    """Exercise both lazy constructor import paths without constructing a model."""
    if sys.platform == 'win32':
        import ctypes
        runtime = verify_native_cairo(Path(upstream_root).parent / 'upstream-native-cairo', lock)
        dll = runtime / lock['windows_cairo']['entrypoint']
        # LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32;
        # neither the current directory nor ambient PATH can supply dependencies.
        _CAIRO_HANDLES.append(os.add_dll_directory(str(dll.parent)))
        _CAIRO_HANDLES.append(ctypes.WinDLL(str(dll), winmode=0x00001100))
        os.environ['CAIROCFFI_DLL_DIRECTORIES'] = str(dll.parent)
    sys.path.insert(0, str(Path(upstream_root).resolve()))
    for module in ['starvector.model.models.starvector_v1', 'starvector.model.models.starvector_v2']:
        importlib.import_module(module)
    cairo = importlib.import_module('cairocffi')
    version = cairo.cairo_version_string()
    if sys.platform == 'win32' and version != lock['windows_cairo']['version']:
        fail('loaded native Cairo version differs: ' + version)
    # Check the native rendering path as well as dlopen/version resolution. This
    # tiny setup probe does not replace the production resvg acceptance renderer.
    png = importlib.import_module('cairosvg').svg2png(bytestring=b'<svg xmlns="http://www.w3.org/2000/svg" width="2" height="2"><rect width="2" height="2" fill="red"/></svg>')
    if len(png) < 24 or png[:8] != b'\x89PNG\r\n\x1a\n' or png[12:16] != b'IHDR' or struct.unpack('>II', png[16:24]) != (2, 2):
        fail('native Cairo render probe did not produce a 2x2 PNG')
    return {'constructor_imports': ['starvector_v1', 'starvector_v2'], 'cairo_version': version, 'cairo_probe_sha256': hashlib.sha256(png).hexdigest()}


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


def absolute_regular_file(value, description):
    item = Path(value)
    if not item.is_absolute():
        fail(description + ' path must be absolute')
    for component in [item, *item.parents]:
        if component.is_symlink() or getattr(component, 'is_junction', lambda: False)():
            fail(description + ' path must not traverse links')
    if not item.is_file():
        fail(description + ' is missing or not a regular file')
    return item


def local_file(root, relative):
    # Corpus paths use forward slashes on every host. Windows considers /name
    # drive-relative, so Path.is_absolute() alone does not reject that spelling.
    if not isinstance(relative, str) or not relative or relative.startswith('/') or Path(relative).is_absolute() or '\\' in relative or any(x in ('..', '.') for x in relative.split('/')):
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


def select_rows(assets_root, tier):
    expected_budget = {'1b': 7933, '8b': 15422}.get(tier)
    if expected_budget is None:
        fail('unsupported StarVector tier')
    index = json.loads(local_file(assets_root, 'starvector-terminal-row-index-v1.json').read_text())
    rows = index.get('rows', [])
    if index.get('schema_version') != 2 or len(rows) != 120 or [r.get('case_index') for r in rows] != list(range(120)):
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
        detail_budget = row.get('detail_budgets', {}).get(tier)
        if not isinstance(detail_budget, dict):
            fail('row lacks model-specific shipping Detail budget')
        budget = detail_budget['maxNewTokens']
        if sampling.get('topP') != 1.0 or sampling.get('topK') != 1 or sampling.get('repetitionPenalty') != 1.0:
            fail('parity requires the declared greedy sampling contract')
        if detail_budget.get('maxWallTimeMs') != 120000 or detail_budget.get('maxSvgBytes') != 262144:
            fail('row differs from the shipping wall-time or SVG byte budget')
        if isinstance(budget, bool) or not isinstance(budget, int) or budget != expected_budget:
            fail('row differs from the model shipping new-token budget')
        selected.append({'case_index': case_index, 'source_case_index': source_index, 'seed': case_index,
                         'input_png': str(png), 'input_png_sha256': row['png_sha256'],
                         'sampling': sampling, 'detail_budget': detail_budget})
    return selected


def source_rows_sha256(rows):
    fields = ['dataset', 'revision', 'row_index', 'filename', 'svg_sha256']
    serialized = ''.join(json.dumps({key: row[key] for key in fields}, ensure_ascii=False,
                                    separators=(',', ':')) + '\n'
                         for row in rows)
    return hashlib.sha256(serialized.encode()).hexdigest()


def select_diagnostic_case_9(assets_root, tier):
    """Authenticate the retired readiness corpus and expose its fixed case 9 diagnostic view."""
    if tier != '1b':
        fail('case 9 diagnostic is fixed to StarVector 1B')
    index_path = local_file(assets_root, 'starvector-terminal-row-index-v1.json')
    if digest(index_path) != DIAGNOSTIC_CORPUS_SHA256:
        fail('case 9 diagnostic corpus index is not the authenticated readiness input')
    index = json.loads(index_path.read_text())
    rows = index.get('rows', [])
    if (index.get('schema_version') != 1 or len(rows) != 120
            or [row.get('case_index') for row in rows] != list(range(120))
            or index.get('row_identity_sha256') != DIAGNOSTIC_ROWS_SHA256
            or source_rows_sha256(rows) != DIAGNOSTIC_ROWS_SHA256):
        fail('case 9 diagnostic corpus row identities drifted')
    row = rows[DIAGNOSTIC_CASE_INDEX]
    png = verified_file(assets_root, row['input_png_path'], row['png_sha256'])
    if row.get('sampling') != DIAGNOSTIC_SAMPLING:
        fail('case 9 diagnostic greedy sampling drifted')
    return {'case_index': DIAGNOSTIC_CASE_INDEX, 'source_case_index': DIAGNOSTIC_CASE_INDEX,
            'seed': 7, 'input_png': str(png), 'input_png_sha256': row['png_sha256'],
            'sampling': dict(DIAGNOSTIC_SAMPLING), 'detail_budget': dict(DIAGNOSTIC_BUDGET)}


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
    sanitizer = absolute_regular_file(args.sanitizer, 'production sanitizer binary')
    args.sanitizer = str(sanitizer)
    rows = ([select_diagnostic_case_9(args.assets_root, args.tier)]
            if args.command in ('diagnose-case-9', '_diagnostic_worker')
            else select_rows(args.assets_root, args.tier))
    runtime = import_upstream_runtime(args.upstream_root, lock) if packages else None
    return {'lock': lock, 'source_sha256': source_hash, 'model_root': str(model_root), 'model_inventory_sha256': model_hash,
            'config_path': str(config_path), 'processor_path': str(processor_path), 'components': components,
            'component_configs': configs, 'rows': rows, 'weight_map': mapping, 'runtime': runtime,
            'sanitizer_sha256': digest(sanitizer)}


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


def complete_svg_prefix(value):
    """Return the end of one structurally complete SVG root, or None."""
    stack = []
    index = 0
    length = len(value)
    while index < length and value[index].isspace():
        index += 1
    if not value[index:].startswith('<svg'):
        return None
    while index < length:
        if value[index] != '<':
            if not stack and not value[index].isspace():
                return None
            index += 1
            continue
        if value.startswith('<!--', index):
            end = value.find('-->', index + 4)
            if end < 0:
                return None
            index = end + 3
            continue
        if value.startswith('<![CDATA[', index):
            end = value.find(']]>', index + 9)
            if end < 0:
                return None
            index = end + 3
            continue
        if value.startswith('<?', index):
            end = value.find('?>', index + 2)
            if end < 0:
                return None
            index = end + 2
            continue
        quote = None
        end = index + 1
        while end < length:
            char = value[end]
            if quote:
                if char == quote:
                    quote = None
            elif char in "'\"":
                quote = char
            elif char == '>':
                break
            end += 1
        if end >= length:
            return None
        body = value[index + 1:end].strip()
        if not body or body.startswith('!'):
            return None
        closing = body.startswith('/')
        self_closing = body.endswith('/')
        name_body = body[1:].lstrip() if closing else body
        name = name_body.split(None, 1)[0].rstrip('/')
        if not name:
            return None
        if closing:
            if not stack or stack[-1] != name:
                return None
            stack.pop()
            if not stack:
                return end + 1 if value[end + 1:].strip() == '' else None
        elif not stack:
            if name != 'svg':
                return None
            if self_closing:
                return end + 1 if value[end + 1:].strip() == '' else None
            stack.append(name)
        elif not self_closing:
            stack.append(name)
        index = end + 1
    return None


def bounded_completion(value, tokens, now, deadline, max_bytes):
    end = complete_svg_prefix(value)
    byte_count = len(value[:end].encode()) if end is not None else len(value.encode())
    if end is not None and byte_count <= max_bytes and now <= deadline:
        return {'complete_root': True, 'completion_tokens': tokens,
                'completion_bytes': byte_count, 'completion_end': end}
    if byte_count > max_bytes:
        return {'byte_exceeded': True}
    if now > deadline:
        return {'deadline_exceeded': True}
    return {}


def classify_generation(raw, observation, budget, max_bytes):
    end = complete_svg_prefix(raw)
    completed = observation.get('complete_root') is True
    if not completed and end is not None and not observation.get('deadline_exceeded') and not observation.get('byte_exceeded'):
        completed = observation.get('generated_tokens', budget + 1) <= budget and len(raw[:end].encode()) <= max_bytes
    if completed and end is not None:
        if len(raw.encode()) <= max_bytes and observation.get('completion_tokens', observation.get('generated_tokens', budget + 1)) <= budget:
            return raw, 'complete'
    if observation.get('deadline_exceeded'):
        return raw, 'wall_time_limit'
    if observation.get('byte_exceeded') or len(raw.encode()) > max_bytes:
        return raw, 'byte_limit'
    if observation.get('generated_tokens', 0) >= budget:
        return raw, 'token_limit'
    # Natural EOS without a complete root still goes through the sanitizer and
    # is rejected as malformed; EOS alone is never proof of valid SVG output.
    return raw, 'complete'


def diagnostic_generation_outcome(raw, finish_reason):
    """Disambiguate the oracle's normal `complete` bucket for diagnostic evidence."""
    structural_complete_root = complete_svg_prefix(raw) is not None
    if finish_reason == 'complete':
        outcome = 'bounded_complete_root' if structural_complete_root else 'early_eos_incomplete'
    else:
        outcome = finish_reason
    return {'structural_complete_root': structural_complete_root,
            'diagnostic_outcome': outcome}


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
    tokenizer = getattr(model.model.processor, 'tokenizer', None)
    class BoundedCompletion(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            now = time.monotonic()
            if tokenizer is not None:
                decoded = tokenizer.decode(input_ids[0], skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)
                bounded = bounded_completion(decoded, int(input_ids.shape[-1]), now, deadline,
                                              row['detail_budget']['maxSvgBytes'])
                observation.update(bounded)
                if bounded.get('complete_root'):
                    return True
                if bounded.get('byte_exceeded') or bounded.get('deadline_exceeded'):
                    return True
            if tokenizer is None and now > deadline:
                observation['deadline_exceeded'] = True
                return True
            return False
    def bounded_generate(**kwargs):
        if kwargs.get('do_sample') is not False or kwargs.get('num_beams') != 1:
            fail('upstream wrapper did not select greedy decoding')
        observation['prefix_length'] = int(kwargs['inputs_embeds'].shape[1])
        kwargs.pop('max_length', None)
        kwargs['max_new_tokens'] = budget
        kwargs['stopping_criteria'] = StoppingCriteriaList([*kwargs.get('stopping_criteria', []), BoundedCompletion()])
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
    raw, finish_reason = classify_generation(result[0], observation, budget, row['detail_budget']['maxSvgBytes'])
    generated_bytes = len(raw.encode())
    observation['generated_bytes'] = generated_bytes
    observation['finish_reason'] = finish_reason
    return raw, observation


def durable_json(path, value):
    # Exclusive creates preserve original attempts. Final manifests appear only
    # after every case has succeeded; interrupted runs retain their transcript.
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def render_upstream_svg(sanitizer, raw_path, rendered, case_index):
    # Keep the exact generated SVG and renderer diagnostics even when policy
    # rejects it. Comparison uses an explicit raster size, never rewritten SVG.
    result = subprocess.run([sanitizer, 'run', str(raw_path), str(rendered), '--preview-size', '512'],
                            capture_output=True, text=True, encoding='utf-8', errors='strict',
                            timeout=60, check=False)
    case_root = Path(raw_path).parent
    (case_root / 'sanitizer.stdout.log').write_bytes(result.stdout.encode('utf-8'))
    (case_root / 'sanitizer.stderr.log').write_bytes(result.stderr.encode('utf-8'))
    if result.returncode:
        fail('canonical renderer failed for case ' + str(case_index) + ': exit ' + str(result.returncode)
             + '; see sanitizer.stderr.log')
    try:
        event = json.loads(result.stdout)
    except (ValueError, TypeError):
        fail('canonical renderer returned invalid JSON for case ' + str(case_index)
             + '; see sanitizer.stdout.log')
    if not isinstance(event, dict):
        fail('canonical renderer returned invalid result for case ' + str(case_index))
    if event.get('outcome') == 'rejected':
        if (not isinstance(event.get('error_code'), str) or not event['error_code']
                or event.get('canonical_svg_sha256') is not None
                or event.get('preview_png_sha256') is not None
                or event.get('published_paths') != [] or event.get('staging_residue') != []
                or event.get('result_contains_inline_svg') is not False or os.path.lexists(rendered)):
            fail('canonical renderer returned invalid rejection for case ' + str(case_index))
        raise SvgCaseRejected(case_index, event)
    if event.get('outcome') != 'sanitized_inert':
        fail('canonical renderer returned invalid result for case ' + str(case_index))
    svg = absolute_regular_file(Path(rendered) / 'canonical.svg', 'canonical renderer SVG')
    preview = absolute_regular_file(Path(rendered) / 'preview.png', 'canonical renderer preview')
    if (set(event) != {'outcome', 'error_code', 'canonical_svg_sha256', 'preview_png_sha256',
                       'published_paths', 'staging_residue', 'result_contains_inline_svg'}
            or event.get('error_code') != 'sanitized_inert'
            or event.get('canonical_svg_sha256') != digest(svg)
            or event.get('preview_png_sha256') != digest(preview)
            or event.get('published_paths') != ['canonical.svg', 'preview.png']
            or event.get('staging_residue') != []
            or event.get('result_contains_inline_svg') is not False):
        fail('canonical renderer returned invalid success for case ' + str(case_index))
    return event


def collect_cases(args, facts, model, device, output, tier_root, record, cases=None, rejections=None):
    cases = [] if cases is None else cases
    rejections = [] if rejections is None else rejections
    for row in facts['rows']:
        case_root = tier_root / ('case-%02d' % row['case_index']); case_root.mkdir()
        record({'event': 'case_started', 'started_at': time.time(), **row})
        raw, generation = generate(model, row, device)
        # Serialize the upstream Python string identically on Windows and Unix. Locale-default
        # text I/O turned U+2013 into CP1252 0x96 on the Windows oracle host.
        raw_path = case_root / 'raw.svg'; raw_path.write_bytes(raw.encode('utf-8'))
        identity = {key: row[key] for key in ['case_index', 'source_case_index', 'seed', 'input_png_sha256']}
        if generation['finish_reason'] != 'complete':
            rejection = {**identity, 'outcome': 'rejected', 'rejection_stage': 'generation_limit',
                         'rejection_code': generation['finish_reason'],
                         'rejection_reason': 'upstream StarVector stopped at the ' + generation['finish_reason'],
                         'upstream_raw_svg': raw_path.relative_to(output).as_posix(),
                         'upstream_raw_svg_sha256': digest(raw_path)}
            cases.append(rejection); rejections.append(rejection)
            record({'event': 'case_rejected', **rejection, 'error_code': rejection['rejection_reason'],
                    'raw_svg_sha256': rejection['upstream_raw_svg_sha256'], **generation})
            continue
        rendered = case_root / 'rendered'
        try:
            render_upstream_svg(args.sanitizer, raw_path, rendered, row['case_index'])
        except SvgCaseRejected as exc:
            stdout = case_root / 'sanitizer.stdout.log'; stderr = case_root / 'sanitizer.stderr.log'
            rejection = {**identity, 'outcome': 'rejected', 'rejection_stage': 'sanitizer',
                         'rejection_reason': exc.event['error_code'],
                         'upstream_raw_svg': raw_path.relative_to(output).as_posix(),
                         'upstream_raw_svg_sha256': digest(raw_path),
                         'sanitizer_stdout': stdout.relative_to(output).as_posix(),
                         'sanitizer_stdout_sha256': digest(stdout),
                         'sanitizer_stderr': stderr.relative_to(output).as_posix(),
                         'sanitizer_stderr_sha256': digest(stderr)}
            cases.append(rejection); rejections.append(rejection)
            record({'event': 'case_rejected', **rejection, 'error_code': rejection['rejection_reason'],
                    'raw_svg_sha256': rejection['upstream_raw_svg_sha256'], **generation})
            continue
        svg = local_file(rendered, 'canonical.svg'); preview = local_file(rendered, 'preview.png')
        from PIL import Image
        with Image.open(preview) as image:
            if image.size != (512, 512):
                fail('upstream canonical preview is not 512x512')
        case = {**identity, 'outcome': 'accepted'}
        case.update(upstream_svg=svg.relative_to(output).as_posix(), upstream_svg_sha256=digest(svg),
                    upstream_preview_png=preview.relative_to(output).as_posix(), upstream_preview_png_sha256=digest(preview))
        cases.append(case)
        record({'event': 'case_completed', **case, **generation, 'raw_svg_sha256': digest(raw_path)})
    return cases, rejections


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
    cases, rejections = [], []
    with transcript_path.open('x') as transcript:
        def record(event):
            transcript.write(json.dumps(event, separators=(',', ':')) + '\n'); transcript.flush(); os.fsync(transcript.fileno())
        record({'event': 'start', 'implementation_revision': facts['lock']['implementation_revision'],
                'source_sha256': facts['source_sha256'], 'checkpoint_inventory_sha256': facts['model_inventory_sha256'],
                'sanitizer_sha256': facts['sanitizer_sha256'],
                'components': facts['components'], 'runtime': facts['runtime'], 'attention_implementation': 'eager', 'embedding_initialization': 'exact-local-tokenizer-size-before-strict-checkpoint-load',
                'max_rss_gib': args.max_rss_gib, 'max_vram_gib': args.max_vram_gib, 'timeout_seconds': args.timeout_seconds})
        try:
            model, coverage = load_model(facts, device)
            record({'event': 'model_loaded', **coverage})
            try:
                collect_cases(args, facts, model, device, output, tier_root, record, cases, rejections)
            finally:
                del model
                torch.cuda.empty_cache()
            record({'event': 'completed', 'cases': len(cases),
                    'accepted_cases': len(cases) - len(rejections), 'rejected_cases': len(rejections),
                    'peak_cuda_bytes': torch.cuda.max_memory_allocated(device)})
        except BaseException as exc:
            event = {'event': 'failed', 'error': str(exc), 'completed_cases': len(cases)}
            if isinstance(exc, CollectedSvgRejections):
                event['failure_kind'] = 'svg_case_rejections'
            if rejections:
                event.update(collected_cases=len(cases), rejected_cases=rejections)
            record(event)
            raise
    config_copy = tier_root / 'config.json'; shutil.copyfile(facts['config_path'], config_copy)
    processor_copy = tier_root / 'processor.json'; shutil.copyfile(facts['processor_path'], processor_copy)
    value = {'schema_version': 2, 'upstream_reference': reference_metadata(facts, args.tier, config_copy, processor_copy, transcript_path),
        'config_path': config_copy.relative_to(output).as_posix(), 'processor_path': processor_copy.relative_to(output).as_posix(),
        'transcript_path': transcript_path.relative_to(output).as_posix(), 'cases': cases}
    suffix = '.pending.json' if args.defer_manifest else '.json'
    durable_json(output / ('upstream-reference-' + args.tier + suffix), value)


def diagnostic_worker(args, facts):
    """Run only authenticated legacy case 9 and emit evidence that cannot be promoted."""
    import torch
    sys.path.insert(0, str(Path(args.upstream_root).resolve()))
    if args.tier != '1b' or not args.device.startswith('cuda:') or not torch.cuda.is_available():
        fail('case 9 diagnostic requires the coordinator-admitted CUDA 1B device')
    device = torch.device(args.device)
    free, total = torch.cuda.mem_get_info(device)
    if free < args.min_free_vram_gib * 1024**3:
        fail('insufficient free CUDA memory for admitted diagnostic bound')
    if args.max_vram_gib * 1024**3 > total:
        fail('CUDA allocation cap exceeds device memory')
    torch.cuda.set_per_process_memory_fraction(args.max_vram_gib * 1024**3 / total, device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    output = Path(args.output)
    tier_root = output / 'diagnostic-case-9'
    tier_root.mkdir(parents=True, exist_ok=False)
    transcript_path = tier_root / 'transcript.jsonl'
    row = facts['rows'][0]
    with transcript_path.open('x') as transcript:
        def record(event):
            transcript.write(json.dumps(event, separators=(',', ':')) + '\n')
            transcript.flush(); os.fsync(transcript.fileno())
        record({'event': 'start', 'kind': 'starvector_upstream_case_9_diagnostic',
                'acceptance_use': 'diagnostic_only', 'usable_for_terminal_acceptance': False,
                'implementation_revision': facts['lock']['implementation_revision'],
                'source_sha256': facts['source_sha256'],
                'checkpoint_inventory_sha256': facts['model_inventory_sha256'],
                'runtime': facts['runtime'], 'attention_implementation': 'eager',
                'max_rss_gib': args.max_rss_gib, 'max_vram_gib': args.max_vram_gib,
                'timeout_seconds': args.timeout_seconds})
        model, coverage = load_model(facts, device)
        try:
            record({'event': 'model_loaded', **coverage})
            record({'event': 'case_started', 'started_at': time.time(), **row})
            generation_started = time.monotonic()
            raw, generation = generate(model, row, device)
            generation_elapsed_seconds = time.monotonic() - generation_started
            raw_path = tier_root / 'raw.svg'
            raw_path.write_bytes(raw.encode('utf-8'))
            diagnostic = diagnostic_generation_outcome(raw, generation['finish_reason'])
            result = {'schema_version': 1, 'kind': 'starvector_upstream_case_9_diagnostic',
                      'acceptance_use': 'diagnostic_only', 'usable_for_terminal_acceptance': False,
                      'tier': '1b', 'case_index': DIAGNOSTIC_CASE_INDEX,
                      'source_case_index': DIAGNOSTIC_CASE_INDEX, 'seed': 7,
                      'input_png_sha256': row['input_png_sha256'],
                      'sampling': row['sampling'], 'detail_budget': row['detail_budget'],
                      'finish_reason': generation['finish_reason'],
                      **diagnostic,
                      'generated_tokens': generation['generated_tokens'],
                      'generated_bytes': generation['generated_bytes'],
                      'generation_elapsed_seconds': generation_elapsed_seconds,
                      'raw_svg': raw_path.relative_to(output).as_posix(),
                      'raw_svg_sha256': digest(raw_path),
                      'implementation_revision': facts['lock']['implementation_revision'],
                      'checkpoint_revision': facts['lock']['checkpoints']['1b']['revision'],
                      'checkpoint_inventory_sha256': facts['model_inventory_sha256'],
                      'transcript': transcript_path.relative_to(output).as_posix()}
            record({'event': 'case_completed', **result})
        finally:
            del model
            torch.cuda.empty_cache()
    result['transcript_sha256'] = digest(transcript_path)
    durable_json(output / 'diagnostic-case-9.json', result)


def reference_metadata(facts, tier, config_path, processor_path, transcript_path):
    checkpoint = facts['lock']['checkpoints'][tier]
    return {'implementation_repository': facts['lock']['implementation_repository'],
            'implementation_revision': facts['lock']['implementation_revision'],
            'checkpoint_repository': checkpoint['repository'], 'checkpoint_revision': checkpoint['revision'],
            'checkpoint_inventory_sha256': facts['model_inventory_sha256'], 'config_sha256': digest(config_path),
            'processor_sha256': digest(processor_path), 'transcript_sha256': digest(transcript_path)}


def verify_collected_rejections(output, tier, rows):
    transcript = absolute_regular_file(Path(output) / ('upstream-' + tier) / 'transcript.jsonl',
                                       'upstream rejection transcript')
    try:
        events = [json.loads(line) for line in transcript.read_text().splitlines()]
    except (ValueError, OSError, TypeError):
        fail('upstream rejection transcript is invalid')
    starts = [event for event in events if event.get('event') == 'case_started']
    results = [event for event in events if event.get('event') in ('case_completed', 'case_rejected')]
    if len(rows) != 20 or len(starts) != len(rows) or len(results) != len(rows):
        fail('upstream rejection transcript does not cover all planned cases')
    for index, (row, started, result) in enumerate(zip(rows, starts, results)):
        identity = {key: row[key] for key in ['case_index', 'source_case_index', 'seed', 'input_png_sha256']}
        if index != row['case_index'] or any(started.get(key) != value or result.get(key) != value for key, value in identity.items()):
            fail('upstream rejection transcript case identity differs')
    rejected = [event for event in results if event['event'] == 'case_rejected']
    final = events[-1] if events else {}
    keys = ['case_index', 'source_case_index', 'seed', 'input_png_sha256', 'error_code',
            'raw_svg_sha256', 'sanitizer_stdout', 'sanitizer_stderr']
    summaries = [{key: event.get(key) for key in keys} for event in rejected]
    if (not rejected or final.get('event') != 'failed' or final.get('failure_kind') != 'svg_case_rejections'
            or final.get('collected_cases') != len(rows)
            or final.get('completed_cases') != len(rows) - len(rejected)
            or final.get('rejected_cases') != summaries):
        fail('upstream rejection transcript has no authenticated terminal rejection summary')
    tier_root = Path(output) / ('upstream-' + tier)
    for item in summaries:
        index = item['case_index']
        case_root = tier_root / ('case-%02d' % index)
        raw = absolute_regular_file(case_root / 'raw.svg', 'rejected raw SVG')
        stdout = absolute_regular_file(case_root / 'sanitizer.stdout.log', 'rejected sanitizer stdout')
        absolute_regular_file(case_root / 'sanitizer.stderr.log', 'rejected sanitizer stderr')
        expected_stdout = (case_root / 'sanitizer.stdout.log').relative_to(output).as_posix()
        expected_stderr = (case_root / 'sanitizer.stderr.log').relative_to(output).as_posix()
        try:
            sanitizer_event = json.loads(stdout.read_text())
        except (ValueError, OSError, TypeError):
            fail('rejected sanitizer stdout is invalid')
        if (item['raw_svg_sha256'] != digest(raw) or item['sanitizer_stdout'] != expected_stdout
                or item['sanitizer_stderr'] != expected_stderr or sanitizer_event.get('outcome') != 'rejected'
                or sanitizer_event.get('error_code') != item['error_code']):
            fail('rejected case evidence differs from transcript')
    return summaries


def supervise(args, facts, diagnostic=False):
    import psutil
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    tier_directory = 'diagnostic-case-9' if diagnostic else 'upstream-' + args.tier
    result_file = 'diagnostic-case-9.json' if diagnostic else 'upstream-reference-' + args.tier + '.json'
    if (output / tier_directory).exists() or (output / result_file).exists():
        fail('output already contains this run; preserve the attempt and select a fresh output directory')
    worker_command = '_diagnostic_worker' if diagnostic else '_worker'
    command = [sys.executable, str(Path(__file__).resolve()), worker_command, *sys.argv[2:]]
    start = time.monotonic()
    process_log = 'diagnostic-case-9-process.log' if diagnostic else 'upstream-' + args.tier + '-process.log'
    with (output / process_log).open('x') as log:
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
                transcript_path = output / tier_directory / 'transcript.jsonl'
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
            if process.returncode == 3 and not diagnostic:
                verify_collected_rejections(output, args.tier, facts['rows'])
                raise CollectedSvgRejections('upstream tier completed with rejected SVG cases; '
                                             'preserved process log, raw SVGs, sanitizer logs, and transcript')
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
    if len(sys.argv) == 4 and sys.argv[1] == 'provision-cairo':
        print(json.dumps(provision_native_cairo(sys.argv[2], sys.argv[3], json.loads(LOCK.read_text()))))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['validate', 'prepare', 'diagnose-case-9', '_worker',
                                            '_diagnostic_worker'])
    for key in ['upstream-root', 'weights-root', 'assets-root', 'output', 'components-root', 'sanitizer']:
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--tier', choices=['1b', '8b'], required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--timeout-seconds', type=int, default=3600)
    parser.add_argument('--max-rss-gib', type=float, default=40)
    parser.add_argument('--max-vram-gib', type=float, default=30)
    parser.add_argument('--min-free-vram-gib', type=float, default=24)
    parser.add_argument('--defer-manifest', action='store_true')
    args = parser.parse_args()
    if not (1 <= args.timeout_seconds <= 14400 and 1 <= args.max_rss_gib <= 64 and 1 <= args.max_vram_gib <= 80 and 1 <= args.min_free_vram_gib <= args.max_vram_gib):
        fail('invalid execution resource bounds')
    for name in ['HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE']:
        os.environ[name] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    facts = validate(args)
    if args.command == 'validate':
        print(json.dumps({'status': 'validated', 'tier': args.tier, 'model_inventory_sha256': facts['model_inventory_sha256'], 'source_sha256': facts['source_sha256'], 'cases': len(facts['rows']), 'runtime': facts['runtime']}))
    elif args.command == '_worker':
        worker(args, facts)
    elif args.command == '_diagnostic_worker':
        diagnostic_worker(args, facts)
    elif args.command == 'diagnose-case-9':
        supervise(args, facts, diagnostic=True)
    else:
        supervise(args, facts)


if __name__ == '__main__':
    try:
        main()
    except CollectedSvgRejections as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(3)
    except (ValueError, OSError, subprocess.SubprocessError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
