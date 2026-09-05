#!/usr/bin/env python3
"""CPU-only contract/failure-path tests; no model weights, network, or GPU."""
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('oracle', Path(__file__).with_name('starvector-terminal-upstream-oracle.py'))
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)


class OracleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_actual_lock_metadata_matches_v2_consumer_contract(self):
        lock = json.loads((Path(__file__).parent.parent / 'release/starvector-terminal-upstream-lock-v1.json').read_text())
        artifact = self.root / 'artifact.json'; artifact.write_text('{"cpu_fixture":true}\n')
        facts = {'lock': lock, 'model_inventory_sha256': 'a' * 64}
        reference = oracle.reference_metadata(facts, '1b', artifact, artifact, artifact)
        self.assertEqual(reference['implementation_repository'], 'https://github.com/joanrod/star-vector')
        self.assertEqual(reference['implementation_revision'], '0e083c1911760aa31bc576ca7f337a7f8ee605ec')
        self.assertEqual(reference['checkpoint_repository'], 'starvector/starvector-1b-im2svg')
        self.assertEqual(reference['config_sha256'], oracle.digest(artifact))
        self.assertEqual(lock['required_packages']['torch'], '2.7.1+cu128')
        self.assertEqual(lock['required_packages']['torchvision'], '0.22.1+cu128')

    def check_package_versions(self, versions, error=None):
        class ReachedCheckpointValidation(Exception):
            pass
        args = SimpleNamespace(upstream_root=self.root, weights_root=self.root)
        def installed(name):
            if name not in versions:
                raise oracle.importlib.metadata.PackageNotFoundError(name)
            return versions[name]
        with patch.object(oracle, 'source_identity', return_value='source-verified'), \
             patch.object(oracle.importlib.metadata, 'version', side_effect=installed), \
             patch.object(oracle, 'local_file', side_effect=ReachedCheckpointValidation) as checkpoint:
            if error is None:
                with self.assertRaises(ReachedCheckpointValidation):
                    oracle.validate(args)
                checkpoint.assert_called_once()
            else:
                with self.assertRaisesRegex(ValueError, error):
                    oracle.validate(args)
                checkpoint.assert_not_called()

    def test_exact_locked_package_versions_reach_checkpoint_validation(self):
        versions = json.loads(oracle.LOCK.read_text())['required_packages']
        self.check_package_versions(versions)

    def test_wrong_cuda_build_or_package_version_fails_before_checkpoint_access(self):
        locked = json.loads(oracle.LOCK.read_text())['required_packages']
        for name, wrong in [('torch', '2.7.1'), ('torch', '2.7.1+cpu'),
                            ('torch', '2.7.1+cu126'), ('torch', '2.7.0+cu128'),
                            ('torchvision', '0.22.1'), ('torchvision', '0.22.1+cu126'),
                            ('transformers', '4.49.1'), ('transformers', '4.49.0+custom')]:
            with self.subTest(package=name, installed=wrong):
                self.check_package_versions({**locked, name: wrong}, 'package version mismatch: ' + name)

    def test_missing_locked_package_fails_before_checkpoint_access(self):
        versions = json.loads(oracle.LOCK.read_text())['required_packages']
        del versions['torchvision']
        self.check_package_versions(versions, 'missing validation-only package: torchvision')

    @unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('transformers'), 'requires existing CPU torch/transformers environment')
    def test_real_hf_generate_boundary_and_case_deadline_for_both_backbones(self):
        import torch
        from PIL import Image
        from transformers import GPTBigCodeConfig, GPTBigCodeForCausalLM, Starcoder2Config, Starcoder2ForCausalLM
        image = self.root / 'input.png'; Image.new('RGB', (16, 16), 'white').save(image)
        row = {'seed': 3, 'input_png': str(image), 'sampling': {}, 'detail_budget': {'maxNewTokens': 3, 'maxSvgBytes': 1024, 'maxWallTimeMs': 120000}}
        configurations = [
            (GPTBigCodeConfig(vocab_size=32, n_embd=16, n_layer=1, n_head=2, n_positions=64, bos_token_id=1, eos_token_id=None, pad_token_id=0), GPTBigCodeForCausalLM),
            (Starcoder2Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64, bos_token_id=1, eos_token_id=None, pad_token_id=0), Starcoder2ForCausalLM)]
        for config, constructor in configurations:
            lm = constructor(config).eval()
            class Wrapper:
                model = SimpleNamespace(processor=lambda **kwargs: {'pixel_values': torch.zeros(1,3,16,16)}, svg_transformer=SimpleNamespace(transformer=lm))
                def generate_im2svg(self, batch, **kwargs):
                    output = lm.generate(inputs_embeds=torch.zeros(1,19,16), attention_mask=torch.ones(1,19,dtype=torch.long), do_sample=False, num_beams=1, max_length=kwargs['max_length'], pad_token_id=0)
                    self.output_length = output.shape[1]
                    return ['<svg xmlns="http://www.w3.org/2000/svg"/>']
            wrapper = Wrapper()
            _, observed = oracle.generate(wrapper, row, torch.device('cpu'))
            self.assertEqual(observed['prefix_length'], 19)
            self.assertEqual(observed['generated_tokens'], 3)
            self.assertEqual(wrapper.output_length, 3)
            with patch.object(oracle.time, 'monotonic', side_effect=[0.0] + [1000.0]*100):
                with self.assertRaisesRegex(ValueError, 'case wall-time budget'):
                    oracle.generate(wrapper, row, torch.device('cpu'))

    def rows(self):
        rows = []
        for index in range(120):
            path = self.root / ('%s.png' % index)
            path.write_bytes(('distinct PNG fixture %s' % index).encode())
            rows.append({'case_index': index, 'input_png_path': path.name, 'png_sha256': oracle.digest(path),
                         'sampling': {'temperature': 0.0, 'topP': 1.0, 'topK': 1, 'repetitionPenalty': 1.0, 'seed': index},
                         'detail_budget': {'maxNewTokens': 4000, 'maxSvgBytes': 262144, 'maxWallTimeMs': 120000}})
        self.save_rows(rows)
        return rows

    def save_rows(self, rows):
        (self.root / 'starvector-terminal-row-index-v1.json').write_text(json.dumps({'rows': rows}))

    def test_exact_balanced_twenty_rows_and_seed_identity(self):
        self.rows()
        result = oracle.select_rows(self.root)
        self.assertEqual([r['source_case_index'] for r in result], [*range(5), *range(30, 35), *range(60, 65), *range(90, 95)])
        self.assertEqual([r['seed'] for r in result], list(range(20)))
        self.assertEqual(result[10]['detail_budget']['maxNewTokens'], 4000)

    def test_duplicate_images_are_rejected_across_distinct_source_rows(self):
        rows = self.rows()
        rows[30]['input_png_path'] = rows[0]['input_png_path']
        rows[30]['png_sha256'] = rows[0]['png_sha256']
        self.save_rows(rows)
        with self.assertRaisesRegex(ValueError, 'distinct'):
            oracle.select_rows(self.root)

    def test_changed_input_bytes_are_rejected(self):
        self.rows()
        (self.root / '60.png').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            oracle.select_rows(self.root)

    def test_non_greedy_and_unsupported_sampling_are_rejected(self):
        for key, value in [('temperature', 0.1), ('topK', 2), ('topP', 0.9), ('repetitionPenalty', 1.1)]:
            rows = self.rows(); rows[0]['sampling'][key] = value; self.save_rows(rows)
            with self.assertRaisesRegex(ValueError, 'greedy'):
                oracle.select_rows(self.root)

    def test_bad_budget_and_row_order_are_rejected(self):
        rows = self.rows(); rows[0]['detail_budget']['maxNewTokens'] = True; self.save_rows(rows)
        with self.assertRaisesRegex(ValueError, 'token budget'):
            oracle.select_rows(self.root)
        rows[0], rows[1] = rows[1], rows[0]; self.save_rows(rows)
        with self.assertRaisesRegex(ValueError, 'ordered'):
            oracle.select_rows(self.root)

    def test_path_escape_and_symlink_are_rejected(self):
        (self.root / 'real').write_text('bytes')
        (self.root / 'link').symlink_to(self.root / 'real')
        for path in ['../real', '/real', 'link', 'a\\b']:
            with self.assertRaises(ValueError):
                oracle.local_file(self.root, path)

    def test_inventory_matches_native_json_byte_order(self):
        (self.root / 'z').write_bytes(b'z'); (self.root / 'A').write_bytes(b'a')
        expected = [{'path': p, 'byte_size': 1, 'sha256': hashlib.sha256(b).hexdigest()} for p, b in [('A', b'a'), ('z', b'z')]]
        self.assertEqual(oracle.inventory(self.root), hashlib.sha256(json.dumps(expected, separators=(',', ':')).encode()).hexdigest())

    def test_tied_head_is_the_only_missing_tensor_allowed(self):
        embedding = 'model.svg_transformer.transformer.transformer.wte.weight'
        head = 'model.svg_transformer.transformer.lm_head.weight'
        expected = {embedding: [16, 4], head: [16, 4], 'vision.weight': [2, 4]}
        observed = {embedding: [16, 4], 'vision.weight': [2, 4]}
        mapping = dict.fromkeys(observed, 'model.safetensors')
        self.assertEqual(oracle.check_tensor_coverage(expected, mapping, observed), {head: embedding})
        del observed['vision.weight']; del mapping['vision.weight']
        with self.assertRaisesRegex(ValueError, 'coverage'):
            oracle.check_tensor_coverage(expected, mapping, observed)

    def test_wrong_shapes_or_extra_tensor_cannot_claim_strict_loading(self):
        with self.assertRaisesRegex(ValueError, 'shape mismatch'):
            oracle.check_tensor_coverage({'x': [2, 4]}, {'x': 'm'}, {'x': [4, 2]})
        with self.assertRaisesRegex(ValueError, 'coverage'):
            oracle.check_tensor_coverage({'x': [2]}, {'x': 'm'}, {'x': [2], 'unused': [2]})

    def shard(self, header, payload=b'1234'):
        data = json.dumps(header).encode()
        (self.root / 'model.safetensors').write_bytes(struct.pack('<Q', len(data)) + data + payload)
        (self.root / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'x': 'model.safetensors'}}))

    def test_safetensors_headers_bind_all_shards_to_index(self):
        self.shard({'x': {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}})
        self.assertEqual(oracle.checkpoint_map(self.root), {'x': 'model.safetensors'})
        self.shard({'wrong': {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}})
        with self.assertRaisesRegex(ValueError, 'misindexed'):
            oracle.checkpoint_map(self.root)

    def test_truncated_safetensors_payload_is_rejected(self):
        self.shard({'x': {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 8]}})
        with self.assertRaisesRegex(ValueError, 'data range'):
            oracle.checkpoint_map(self.root)

    def test_source_hash_binds_actual_checkout_and_rejects_extra_python(self):
        directory = self.root / 'starvector'; directory.mkdir()
        (directory / 'source.py').write_text('def upstream(): return 1\n')
        entries = [{'path': 'starvector/source.py', 'sha256': oracle.digest(directory / 'source.py')}]
        lock = {'implementation_revision': 'a' * 40, 'python_source_sha256': hashlib.sha256(oracle.canonical(entries)).hexdigest()}
        with patch.object(oracle.subprocess, 'check_output', side_effect=['a' * 40, 'starvector/source.py\n']):
            self.assertEqual(oracle.source_identity(self.root, lock), lock['python_source_sha256'])
        (directory / 'source.py').write_bytes(b'def upstream(): return 1\r\n')
        changed = [{'path': 'starvector/source.py', 'sha256': oracle.digest(directory / 'source.py')}]
        changed_hash = hashlib.sha256(oracle.canonical(changed)).hexdigest()
        with patch.object(oracle.subprocess, 'check_output', side_effect=['a' * 40, 'starvector/source.py\n']):
            with self.assertRaisesRegex(ValueError, 'expected=' + lock['python_source_sha256'] + ' actual=' + changed_hash + ' files=1'):
                oracle.source_identity(self.root, lock)
        (directory / 'source.py').write_text('def upstream(): return 1\n')
        (directory / 'injected.py').write_text('raise RuntimeError()')
        with patch.object(oracle.subprocess, 'check_output', side_effect=['a' * 40, 'starvector/source.py\n']):
            with self.assertRaisesRegex(ValueError, 'untracked'):
                oracle.source_identity(self.root, lock)

    def test_completed_manifest_cannot_overwrite_prior_evidence(self):
        path = self.root / 'manifest.json'
        oracle.durable_json(path, {'completed': 20})
        with self.assertRaises(FileExistsError):
            oracle.durable_json(path, {'completed': 0})
        self.assertEqual(json.loads(path.read_text()), {'completed': 20})


if __name__ == '__main__':
    unittest.main()
