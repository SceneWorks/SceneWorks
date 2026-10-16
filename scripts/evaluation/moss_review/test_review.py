"""CPU synthetic records only: fixtures do not constitute image or quality evidence."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import struct
import zlib
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('review', Path(__file__).with_name('review.py'))
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def fixture_png(seed=0, metadata=False):
    def chunk(kind, body):
        return struct.pack('>I', len(body)) + kind + body + struct.pack('>I', zlib.crc32(kind + body) & 0xffffffff)
    chunks = [chunk(b'IHDR', struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0))]
    if metadata: chunks.append(chunk(b'iTXt', b'workflow\x00\x00\x00\x00\x00generator identity private'))
    chunks += [chunk(b'IDAT', zlib.compress(bytes([0, seed % 256, 17, 33]))), chunk(b'IEND', b'')]
    return b'\x89PNG\r\n\x1a\n' + b''.join(chunks)


class ReviewPackets(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.run = self.root / 'run'; self.run.mkdir()
        self.public = self.root / 'public'; self.private = self.root / 'private'; self.evidence = self.root / 'evidence'; self.evidence.mkdir()
        self.manifest = r.ev.template(); self.lock = self.root / 'lock.json'
        r.save_new(self.lock, r.ev.seal(self.manifest))
        specs = {role: dict(identity=dict(reviewer=role + '-fresh-context', reviewer_kind='machine_assisted', model='synthetic-test-model', version='fixture-v1', tool_revision='test-only'), task_contexts=[role + '-fixture-task'], additional_instructions='CPU fixtures only.') for role in r.ROLES}
        spec_path = self.root / 'evaluators.json'; r.save_new(spec_path, specs)
        self.protocol_path = self.root / 'protocol.json'
        r.freeze(self.lock, self.run, spec_path, self.protocol_path)
        self.proto = r.load(self.protocol_path); self.protocol = r.ev.unseal(self.proto)
        self.results = {}
        for cell in r.ev.cells(self.manifest):
            out = self.run / (cell['id'] + '.png'); out.write_bytes(fixture_png(seed=sum(cell['id'].encode()), metadata=True))
            self.results[cell['id']] = dict(cell=cell, status='ok', output=dict(path=out.name, sha256=r.ev.file_hash(out)), elapsed_seconds=1,
                metrics=dict(cold_seconds=None, warm_seconds=1, latency_condition='resident_model', peak_gpu_bytes=10, gpu_memory_method='CPU fixture only', peak_host_rss_bytes=20, download_bytes=0))
        r.export_verified(self.proto, self.manifest, self.results, self.run, self.public, self.private)
        self.mapping = r.load(self.private / 'mapping.json'); self.ap = r.load(self.public / 'annotations.packet.json'); self.jp = r.load(self.public / 'judgments.packet.json')
        transcript = self.evidence / 'transcript.txt'; transcript.write_text('Synthetic CPU test transcript. No pixels were viewed or judged.')
        self.transcript = dict(path=transcript.name, sha256=r.ev.file_hash(transcript))
        self.raw = []; self.ratings = []; self.alignment = []
        by_case = {c['id']: c for c in self.manifest['corpus']['cases']}
        for image in r.ev.unseal(self.ap)['images']:
            private = r.ev.unseal(self.mapping)['images'][image['image_id']]
            case = by_case[self.results[private['cell_id']]['cell']['case_id']]
            self.raw.append(dict(image_id=image['image_id'], output_sha256=image['sha256'], text=['independently observed text', 'unexpected extra text'], boxes=[dict(observation_id='one', label='observed thing one', box=[.1,.1,.4,.4]), dict(observation_id='two', label='observed thing two', box=[.5,.5,.9,.9])], **self.provenance('annotator')))
            indices = [0] + [None] * (len(case['expected_text']) - 1) if case['expected_text'] else []
            self.alignment.append(dict(image_id=image['image_id'], text_indices=indices, box_ids={b['id']: ['one', 'two'][i] for i,b in enumerate(case['boxes'])}, **self.provenance('custodian')))
        image_hashes = {i['image_id']: i['sha256'] for i in r.ev.unseal(self.jp)['images']}
        for pair in r.ev.unseal(self.jp)['pairs']:
            self.ratings.append(dict(pair_id=pair['pair_id'], winner='B', scores={a: {k: 3 if a == 'A' else 4 for k in r.AXES} for a in ('A','B')}, viewed_image_hashes={a: image_hashes[pair['arms'][a]['image_id']] for a in ('A','B')}, **self.provenance('judge')))

    def provenance(self, role):
        return dict(evaluator=self.protocol['roles'][role]['identity'], instructions_sha256=self.protocol['roles'][role]['instructions_sha256'], transcript=self.transcript, task_context=role + '-fixture-task', uncertainty='high', notes='Synthetic CPU fixture, not a real judgment.')

    def translate(self, **changes):
        args = dict(manifest=self.manifest, results=self.results, protocol_doc=self.proto, mapping_doc=self.mapping, annotation_packet=self.ap, judge_packet=self.jp, raw=self.raw, ratings=self.ratings, alignments=self.alignment, public=self.public, evidence=self.evidence)
        args.update(changes)
        return r.translate_verified(**args)

    def test_packets_do_not_disclose_models_cells_or_annotation_answers(self):
        for packet in (self.ap, self.jp):
            data = json.dumps(packet)
            for forbidden in r.ev.CONFIG_IDS:
                self.assertNotIn(forbidden, data)
            self.assertNotIn('scored.', data)
        annotation = r.ev.unseal(self.ap)
        self.assertNotIn('prompt', annotation)
        self.assertNotIn('expected_text', json.dumps(annotation))
        self.assertNotIn('requirements', json.dumps(annotation))
        self.assertNotIn('boxes', json.dumps(annotation['images']))
        self.assertEqual(len(annotation['images']), 288)
        self.assertEqual(len(r.ev.unseal(self.jp)['pairs']), 432)
        self.assertEqual(len(list((self.public / 'annotation-batches').glob('*.json'))), 12)
        self.assertEqual(len(list((self.public / 'judge-batches').glob('*.json'))), 18)
        for batch in (self.public / 'judge-batches').glob('*.json'):
            self.assertLessEqual(len(r.sealed(batch)['pairs']), 24)

    def test_observed_text_survives_translation_and_legacy_report_accepts_pairs(self):
        annotations, judgments = self.translate()
        key = next(k for k in annotations if annotations[k]['text'])
        self.assertEqual(annotations[key]['text'][0], 'independently observed text')
        self.assertNotEqual(annotations[key]['output_sha256'], annotations[key]['review_image_sha256'])
        self.assertEqual(judgments[0]['rubric'], {k: 3 for k in r.AXES})
        pairs, secret = r.legacy_pairs(self.proto, self.mapping, self.jp, self.results, self.public)
        path = self.private / 'legacy-pairs.json'; r.save_new(path, pairs); r.save_new(str(path) + '.private.json', secret)
        ann_path = self.private / 'annotations.json'; ratings_path = self.private / 'judgments.json'
        r.save_new(ann_path, annotations); r.save_new(ratings_path, judgments)
        with patch.object(r.ev, 'receipts', return_value=(self.manifest, self.results)):
            report = r.ev.report(self.lock, self.run, path, ratings_path, ann_path)
        self.assertEqual(report['blind_pairs'], 432)
        self.assertFalse(report['recommendation_eligible'])  # no real DMD parity fixture evidence
        tables = r.summaries(self.manifest, self.results, self.mapping, self.ratings, annotations)
        self.assertEqual(len(tables['tables']), 108)
        self.assertTrue(all(t['comparisons'] == 4 for t in tables['tables']))
        self.assertTrue(all(v['measured_summary']['cold_seconds']['median'] is None for t in tables['tables'] for v in t['costs'].values()))

    def test_complete_translation_writes_hash_bound_private_artifacts(self):
        paths = [self.private / name for name in ('observations.json', 'ratings.json', 'alignments.json')]
        for path, value in zip(paths, (self.raw, self.ratings, self.alignment)): r.save_new(path, value)
        out = self.root / 'translated'
        with patch.object(r.ev, 'receipts', return_value=(self.manifest, self.results)):
            r.validate_translate(self.lock, self.run, self.protocol_path, self.public, self.private, *paths, self.evidence, out)
        proof = r.sealed(out / 'review-receipt.json')
        self.assertEqual(proof['actual_judgments'], 432)
        self.assertEqual(proof['independent_image_observations'], 288)
        for name, digest in proof['outputs'].items(): self.assertEqual(r.ev.file_hash(out / name), digest)
        self.assertEqual(len(r.load(out / 'analyst-tables.json')['tables']), 108)

    def test_missing_duplicate_and_remapped_judgments_fail(self):
        for ratings in (self.ratings[:-1], self.ratings[:-1] + [self.ratings[0]]):
            with self.assertRaisesRegex(ValueError, '432 unique'): self.translate(ratings=ratings)
        bad = copy.deepcopy(r.ev.unseal(self.mapping)); ids = list(bad['pairs']); bad['pairs'][ids[0]] = bad['pairs'][ids[1]]
        with self.assertRaisesRegex(ValueError, 'matchup coverage'): self.translate(mapping_doc=r.ev.seal(bad))

    def test_changed_evaluator_instruction_transcript_and_viewed_hash_fail(self):
        for field, value in (('instructions_sha256', '0'*64), ('evaluator', dict(self.ratings[0]['evaluator'], version='unfrozen'))):
            ratings = copy.deepcopy(self.ratings); ratings[0][field] = value
            with self.assertRaises(ValueError): self.translate(ratings=ratings)
        ratings = copy.deepcopy(self.ratings); ratings[0]['viewed_image_hashes']['A'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'viewed image hash'): self.translate(ratings=ratings)
        (self.evidence / 'transcript.txt').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'artifact hash mismatch'): self.translate()

    def test_missing_annotation_and_answer_fabricating_alignment_fail(self):
        with self.assertRaisesRegex(ValueError, 'independent observations'): self.translate(raw=self.raw[:-1])
        alignments = copy.deepcopy(self.alignment)
        target = next(a for a in alignments if a['text_indices']); target['text_indices'][0] = 100
        with self.assertRaisesRegex(ValueError, 'text observation'): self.translate(alignments=alignments)

    def test_missing_arm_is_unavailable_not_a_win(self):
        pair = r.ev.unseal(self.jp)['pairs'][0]; mapping = r.ev.unseal(self.mapping)
        cell = mapping['pairs'][pair['pair_id']]['A']; self.results[cell]['status'] = 'failed'
        with self.assertRaisesRegex(ValueError, 'availability changed'): self.translate()

    def test_reproducible_failed_outputs_remain_unavailable(self):
        results = copy.deepcopy(self.results)
        for row in results.values(): row['status'] = 'failed'
        public, private = self.root / 'failed-public', self.root / 'failed-private'
        r.export_verified(self.proto, self.manifest, results, self.run, public, private)
        ap = r.load(public / 'annotations.packet.json'); jp = r.load(public / 'judgments.packet.json'); mapping = r.load(private / 'mapping.json')
        ratings = [dict(pair_id=p['pair_id'], winner='unavailable', scores=dict(A=None, B=None), viewed_image_hashes=dict(A=None, B=None), **self.provenance('judge')) for p in r.ev.unseal(jp)['pairs']]
        ann, judgments = r.translate_verified(self.manifest, results, self.proto, mapping, ap, jp, [], ratings, [], public, self.evidence)
        self.assertEqual(ann, {})
        self.assertTrue(all(j['winner'] == 'unavailable' and 'rubric' not in j for j in judgments))
        ratings[0]['winner'] = 'A'
        with self.assertRaisesRegex(ValueError, 'cannot become win'): r.translate_verified(self.manifest, results, self.proto, mapping, ap, jp, [], ratings, [], public, self.evidence)

    def test_checkpoint_merge_refuses_duplicate_records(self):
        one, two = self.root / 'one.json', self.root / 'two.json'
        r.save_new(one, [dict(image_id='opaque-one')]); r.save_new(two, [dict(image_id='opaque-two')])
        r.merge_records('image_id', [one, two], self.root / 'merged.json')
        with self.assertRaisesRegex(ValueError, 'duplicate checkpoint'): r.merge_records('image_id', [one, one], self.root / 'duplicate.json')

    def test_actual_contexts_cannot_cross_roles_or_change_frozen_identity(self):
        raw, ratings, alignments = copy.deepcopy((self.raw, self.ratings, self.alignment))
        for record in raw + ratings + alignments:
            record['task_context'] = 'same-prompt-aware-context'
        with self.assertRaisesRegex(ValueError, 'context reused across'): self.translate(raw=raw, ratings=ratings, alignments=alignments)
        raw = copy.deepcopy(self.raw); raw[0]['task_context'] = 'undeclared-annotation-context'
        with self.assertRaisesRegex(ValueError, 'not bound to frozen evaluator'): self.translate(raw=raw)
        specs = r.load(self.root / 'evaluators.json')
        specs['judge']['task_contexts'] = specs['annotator']['task_contexts']
        altered = self.root / 'overlap-evaluators.json'; r.save_new(altered, specs)
        with self.assertRaisesRegex(ValueError, 'context reused across'): r.freeze(self.lock, self.root / 'fresh-overlap-run', altered, self.root / 'overlap-freeze.json')

    def test_orphan_scored_output_prevents_late_freeze(self):
        outputs = self.run / 'outputs'; outputs.mkdir()
        (outputs / 'scored.moss_base.english_signage-1.173.png').write_bytes(b'orphan CPU fixture')
        with self.assertRaisesRegex(ValueError, 'orphan outputs'): r.freeze(self.lock, self.run, self.root / 'evaluators.json', self.root / 'orphan-freeze.json')

    def test_checked_empty_run_cannot_authorize_another_completed_run(self):
        other_run = self.root / 'empty-other-run'
        protocol_path = self.root / 'wrong-run-protocol.json'
        r.freeze(self.lock, other_run, self.root / 'evaluators.json', protocol_path)
        wrong = r.load(protocol_path)
        with self.assertRaisesRegex(ValueError, 'checked run root differs'): r.export_verified(wrong, self.manifest, self.results, self.run, self.root / 'wrong-public', self.root / 'wrong-private')
        with self.assertRaisesRegex(ValueError, 'checked run root differs'): r.export(self.lock, self.run, protocol_path, self.root / 'wrong-public', self.root / 'wrong-private')
        with self.assertRaisesRegex(ValueError, 'checked run root differs'): r.validate_translate(self.lock, self.run, protocol_path, self.public, self.private, self.root / 'not-opened', self.root / 'not-opened', self.root / 'not-opened', self.evidence, self.root / 'wrong-out')
        bad = r.ev.unseal(self.mapping); bad['run_root'] = str(other_run)
        with self.assertRaisesRegex(ValueError, 'mapping run root differs'): self.translate(mapping_doc=r.ev.seal(bad))

    def test_pixel_decoder_agrees_across_all_png_row_filters(self):
        def chunk(kind, body):
            return struct.pack('>I', len(body)) + kind + body + struct.pack('>I', zlib.crc32(kind + body) & 0xffffffff)
        header = struct.pack('>IIBBBBB', 2, 2, 8, 2, 0, 0, 0)
        # Hand-calculated residual rows for the SAME two RGB rows under PNG filters 0-4.
        rows = [
            ([10,20,30,50,60,70], [15,25,35,60,70,80]),
            ([10,20,30,40,40,40], [15,25,35,45,45,45]),
            ([10,20,30,50,60,70], [5,5,5,10,10,10]),
            ([10,20,30,45,50,55], [10,15,20,28,28,28]),
            ([10,20,30,40,40,40], [5,5,5,10,10,10]),
        ]
        hashes = []
        for kind, (first, second) in enumerate(rows):
            data = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', header) + chunk(b'IDAT', zlib.compress(bytes([kind] + first + [kind] + second))) + chunk(b'IEND', b'')
            hashes.append(r.png_pixels(data)[1])
        self.assertEqual(len(set(hashes)), 1)

    def test_metadata_derivative_preserves_pixels_and_private_originals(self):
        for image, binding in list(r.ev.unseal(self.mapping)['images'].items())[:3]:
            original = self.run / self.results[binding['cell_id']]['output']['path']
            reviewed = self.public / 'images' / (image + '.png')
            self.assertIn(b'generator identity private', original.read_bytes())
            self.assertNotIn(b'generator identity private', reviewed.read_bytes())
            self.assertEqual(r.png_pixels(original.read_bytes())[1], r.png_pixels(reviewed.read_bytes())[1])
            self.assertEqual(r.ev.file_hash(original), binding['original_sha256'])
            self.assertEqual(r.ev.file_hash(reviewed), binding['review_sha256'])
            self.assertEqual(binding['removed_chunks'], ['iTXt'])

    def test_changed_pixels_and_public_packet_fail(self):
        image = r.ev.unseal(self.ap)['images'][0]
        (self.public / image['path']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'artifact hash mismatch'): self.translate()

    def test_freeze_after_scored_and_overlapping_public_private_refused(self):
        receipts = self.run / 'receipts'; receipts.mkdir(); (receipts / 'scored.anything.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'precede all scored'): r.freeze(self.lock, self.run, self.root / 'evaluators.json', self.root / 'late.json')
        with self.assertRaisesRegex(ValueError, 'disjoint'): r.export_verified(self.proto, self.manifest, self.results, self.run, self.root / 'new', self.root / 'new' / 'private')


if __name__ == '__main__': unittest.main()
