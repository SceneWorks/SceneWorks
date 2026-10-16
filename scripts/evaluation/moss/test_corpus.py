"""Source-truth annotation regressions; no image generation or optional packages."""
import json
from pathlib import Path
import unittest


class CorpusAnnotations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = json.loads(Path(__file__).with_name('corpus.json').read_text())
        cls.cases = {case['id']: case for case in cls.corpus['cases']}

    def test_object_constraints_reach_every_prompt_representation(self):
        for case_id, clause, criterion in (
            ('json_objects-1', 'Use a plain grey background.', 'Plain grey background'),
            ('json_objects-2', 'Include no extra objects.', 'No extra objects'),
        ):
            case = self.cases[case_id]
            self.assertIn(criterion, case['requirements'])
            self.assertIn(clause, case['layout_spec']['summary']['caption'])
            for key in ('prompt_raw', 'prompt_enhanced'):
                prose, encoded = case[key].split(' Final layout JSON: ', 1)
                self.assertIn(clause, prose)
                self.assertEqual(json.loads(encoded), case['layout_spec'])

    def test_long_prompts_score_early_and_late_requirements(self):
        expected = {
            'long_detail-1': (
                'Wide isometric coastal observatory cutaway at dawn',
                'Copper roof seams, weathered wooden floorboards and ivy on left exterior wall',
                'No writing or logos',
            ),
            'long_detail-2': (
                'Overhead editorial photograph of craft workbench',
                'Every object separate and physically plausible',
                'No hands, text, watermark or decorative border',
            ),
        }
        for case_id, criteria in expected.items():
            for criterion in criteria:
                self.assertIn(criterion, self.cases[case_id]['requirements'])

    def test_explicit_text_exclusions_have_blind_checklist_criteria(self):
        for case in self.cases.values():
            prompt = case['prompt_raw']
            criterion = next((clause for clause in (
                'No other words', 'No other text', 'No additional text',
                'No writing or logos', 'No hands, text, watermark or decorative border',
                'No branding or text', 'No text',
            ) if clause.lower() in prompt.lower()), None)
            if criterion is not None:
                self.assertIn(criterion, case['requirements'], case['id'])


if __name__ == '__main__':
    unittest.main()
