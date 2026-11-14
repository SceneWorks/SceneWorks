"""Network-free tests of the Qwen3.8 27B describer provisioning checks (no download, no model)."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

_spec = importlib.util.spec_from_file_location(
    "qwen38_describer_provision", Path(__file__).with_name("qwen38_describer_provision.py"))
provision = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(provision)

TIERS = {"q4": 30, "q8": 50}
ENTRIES = [
    (".gitattributes", 5), ("README.md", 7),
    ("q4/config.json", 10), ("q4/model.safetensors", 20),
    ("q8/config.json", 10), ("q8/model.safetensors", 40),
]


class ExpectedFilesTest(unittest.TestCase):
    def test_keeps_only_tier_files(self):
        self.assertEqual(provision.expected_files(ENTRIES, TIERS), {
            "q4/config.json": 10, "q4/model.safetensors": 20,
            "q8/config.json": 10, "q8/model.safetensors": 40,
        })

    def test_refuses_a_tier_total_that_differs_from_the_catalog(self):
        with self.assertRaisesRegex(RuntimeError, "q8/ .* totals 50 bytes, the catalog says 51"):
            provision.expected_files(ENTRIES, {"q4": 30, "q8": 51})

    def test_refuses_a_missing_tier_or_unsized_file(self):
        with self.assertRaisesRegex(RuntimeError, "no sized files under q4/"):
            provision.expected_files([e for e in ENTRIES if not e[0].startswith("q4/")], TIERS)
        with self.assertRaisesRegex(RuntimeError, "no sized files under q4/"):
            provision.expected_files([("q4/config.json", None), *ENTRIES[3:]], TIERS)

    def test_pins_an_exact_revision_and_both_quantised_tiers(self):
        self.assertRegex(provision.REVISION, r"^[0-9a-f]{40}$")
        self.assertEqual(sorted(provision.TIER_BYTES), ["q4", "q8"])


class MismatchedFilesTest(unittest.TestCase):
    def setUp(self):
        # Scratch beside this file, inside the checkout.
        self.scratch = tempfile.TemporaryDirectory(dir=Path(__file__).parent, prefix="qwen38-provision-unit-")
        self.cache = Path(self.scratch.name)
        self.snapshot = provision.snapshot_dir(self.cache)
        self.expected = provision.expected_files(ENTRIES, TIERS)

    def tearDown(self):
        self.scratch.cleanup()

    def write(self, name, size):
        path = self.snapshot.joinpath(*name.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * size)

    def test_snapshot_uses_the_standard_hub_layout(self):
        self.assertEqual(
            self.snapshot.relative_to(self.cache).parts,
            ("models--SceneWorks--qwen3.8-27b-abliterated-candle", "snapshots", provision.REVISION))

    def test_empty_cache_lists_every_file(self):
        self.assertEqual(len(provision.mismatched_files(self.snapshot, self.expected)), 4)

    def test_complete_snapshot_needs_nothing(self):
        for name, size in self.expected.items():
            self.write(name, size)
        self.assertEqual(provision.mismatched_files(self.snapshot, self.expected), [])

    def test_a_truncated_file_is_reported_with_its_size(self):
        for name, size in self.expected.items():
            self.write(name, size)
        self.write("q8/model.safetensors", 39)
        self.assertEqual(provision.mismatched_files(self.snapshot, self.expected),
                         [("q8/model.safetensors", 40, 39)])


class FreeSpaceTest(unittest.TestCase):
    def test_refuses_below_the_minimum(self):
        with self.assertRaisesRegex(RuntimeError, "69.9 GB free; provisioning needs at least 70 GB"):
            provision.require_free_space(69_900_000_000)

    def test_accepts_the_minimum(self):
        provision.require_free_space(provision.MIN_FREE_BYTES)


if __name__ == "__main__":
    unittest.main()
