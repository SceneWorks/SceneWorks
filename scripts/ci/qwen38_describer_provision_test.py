"""Network-free tests of the Qwen3.8 27B describer provisioning checks (no download, no model)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

_spec = importlib.util.spec_from_file_location(
    "qwen38_describer_provision", Path(__file__).with_name("qwen38_describer_provision.py"))
provision = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(provision)

TIERS = {"q4": 40, "q8": 60}
FOLDERS = {"q4": "q4", "q8": "q8"}
NAMES = ("config.json", "model.safetensors.index.json", "model-*.safetensors")
ENTRIES = [
    (".gitattributes", 5), ("README.md", 7),
    # A sidecar the preparer copied that the row does not name: neither downloaded nor totalled.
    ("q4/special_tokens_map.json", 3),
    ("q4/config.json", 10), ("q4/model.safetensors.index.json", 4),
    ("q4/model-00001-of-00002.safetensors", 16), ("q4/model-00002-of-00002.safetensors", 10),
    ("q8/config.json", 10), ("q8/model.safetensors.index.json", 4),
    ("q8/model-00001-of-00002.safetensors", 26), ("q8/model-00002-of-00002.safetensors", 20),
]


class ExpectedFilesTest(unittest.TestCase):
    def test_keeps_only_the_files_the_rows_name(self):
        self.assertEqual(provision.expected_files(ENTRIES, TIERS, FOLDERS, NAMES), {
            "q4/config.json": 10, "q4/model.safetensors.index.json": 4,
            "q4/model-00001-of-00002.safetensors": 16, "q4/model-00002-of-00002.safetensors": 10,
            "q8/config.json": 10, "q8/model.safetensors.index.json": 4,
            "q8/model-00001-of-00002.safetensors": 26, "q8/model-00002-of-00002.safetensors": 20,
        })

    def test_refuses_a_tier_total_that_differs_from_the_catalog(self):
        with self.assertRaisesRegex(RuntimeError, "q8/ .* totals 60 bytes, the catalog says 61"):
            provision.expected_files(ENTRIES, {"q4": 40, "q8": 61}, FOLDERS, NAMES)

    def test_refuses_a_missing_tier_named_file_shards_or_unsized_file(self):
        with self.assertRaisesRegex(RuntimeError, "no sized files under q4/"):
            provision.expected_files([e for e in ENTRIES if not e[0].startswith("q4/")], TIERS, FOLDERS, NAMES)
        with self.assertRaisesRegex(RuntimeError, "no sized files under q4/ for q4/model.safetensors.index.json"):
            provision.expected_files([e for e in ENTRIES if e[0] != "q4/model.safetensors.index.json"],
                                     TIERS, FOLDERS, NAMES)
        with self.assertRaisesRegex(RuntimeError, "no sized files under q8/ for q8/model-\\*.safetensors"):
            provision.expected_files([e for e in ENTRIES if not e[0].startswith("q8/model-")],
                                     TIERS, FOLDERS, NAMES)
        with self.assertRaisesRegex(RuntimeError, "no sized files under q4/"):
            provision.expected_files([("q4/config.json", None), *ENTRIES[4:]], TIERS, FOLDERS, NAMES)

    def test_pins_an_exact_revision_and_both_quantised_tiers(self):
        self.assertRegex(provision.REVISION, r"^[0-9a-f]{40}$")
        self.assertEqual(sorted(provision.TIER_BYTES), ["q4", "q8"])
        self.assertEqual(provision.TIER_FOLDERS, {"q4": "q4-full", "q8": "q8-full"})


class MismatchedFilesTest(unittest.TestCase):
    def setUp(self):
        # Scratch beside this file, inside the checkout.
        self.scratch = tempfile.TemporaryDirectory(dir=Path(__file__).parent, prefix="qwen38-provision-unit-")
        self.cache = Path(self.scratch.name)
        self.snapshot = provision.snapshot_dir(self.cache)
        self.expected = provision.expected_files(ENTRIES, TIERS, FOLDERS, NAMES)

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
        self.assertEqual(len(provision.mismatched_files(self.snapshot, self.expected)), 8)

    def test_complete_snapshot_needs_nothing(self):
        for name, size in self.expected.items():
            self.write(name, size)
        self.assertEqual(provision.mismatched_files(self.snapshot, self.expected), [])

    def test_a_truncated_file_is_reported_with_its_size(self):
        for name, size in self.expected.items():
            self.write(name, size)
        self.write("q8/model-00001-of-00002.safetensors", 25)
        self.assertEqual(provision.mismatched_files(self.snapshot, self.expected),
                         [("q8/model-00001-of-00002.safetensors", 26, 25)])


BF16_ENTRIES = [
    (".gitattributes", 5), ("README.md", 7),
    *((name, 10) for name in provision.BF16_FILES),
    ("model-00001-of-00002.safetensors", 600), ("model-00002-of-00002.safetensors", 320),
    ("extra/model-00001.safetensors", 999),
]
BF16_TOTAL = 10 * len(provision.BF16_FILES) + 920


class Bf16Test(unittest.TestCase):
    def test_keeps_the_named_files_and_the_top_level_shards(self):
        expected = provision.bf16_expected_files(BF16_ENTRIES, estimated=BF16_TOTAL)
        self.assertEqual(sorted(expected), sorted([*provision.BF16_FILES, "model-00001-of-00002.safetensors",
                                                   "model-00002-of-00002.safetensors"]))
        self.assertEqual(sum(expected.values()), BF16_TOTAL)

    def test_refuses_a_missing_named_file_no_shards_or_a_total_off_the_estimate(self):
        without_index = [e for e in BF16_ENTRIES if e[0] != "model.safetensors.index.json"]
        with self.assertRaisesRegex(RuntimeError, "no sized model.safetensors.index.json"):
            provision.bf16_expected_files(without_index, estimated=BF16_TOTAL)
        without_shards = [e for e in BF16_ENTRIES if not e[0].startswith("model-")]
        with self.assertRaisesRegex(RuntimeError, "no sized model-\\*.safetensors shards"):
            provision.bf16_expected_files(without_shards, estimated=BF16_TOTAL)
        with self.assertRaisesRegex(RuntimeError, "the catalog estimates"):
            provision.bf16_expected_files(BF16_ENTRIES, estimated=BF16_TOTAL * 2)

    def test_matches_the_catalog_bf16_row(self):
        specs = json.loads((Path(__file__).parents[2] / "config/manifests/registry/specs.json").read_text())
        rows = [row for row in specs["vision_describe_qwen3_8_27b"]["product"]["downloads"]
                if row.get("variant") == "bf16"]
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual((row["repo"], row["revision"]), (provision.BF16_REPO, provision.BF16_REVISION))
        self.assertEqual(row["files"], [*provision.BF16_FILES, provision.BF16_SHARD_PATTERN])
        self.assertEqual(row["estimatedSizeBytes"], provision.BF16_ESTIMATED_BYTES)

    def test_quantised_repository_revision_and_totals_match_the_catalog_rows(self):
        specs = json.loads((Path(__file__).parents[2] / "config/manifests/registry/specs.json").read_text())
        rows = {row["variant"]: row for row in specs["vision_describe_qwen3_8_27b"]["product"]["downloads"]
                if row["repo"] == provision.REPO}
        self.assertEqual(sorted(rows), sorted(provision.TIER_BYTES))
        for tier, total in provision.TIER_BYTES.items():
            row = rows[tier]
            self.assertEqual(row["revision"], provision.REVISION, tier)
            self.assertEqual(row["estimatedSizeBytes"], total, tier)
            # The download fetches exactly the files the catalog row names.
            folder = provision.TIER_FOLDERS[tier]
            self.assertEqual(row["files"], provision.tier_patterns(folder), tier)

    def test_plans_keep_quantised_as_the_default_and_size_bf16_disk(self):
        repo, revision, _, read, patterns, minimum = provision.plan("quantised")
        self.assertEqual((repo, revision, read, patterns, minimum),
                         (provision.REPO, provision.REVISION, provision.expected_files,
                          [*provision.tier_patterns("q4-full"), *provision.tier_patterns("q8-full")], 70 * 10**9))
        repo, revision, _, read, patterns, minimum = provision.plan("bf16")
        self.assertEqual((repo, read, minimum), (provision.BF16_REPO, provision.bf16_expected_files, 75 * 10**9))
        self.assertEqual(patterns, [*provision.BF16_FILES, "model-*.safetensors"])
        with self.assertRaisesRegex(RuntimeError, "unknown provisioning mode"):
            provision.plan("q2")
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent, prefix="qwen38-provision-unit-") as cache:
            self.assertEqual(provision.snapshot_dir(Path(cache), repo, revision).relative_to(cache).parts,
                             ("models--huihui-ai--Huihui-Qwen3.8-27B-abliterated", "snapshots", revision))
        with self.assertRaisesRegex(RuntimeError, "74.9 GB free; provisioning needs at least 75 GB"):
            provision.require_free_space(74_900_000_000, provision.BF16_MIN_FREE_BYTES)


class FreeSpaceTest(unittest.TestCase):
    def test_refuses_below_the_minimum(self):
        with self.assertRaisesRegex(RuntimeError, "69.9 GB free; provisioning needs at least 70 GB"):
            provision.require_free_space(69_900_000_000)

    def test_accepts_the_minimum(self):
        provision.require_free_space(provision.MIN_FREE_BYTES)


if __name__ == "__main__":
    unittest.main()
