"""Stdlib-only tests for scripts/h3-finetune-terminal-float-variants.py.

Profiles come from the real casebook shape (`stagingAssets[<id>]`: 16 rows with
repo/revision/file/expectedBytes/expectedSha256/legal). The tests use a synthetic
casebook for the profile derivation, and a TINY two-shard profile (with an injected
pure-Python converter) for the conversion path - no numpy install, real weights, GPU,
model, browser or network. Every fake receipt is unit_fake and cannot be campaign proof.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "h3-finetune-terminal-float-variants.py"
_spec = importlib.util.spec_from_file_location("h3_finetune_terminal_float_variants", MODULE_PATH)
speech = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(speech)

REPO = "SceneWorks/minimax-h3-mlx"
REV = "137ce668c55a20bc0935fd1cf2a3de8448abb7f4"
PREFIX = "bf16/transformer"
SHARD_A = f"{PREFIX}/shard-a.safetensors"
SHARD_B = f"{PREFIX}/shard-b.safetensors"
CONFIG = f"{PREFIX}/config.json"
INDEX = f"{PREFIX}/diffusion_pytorch_model.safetensors.index.json"
LICENSE = "MiniMax-H3-Community-License-Agreement"
CREDIT = "MiniMax H3"
GHA = "a" * 40


def bf16_bytes(values) -> bytes:
    out = bytearray()
    for value in values:
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        out += struct.pack("<H", bits >> 16)
    return bytes(out)


def write_safetensors(path: Path, tensors, metadata=None) -> None:
    header = {}
    if metadata is not None:
        header["__metadata__"] = metadata
    offset = 0
    payload = bytearray()
    for name, dtype, shape, data in tensors:
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(data)]}
        payload += data
        offset += len(data)
    body = json.dumps(header, separators=(",", ":")).encode("utf-8")
    body += b" " * ((-(8 + len(body))) % 8)
    path.write_bytes(struct.pack("<Q", len(body)) + body + bytes(payload))


def read_safetensors(path: Path):
    raw = path.read_bytes()
    length = struct.unpack("<Q", raw[:8])[0]
    return json.loads(raw[8:8 + length]), raw[8 + length:]


def tensor_bytes(header, data, name) -> bytes:
    start, end = header[name]["data_offsets"]
    return data[start:end]


def fake_converter(calls):
    def convert(chunk, source_dtype, target_dtype):
        calls.append({"source": source_dtype, "target": target_dtype, "bytes": len(chunk)})
        if source_dtype != "BF16":
            return bytes(chunk), {"precisionLoss": 0, "underflow": 0, "nan": 0, "inf": 0}
        stats = {"precisionLoss": 0, "underflow": 0, "nan": 0, "inf": 0}
        out = bytearray()
        for index in range(0, len(chunk), 2):
            word = struct.unpack_from("<H", chunk, index)[0]
            bits = word << 16
            value = struct.unpack("<f", struct.pack("<I", bits))[0]
            if target_dtype == "F32":
                out += struct.pack("<I", bits)
                continue
            if target_dtype != "F16":
                raise speech.FloatVariantError("input", "bad target")
            try:
                half = struct.pack("<e", value)
            except OverflowError as exc:
                raise speech.FloatVariantError("overflow", f"finite BF16 value overflows F16: {exc}")
            rounded = struct.unpack("<e", half)[0]
            if math.isfinite(value) and math.isinf(rounded):
                raise speech.FloatVariantError("overflow", "finite BF16 value overflows F16")
            if math.isfinite(value) and rounded != value:
                stats["precisionLoss"] += 1
            if value != 0 and rounded == 0:
                stats["underflow"] += 1
            if math.isnan(value):
                stats["nan"] += 1
            if math.isinf(value):
                stats["inf"] += 1
            out += half
        return bytes(out), stats

    return convert


def casebook_with(set_id: str, *, prefix: str = PREFIX, legal=None) -> dict:
    names = [f"{prefix}/config.json"]
    names += [f"{prefix}/diffusion_pytorch_model-{index:05d}-of-00014.safetensors" for index in range(1, 15)]
    names.append(f"{prefix}/diffusion_pytorch_model.safetensors.index.json")
    rows = [
        {
            "repoType": "model",
            "repo": REPO,
            "revision": REV,
            "file": name,
            "expectedBytes": 1,
            "expectedSha256": "a" * 64,
            "legal": legal or {"licenseId": LICENSE, "inheritedLicenseId": LICENSE, "attributionRequired": CREDIT},
        }
        for name in names
    ]
    return {"version": 1, "kind": "preparation-not-campaign-evidence", "stagingAssets": {set_id: rows}}


def tiny_profile() -> dict:
    return {
        "setId": "builtin-shards",
        "repo": REPO,
        "revision": REV,
        "prefix": PREFIX,
        "partition": "FL2VA",
        "files": [SHARD_A, SHARD_B, CONFIG, INDEX],
        "rows": {},
    }


def make_source(root: str, *, shard_a=None, shard_b=None, index_map=None, config=None, mutate_receipt=None):
    profile = tiny_profile()
    base = Path(root) / "staged"
    (base / SHARD_A).parent.mkdir(parents=True, exist_ok=True)
    shard_a = shard_a or [
        ("w", "BF16", [5], bf16_bytes([1.0, -2.0, 0.5, 3.0, 1e-8])),
        ("freq", "F32", [2], struct.pack("<2f", 0.1, 0.2)),
    ]
    shard_b = shard_b or [
        ("aux", "I8", [3], bytes([1, 2, 3])),
        ("h", "F16", [2], struct.pack("<2e", 1.5, -0.25)),
    ]
    write_safetensors(base / SHARD_A, shard_a, {"format": "pt", "custom": "keep-me"})
    write_safetensors(base / SHARD_B, shard_b, {"format": "pt", "custom": "keep-me"})
    (base / CONFIG).write_text(json.dumps(config or {"_class_name": "FixedMiniMaxH3Transformer3DModel", "num_layers": 50}))
    weight_map = index_map if index_map is not None else {
        "w": os.path.basename(SHARD_A),
        "freq": os.path.basename(SHARD_A),
        "aux": os.path.basename(SHARD_B),
        "h": os.path.basename(SHARD_B),
    }
    total = sum(len(item[3]) for item in shard_a) + sum(len(item[3]) for item in shard_b)
    (base / INDEX).write_text(json.dumps({"metadata": {"total_size": total, "format": "pt"}, "weight_map": weight_map}))
    legal = {"licenseId": LICENSE, "inheritedLicenseId": LICENSE, "attributionRequired": CREDIT}
    rows = []
    for relative in profile["files"]:
        path = base / relative
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        profile["rows"][relative] = {"expectedBytes": len(data), "expectedSha256": digest, "legal": legal}
        rows.append({
            "repoType": "model",
            "repo": REPO,
            "revision": REV,
            "file": relative,
            "status": "completed",
            "actualBytes": len(data),
            "observedSha256": digest,
            "expectedSha256": digest,
            "finalPath": str(path),
            "expectedLegal": dict(legal),
            "explicitAcknowledgment": {"sha256": digest, "acknowledged": True, "licenseId": LICENSE, "inheritedLicenseId": LICENSE, "inheritedAcknowledged": True, "attribution": f"{CREDIT} license credit"},
        })
    receipt = {
        "version": 1,
        "kind": "h3-finetune-stage-receipt",
        "proof_kind": "frozen_stage_content",
        "settings": {"privateRoot": root},
        "set": {"ids": ["builtin-shards"]},
        "licenseAuthority": {
            "kind": "user-approved",
            "approvedBy": "coordinator",
            "approvedAt": "2026-10-25T00:00:00Z",
            "reference": "slack://approved-h3",
            "subjects": [{"sha256": "a" * 64, "acknowledged": True, "licenseId": LICENSE}],
            "acknowledged": True,
        },
        "status": "completed",
        "files": rows,
    }
    if mutate_receipt is not None:
        mutate_receipt(receipt)
    return receipt, profile


def stage(tmp: tempfile.TemporaryDirectory, *, chunk_bytes=4, **source_overrides):
    root = tmp.name
    receipt, profile = make_source(root, **source_overrides)
    calls = []
    defaults = {
        "profile_def": profile,
        "converter": fake_converter(calls),
        "chunk_bytes": chunk_bytes,
        "proof_kind": "unit_fake",
        "allow_unit_fake": True,
        "env": {"GITHUB_SHA": GHA},
    }
    result = speech.stage_float_variants(receipt, root, "builtin-shards", **defaults)
    result["_calls"] = calls
    result["_receipt_in"] = receipt
    result["_profile"] = profile
    return result


class FloatVariantTests(unittest.TestCase):
    def test_casebook_profile_derives_exact_16_file_closure(self):
        for set_id, partition in (
            ("builtin-shards", "FL2VA"),
            ("builtin-ref-shards", "Ref2VA"),
            ("community-shards-fl2va", "FL2VA"),
            ("community-shards-ref2va", "Ref2VA"),
        ):
            profile = speech.profile_from_casebook(casebook_with(set_id), set_id)
            self.assertEqual(profile["partition"], partition)
            self.assertEqual(len(profile["files"]), 16)
            self.assertEqual(profile["files"][0], f"{PREFIX}/config.json")
            self.assertEqual(profile["files"][1], f"{PREFIX}/diffusion_pytorch_model-00001-of-00014.safetensors")
            self.assertEqual(profile["files"][-2], f"{PREFIX}/diffusion_pytorch_model-00014-of-00014.safetensors")
            self.assertEqual(profile["files"][-1], f"{PREFIX}/diffusion_pytorch_model.safetensors.index.json")
            self.assertEqual(profile["rows"][profile["files"][0]]["legal"]["licenseId"], LICENSE)

    def test_real_casebook_sets_when_present(self):
        book = speech.load_casebook()
        assets = book.get("stagingAssets") or {}
        for set_id in speech.SOURCE_PROFILE_IDS:
            if not isinstance(assets.get(set_id), list):
                self.skipTest("this checkout's casebook predates the stagingAssets shard sets")
            profile = speech.profile_from_casebook(book, set_id)
            self.assertEqual(len(profile["files"]), 16)
            self.assertTrue(profile["prefix"].endswith(("transformer", "transformer_ref")))

    def test_packed_casebook_set_rejected(self):
        with self.assertRaises(speech.FloatVariantError) as caught:
            speech.profile_from_casebook(casebook_with("builtin-shards", prefix="q4/transformer"), "builtin-shards")
        self.assertEqual(caught.exception.code, "profile")

    def test_full_two_shard_conversion_preserves_keys_shapes_metadata_and_aux(self):
        with tempfile.TemporaryDirectory() as name:
            receipt = stage(SimpleNamespace(name=name))
            self.assertEqual(receipt["status"], "completed")
            for variant, dtype, histogram in (
                ("dense-f16", "F16", {"F16": 2, "F32": 1, "I8": 1}),
                ("dense-f32", "F32", {"F32": 2, "F16": 1, "I8": 1}),
            ):
                self.assertEqual(receipt["variants"][variant]["dtypeHistogram"], histogram)
                header_a, data_a = read_safetensors(Path(name) / "derived" / variant / os.path.basename(SHARD_A))
                header_b, data_b = read_safetensors(Path(name) / "derived" / variant / os.path.basename(SHARD_B))
                self.assertEqual(header_a["__metadata__"], {"format": "pt", "custom": "keep-me"})
                self.assertEqual(header_b["__metadata__"], {"format": "pt", "custom": "keep-me"})
                self.assertEqual((header_a["w"]["dtype"], header_a["w"]["shape"]), (dtype, [5]))
                self.assertEqual((header_a["freq"]["dtype"], header_a["freq"]["shape"]), ("F32", [2]))
                self.assertEqual((header_b["aux"]["dtype"], header_b["aux"]["shape"]), ("I8", [3]))
                self.assertEqual((header_b["h"]["dtype"], header_b["h"]["shape"]), ("F16", [2]))
                source_a, source_data_a = read_safetensors(Path(name) / "staged" / SHARD_A)
                source_b, source_data_b = read_safetensors(Path(name) / "staged" / SHARD_B)
                self.assertEqual(tensor_bytes(header_a, data_a, "freq"), tensor_bytes(source_a, source_data_a, "freq"))
                self.assertEqual(tensor_bytes(header_b, data_b, "h"), tensor_bytes(source_b, source_data_b, "h"))
                self.assertEqual(tensor_bytes(header_b, data_b, "aux"), tensor_bytes(source_b, source_data_b, "aux"))

    def test_bf16_to_f32_is_exact_bit_shift_and_f16_values_are_exact(self):
        with tempfile.TemporaryDirectory() as name:
            receipt = stage(SimpleNamespace(name=name))
            source_a, source_data = read_safetensors(Path(name) / "staged" / SHARD_A)
            header32, data32 = read_safetensors(Path(name) / "derived" / "dense-f32" / os.path.basename(SHARD_A))
            words = tensor_bytes(source_a, source_data, "w")
            expected32 = b"".join(struct.pack("<I", struct.unpack_from("<H", words, index)[0] << 16) for index in range(0, len(words), 2))
            self.assertEqual(tensor_bytes(header32, data32, "w"), expected32)
            header16, data16 = read_safetensors(Path(name) / "derived" / "dense-f16" / os.path.basename(SHARD_A))
            expected16 = b"".join(struct.pack("<e", value) for value in (1.0, -2.0, 0.5, 3.0, 0.0))
            self.assertEqual(tensor_bytes(header16, data16, "w"), expected16)
            stats = receipt["variants"]["dense-f16"]["conversionStats"]
            self.assertEqual((stats["precisionLoss"], stats["underflow"]), (1, 1))
            self.assertEqual(receipt["variants"]["dense-f32"]["conversionStats"]["precisionLoss"], 0)
            self.assertTrue(receipt["variants"]["dense-f32"]["conversionStats"]["exact"])

    def test_index_weight_map_complete_and_total_size_sums_both_shards(self):
        with tempfile.TemporaryDirectory() as name:
            receipt = stage(SimpleNamespace(name=name))
            source_index = json.loads((Path(name) / "staged" / INDEX).read_text())
            for variant, expected in (("dense-f16", 25), ("dense-f32", 35)):
                index = json.loads((Path(name) / "derived" / variant / os.path.basename(INDEX)).read_text())
                self.assertEqual(index["weight_map"], source_index["weight_map"])
                self.assertEqual(index["metadata"]["total_size"], expected)
                all_names = set()
                total = 0
                for shard in (os.path.basename(SHARD_A), os.path.basename(SHARD_B)):
                    header, data = read_safetensors(Path(name) / "derived" / variant / shard)
                    all_names.update(name for name in header if name != "__metadata__")
                    total += sum(end - start for start, end in (entry["data_offsets"] for key, entry in header.items() if key != "__metadata__"))
                self.assertEqual(all_names, set(source_index["weight_map"]))
                self.assertEqual(total, expected)
                self.assertEqual(receipt["variants"][variant]["tensorBytes"], expected)

    def test_duplicate_tensor_across_shards_refused(self):
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaises(speech.FloatVariantError) as caught:
                stage(SimpleNamespace(name=name), shard_b=[("w", "BF16", [1], bf16_bytes([1.0])), ("h", "F16", [1], struct.pack("<e", 1.0))])
            self.assertEqual(caught.exception.code, "input")
            self.assertIn("duplicate tensor", caught.exception.message)

    def test_missing_tensor_refused(self):
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaises(speech.FloatVariantError) as caught:
                stage(SimpleNamespace(name=name), index_map={"w": os.path.basename(SHARD_A), "freq": os.path.basename(SHARD_A), "aux": os.path.basename(SHARD_B), "h": os.path.basename(SHARD_B), "ghost": os.path.basename(SHARD_A)})
            self.assertEqual(caught.exception.code, "input")
            self.assertIn("absent from the shards", caught.exception.message)

    def test_extra_shard_tensor_refused(self):
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaises(speech.FloatVariantError) as caught:
                stage(SimpleNamespace(name=name), index_map={"w": os.path.basename(SHARD_A), "freq": os.path.basename(SHARD_A), "aux": os.path.basename(SHARD_B)})
            self.assertEqual(caught.exception.code, "input")
            self.assertIn("absent from the index", caught.exception.message)

    def test_output_files_are_bounded_and_no_full_tensor(self):
        with tempfile.TemporaryDirectory() as name:
            receipt = stage(SimpleNamespace(name=name), chunk_bytes=4)
            self.assertTrue(all(call["bytes"] <= 4 and call["bytes"] % 2 == 0 for call in receipt["_calls"]))
            self.assertGreater(len([call for call in receipt["_calls"] if call["target"] == "F16"]), 1)
            self.assertLess(max(call["bytes"] for call in receipt["_calls"]), 10)
            for variant in ("dense-f16", "dense-f32"):
                for shard in (os.path.basename(SHARD_A), os.path.basename(SHARD_B)):
                    path = Path(name) / "derived" / variant / shard
                    raw = path.read_bytes()
                    length = struct.unpack("<Q", raw[:8])[0]
                    header = json.loads(raw[8:8 + length])
                    expected = 8 + length + sum(end - start for start, end in (entry["data_offsets"] for key, entry in header.items() if key != "__metadata__"))
                    self.assertEqual(len(raw), expected)

    def test_source_f32_f16_i8_bytes_and_shas_unchanged(self):
        with tempfile.TemporaryDirectory() as name:
            stage(SimpleNamespace(name=name))
            source_a, source_data_a = read_safetensors(Path(name) / "staged" / SHARD_A)
            source_b, source_data_b = read_safetensors(Path(name) / "staged" / SHARD_B)
            for variant in ("dense-f16", "dense-f32"):
                header_a, data_a = read_safetensors(Path(name) / "derived" / variant / os.path.basename(SHARD_A))
                header_b, data_b = read_safetensors(Path(name) / "derived" / variant / os.path.basename(SHARD_B))
                self.assertEqual(hashlib.sha256(tensor_bytes(header_a, data_a, "freq")).hexdigest(), hashlib.sha256(tensor_bytes(source_a, source_data_a, "freq")).hexdigest())
                self.assertEqual(hashlib.sha256(tensor_bytes(header_b, data_b, "h")).hexdigest(), hashlib.sha256(tensor_bytes(source_b, source_data_b, "h")).hexdigest())
                self.assertEqual(hashlib.sha256(tensor_bytes(header_b, data_b, "aux")).hexdigest(), hashlib.sha256(tensor_bytes(source_b, source_data_b, "aux")).hexdigest())

    def test_explicit_authority_and_acknowledgments_unchanged(self):
        with tempfile.TemporaryDirectory() as name:
            receipt = stage(SimpleNamespace(name=name))
            source_receipt = receipt["_receipt_in"]
            self.assertEqual(receipt["source"]["licenseAuthority"], {k: source_receipt["licenseAuthority"][k] for k in ("kind", "approvedBy", "approvedAt", "reference", "subjects", "acknowledged")})
            for output_row, input_row in zip(receipt["source"]["files"], source_receipt["files"]):
                self.assertEqual(output_row["expectedLegal"], input_row["expectedLegal"])
                self.assertEqual(output_row["explicitAcknowledgment"], {k: input_row["explicitAcknowledgment"][k] for k in ("sha256", "acknowledged", "licenseId", "inheritedLicenseId", "inheritedAcknowledged", "attribution")})

    def test_finite_overflow_refused_and_failed_receipt_written(self):
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaises(speech.FloatVariantError) as caught:
                stage(SimpleNamespace(name=name), shard_a=[("w", "BF16", [1], bf16_bytes([70000.0])), ("freq", "F32", [1], struct.pack("<f", 0.5))], shard_b=[("aux", "I8", [1], b"\x01")], index_map={"w": os.path.basename(SHARD_A), "freq": os.path.basename(SHARD_A), "aux": os.path.basename(SHARD_B)})
            self.assertEqual(caught.exception.code, "overflow")
            written = json.loads((Path(name) / "derived" / "float-variants-receipt.json").read_text())
            self.assertEqual(written["status"], "failed")
            self.assertEqual(written["failure"]["code"], "overflow")

    def test_source_hash_mismatch_refused(self):
        with tempfile.TemporaryDirectory() as name:
            receipt, profile = make_source(name)
            path = Path(name) / "staged" / SHARD_A
            path.write_bytes(path.read_bytes() + b"tampered")
            with self.assertRaises(speech.FloatVariantError) as caught:
                speech.stage_float_variants(receipt, name, "builtin-shards", profile_def=profile, converter=fake_converter([]), chunk_bytes=4, allow_unit_fake=True, env={"GITHUB_SHA": GHA})
            self.assertEqual(caught.exception.code, "input")

    def test_unit_fake_refused_and_profile_override_requires_test_flag(self):
        with tempfile.TemporaryDirectory() as name:
            receipt, profile = make_source(name)
            receipt["proof_kind"] = "unit_fake"
            with self.assertRaises(speech.FloatVariantError) as caught:
                speech.stage_float_variants(receipt, name, "builtin-shards", profile_def=profile, converter=fake_converter([]), chunk_bytes=4, allow_unit_fake=False, env={"GITHUB_SHA": GHA})
            self.assertEqual(caught.exception.code, "proof")
        with tempfile.TemporaryDirectory() as name:
            receipt, profile = make_source(name)
            with self.assertRaises(speech.FloatVariantError) as caught:
                speech.stage_float_variants(receipt, name, "builtin-shards", profile_def=profile, converter=None, chunk_bytes=4, allow_unit_fake=True, env={"GITHUB_SHA": GHA})
            self.assertEqual(caught.exception.code, "profile")

    def test_chunk_bytes_must_be_even_and_bounded(self):
        for value in (3, 0, speech.MAX_CHUNK_BYTES + 2):
            with tempfile.TemporaryDirectory() as name:
                with self.assertRaises(speech.FloatVariantError) as caught:
                    stage(SimpleNamespace(name=name), chunk_bytes=value)
                self.assertEqual(caught.exception.code, "input")

    def test_output_dirs_must_be_new(self):
        with tempfile.TemporaryDirectory() as name:
            (Path(name) / "derived").mkdir()
            with self.assertRaises(speech.FloatVariantError) as caught:
                stage(SimpleNamespace(name=name))
            self.assertEqual(caught.exception.code, "output")

    def test_source_final_path_outside_root_refused(self):
        with tempfile.TemporaryDirectory() as name, tempfile.TemporaryDirectory() as outside:
            receipt, profile = make_source(name)
            receipt["files"][0]["finalPath"] = str(Path(outside) / "shard.safetensors")
            Path(receipt["files"][0]["finalPath"]).write_bytes(bf16_bytes([1.0]))
            with self.assertRaises(speech.FloatVariantError) as caught:
                speech.stage_float_variants(receipt, name, "builtin-shards", profile_def=profile, converter=fake_converter([]), chunk_bytes=4, allow_unit_fake=True, env={"GITHUB_SHA": GHA})
            self.assertEqual(caught.exception.code, "path")

    def test_numpy_is_imported_only_lazily_and_version_checked(self):
        self.assertNotIn("numpy", sys.modules)
        with mock.patch.dict(sys.modules, {"numpy": None}):
            with self.assertRaises(speech.FloatVariantError) as caught:
                speech._numpy_version()
            self.assertEqual(caught.exception.code, "tool")
            self.assertIn(speech.NUMPY_REQUIRED, caught.exception.message)
        with mock.patch.dict(sys.modules, {"numpy": SimpleNamespace(__version__="2.3.0")}):
            with self.assertRaises(speech.FloatVariantError) as caught:
                speech._numpy_version()
            self.assertEqual(caught.exception.code, "tool")
        with mock.patch.dict(sys.modules, {"numpy": SimpleNamespace(__version__="2.2.6")}):
            self.assertEqual(speech._numpy_version(), "2.2.6")

    def test_guard_accepts_approved_runners_and_requires_source_sha(self):
        base = {"GITHUB_ACTIONS": "true", "GITHUB_EVENT_NAME": "workflow_dispatch", "RUNNER_TEMP": "/tmp", "GITHUB_SHA": GHA}
        for env in (
            {"RUNNER_OS": "macOS", "RUNNER_NAME": "nax-macos-2"},
            {"RUNNER_OS": "Windows", "RUNNER_NAME": "cuda-windows"},
            {"RUNNER_OS": "Windows", "RUNNER_NAME": "cuda-windows-2"},
        ):
            self.assertEqual(speech.require_production_context(env={**base, **env})["runner"], env["RUNNER_NAME"])
        for env in (
            {"RUNNER_OS": "macOS", "RUNNER_NAME": "michaels-dev-mac"},
            {"RUNNER_OS": "Windows", "RUNNER_NAME": "cuda-windows-3"},
            {"RUNNER_OS": "Linux", "RUNNER_NAME": "ubuntu-latest"},
            {},
        ):
            with self.assertRaises(speech.FloatVariantError) as caught:
                speech.require_production_context(env={**base, **env})
            self.assertEqual(caught.exception.code, "guard")
        for env in ({**base, "GITHUB_SHA": "short"}, {**base, "GITHUB_EVENT_NAME": "push"}, {**base, "GITHUB_ACTIONS": "false"}):
            with self.assertRaises(speech.FloatVariantError) as caught:
                speech.require_production_context(env=env)
            self.assertEqual(caught.exception.code, "guard")

    def test_unit_fake_cannot_be_campaign_proof(self):
        with tempfile.TemporaryDirectory() as name:
            receipt = stage(SimpleNamespace(name=name))
            with self.assertRaises(speech.FloatVariantError) as caught:
                speech.assert_variant_receipt(receipt)
            self.assertEqual(caught.exception.code, "proof")
            receipt["proof_kind"] = "campaign_native"
            self.assertEqual(speech.assert_variant_receipt(receipt), receipt)


if __name__ == "__main__":
    unittest.main()
