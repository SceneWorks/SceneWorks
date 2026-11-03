"""Ordinary metadata/buffer contracts only: never import or initialize a model runtime."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import cv_provision as cv


def message(field, body):
    return vint(field * 8 + 2) + vint(len(body)) + body


def vint(value):
    result = bytearray()
    while value > 127:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


class Response(io.BytesIO):
    def __init__(self, payload, declared=None):
        super().__init__(payload)
        self.headers = {} if declared is None else {"Content-Length": str(declared)}


class ProvisionContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.payload = b"ordinary bounded file fixture"
        self.asset = {"name": "fixture.bin", "bytes": len(self.payload), "sha256": hashlib.sha256(self.payload).hexdigest(), "url": "https://example.invalid/fixture.bin"}
        self.receipt = {"downloadBytes": 0, "assets": []}

    def tearDown(self):
        self.temp.cleanup()

    def test_nonmanual_admission_precedes_git_or_imports(self):
        with patch.object(cv.subprocess, "check_output", side_effect=AssertionError("must not inspect source")):
            for env in ({}, {"GITHUB_ACTIONS": "true", "GITHUB_EVENT_NAME": "pull_request"}):
                with self.assertRaisesRegex(ValueError, "explicit Actions"):
                    cv.admit("a" * 40, "b" * 40, "mlx", env=env)

    def test_oracle_entry_rejects_before_any_model_import(self):
        # Executing a refused entry is safe: no torch/numpy imports are reachable first.
        code = "import sys; import cv_oracle_producer as p; " \
               "sys.argv=['p','--source-sha','" + "a" * 40 + "','--inference-sha','" + "b" * 40 + "','--fixtures','x','--output','y','--install-report','z']; " \
               "\ntry: p.main()\nexcept ValueError: pass\nelse: raise AssertionError('ordinary execution admitted')\n" \
               "assert 'torch' not in sys.modules and 'numpy' not in sys.modules"
        env = dict(os.environ)
        env.pop("GITHUB_ACTIONS", None)
        subprocess.run([sys.executable, "-B", "-c", code], cwd=SCRIPTS, env=env, check=True)

    def test_isolated_script_entries_refuse_ordinary_execution(self):
        env = dict(os.environ)
        env.pop("GITHUB_ACTIONS", None)
        common = ["--source-sha", "a" * 40, "--inference-sha", "b" * 40, "--output", str(self.root / "output")]
        entries = {
            "cv_provision.py": ["--mode", "oracle-input", "--backend", "mlx"],
            "cv_oracle_tools.py": [],
            "cv_oracle_producer.py": ["--fixtures", str(self.root), "--install-report", str(self.root / "missing.json")],
        }
        for name, extra in entries.items():
            result = subprocess.run([sys.executable, "-I", "-B", str(SCRIPTS / name), *common, *extra], env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("explicit Actions dispatch", result.stderr)
        self.assertFalse((self.root / "output").exists())

    def test_digest_checked_download_publishes_regular_file(self):
        cv.acquire(self.asset, self.root / "out", self.receipt, 1000,
                   opener=lambda *_a, **_k: Response(self.payload, len(self.payload)))
        self.assertEqual((self.root / "out").read_bytes(), self.payload)
        self.assertEqual(self.receipt["downloadBytes"], len(self.payload))
        self.assertFalse((self.root / "out.partial").exists())

    def test_oversized_or_wrong_bytes_never_publish(self):
        for payload in (self.payload + b"!", b"x" * len(self.payload)):
            with self.assertRaises(ValueError):
                cv.acquire(self.asset, self.root / "out", self.receipt, 1000,
                           opener=lambda *_a, **_k: Response(payload))
            self.assertFalse((self.root / "out").exists())
            self.assertFalse((self.root / "out.partial").exists())

    def test_budget_blocks_network_and_cached_copy_is_not_a_symlink(self):
        with self.assertRaisesRegex(ValueError, "budget"):
            cv.acquire(self.asset, self.root / "out", self.receipt, 0,
                       opener=lambda *_a, **_k: self.fail("network must not start"))
        cache = self.root / "cache"
        cache.mkdir()
        (cache / self.asset["name"]).write_bytes(self.payload)
        cv.acquire(self.asset, self.root / "out", self.receipt, 0, extra=cache,
                   opener=lambda *_a, **_k: self.fail("cache should satisfy asset"))
        self.assertFalse((self.root / "out").is_symlink())
        self.assertEqual(self.receipt["downloadBytes"], 0)

    def test_corrupt_cached_asset_is_not_silently_accepted(self):
        cache = self.root / "cache"
        cache.mkdir()
        (cache / self.asset["name"]).write_bytes(b"wrong")
        with self.assertRaisesRegex(ValueError, "identity"):
            cv.acquire(self.asset, self.root / "out", self.receipt, 1000, extra=cache,
                       opener=lambda *_a, **_k: self.fail("corrupt cache must surface"))

    def test_onnx_metadata_walk_rejects_external_and_truncated_tensors(self):
        # Protobuf metadata fixture, not executable ONNX and never submitted to a runtime.
        file = self.root / "metadata.pb"
        tensor = message(8, b"metadata")
        file.write_bytes(message(7, message(5, tensor)))
        self.assertEqual(cv.inspect_onnx(file)["tensors"], 1)
        for external in (vint(14 * 8) + vint(1), message(13, message(1, b"location") + message(2, b"missing.data"))):
            file.write_bytes(message(7, message(5, tensor + external)))
            with self.assertRaisesRegex(ValueError, "external"):
                cv.inspect_onnx(file)
        file.write_bytes(message(7, message(5, tensor))[:-1])
        with self.assertRaises(ValueError):
            cv.inspect_onnx(file)

    def test_runtime_extract_is_bounded_and_requires_all_provider_dlls(self):
        archive = self.root / "libs.zip"
        names = ["onnxruntime.dll", "onnxruntime_providers_cuda.dll", "onnxruntime_providers_shared.dll"]
        with zipfile.ZipFile(archive, "w") as out:
            for name in names:
                out.writestr("capi/" + name, b"metadata-only-not-a-library")
        with self.assertRaisesRegex(ValueError, "budget"):
            cv.extract_runtime(archive, self.root / "runtime", "ort", "cuda", 1)
        files = cv.extract_runtime(archive, self.root / "runtime", "ort", "cuda", 1000)
        self.assertEqual({Path(row["file"]).name for row in files}, set(names))
        with zipfile.ZipFile(archive, "w") as out:
            out.writestr("capi/onnxruntime.dll", b"not-a-library")
        with self.assertRaisesRegex(ValueError, "provider"):
            cv.extract_runtime(archive, self.root / "other", "ort", "cuda", 1000)

    def test_missing_or_failed_oracle_receipt_is_rejected(self):
        manifest = cv.load_manifest()
        (self.root / "oracle.json").write_text(json.dumps({"status": "failed"}))
        with self.assertRaisesRegex(ValueError, "provenance"):
            cv.validate_oracles(self.root, "a" * 40, "b" * 40, manifest)

    def test_runtime_pins_match_current_product_provisioners(self):
        manifest = cv.load_manifest()
        win = (cv.ROOT / "apps/desktop/src/cuda_provision.rs").read_text()
        mac = (cv.ROOT / "apps/desktop/scripts/stage-onnxruntime.py").read_text()
        for backend, source in (("mlx", mac), ("cuda", win)):
            for asset in manifest["runtime"][backend]:
                self.assertIn(asset["sha256"], source)
                self.assertIn(Path(asset["url"]).name, source)
            total = sum(a["bytes"] for a in manifest["models"] if backend in a["backends"]) + sum(a["bytes"] for a in manifest["runtime"][backend]) + manifest["photo"]["bytes"]
            self.assertLess(total, manifest["budgets"]["assetDownloadBytes"])
        self.assertEqual({p["name"] for p in manifest["oracle"]["packages"]}, {"torch", "torchvision", "ultralytics", "numpy", "safetensors", "opencv-python", "Pillow"})


if __name__ == "__main__":
    unittest.main()
