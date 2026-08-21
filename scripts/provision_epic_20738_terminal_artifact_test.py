import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("provision-epic-20738-terminal-artifact.py")
SPEC = importlib.util.spec_from_file_location("epic_20738_provisioner", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ProvisionRequestTests(unittest.TestCase):
    def request(self) -> dict:
        return {
            "id": "chroma1-base-q4",
            "repository": "SceneWorks/chroma1-base-mlx",
            "revision": "a" * 40,
            "subdirectory": "q4",
            "allowPatterns": ["q4/**"],
            "destination": "D:/runner-temp/artifact",
        }

    def parse(self, value: dict) -> dict:
        with tempfile.TemporaryDirectory() as directory:
            request = Path(directory) / "request.json"
            request.write_text(json.dumps(value), encoding="utf-8")
            return MODULE.parse_request(request)

    def test_accepts_exact_immutable_confined_request_without_huggingface_dependency(self) -> None:
        self.assertEqual(self.parse(self.request())["allowPatterns"], ["q4/**"])

    def test_rejects_unreviewed_field_and_floating_revision(self) -> None:
        extra = self.request()
        extra["token"] = "secret"
        with self.assertRaisesRegex(ValueError, "fields must be exactly"):
            self.parse(extra)
        floating = self.request()
        floating["revision"] = "main"
        with self.assertRaisesRegex(ValueError, "40-hex"):
            self.parse(floating)

    def test_rejects_allow_pattern_or_subdirectory_escape(self) -> None:
        pattern = self.request()
        pattern["allowPatterns"] = ["../dense/**"]
        with self.assertRaisesRegex(ValueError, "cannot escape"):
            self.parse(pattern)
        subdirectory = self.request()
        subdirectory["subdirectory"] = "../q8"
        with self.assertRaisesRegex(ValueError, "cannot escape"):
            self.parse(subdirectory)


if __name__ == "__main__":
    unittest.main()
