"""Exercise the published-file checks with tiny fixtures, entirely offline."""
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sensenova_product_provision import ENDPOINT, provision


class ProvisionTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.cache = Path(temp.name)
        self.rev = "a" * 40
        self.plan = {"repo": "SceneWorks/test", "revision": self.rev,
                     "files": ["config.json", "q4/model.safetensors"]}
        self.snapshot = self.cache / "models--SceneWorks--test" / "snapshots" / self.rev
        self.snapshot.mkdir(parents=True)
        (self.snapshot / "config.json").write_bytes(b"{}")
        self.api = Mock()
        config = SimpleNamespace(rfilename="config.json", size=2, lfs=None,
                                 blob_id=hashlib.sha1(b"blob 2\x00{}").hexdigest())
        model = SimpleNamespace(rfilename="q4/model.safetensors", size=4,
                                lfs=SimpleNamespace(sha256=hashlib.sha256(b"data").hexdigest()))
        self.api.model_info.return_value = SimpleNamespace(sha=self.rev, siblings=[config, model])

    def test_only_missing_enumerated_files_download_anonymously(self):
        def fetch(**kwargs):
            self.assertEqual(kwargs, dict(repo_id="SceneWorks/test", revision=self.rev,
                                          filename="q4/model.safetensors", cache_dir=str(self.cache),
                                          endpoint=ENDPOINT, token=False))
            (self.snapshot / "q4").mkdir()
            (self.snapshot / kwargs["filename"]).write_bytes(b"data")
        receipt = provision(self.plan, self.cache, self.api, fetch)
        self.assertEqual([f["downloaded"] for f in receipt["files"]], [False, True])
        self.api.model_info.assert_called_once_with("SceneWorks/test", revision=self.rev,
                                                    files_metadata=True, token=False)
        (self.snapshot / "q4/model.safetensors").write_bytes(b"oops")
        with self.assertRaisesRegex(ValueError, "publisher hash mismatch"):
            provision(self.plan, self.cache, self.api, self.fail)

    def test_refuses_unpinned_or_changed_revision_and_unsafe_paths(self):
        for patch in [{"revision":"main"}, {"files":["../secret"]}, {"files":["*"]}]:
            with self.assertRaises(ValueError):
                provision(self.plan | patch, self.cache, self.api, self.fail)
        self.api.model_info.assert_not_called()
        self.api.model_info.return_value.sha = "b" * 40
        with self.assertRaisesRegex(ValueError, "revision differs"):
            provision(self.plan, self.cache, self.api, self.fail)


if __name__ == "__main__":
    unittest.main()
