import importlib.util
import hashlib
import json
import os
import socket
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).with_name("provision-epic-20738-terminal-artifact.py")
SPEC = importlib.util.spec_from_file_location("epic_20738_provisioner", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CacheOnlyProvisionTests(unittest.TestCase):
    REVISION = "a" * 40

    def request(self) -> dict:
        return {
            "id": "chroma1-base-q4",
            "repository": "SceneWorks/chroma1-base-mlx",
            "revision": self.REVISION,
            "subdirectory": "q4",
            "allowPatterns": ["q4/*"],
        }

    def parse(self, value: dict) -> dict:
        with tempfile.TemporaryDirectory() as directory:
            request = Path(directory) / "request.json"
            request.write_text(json.dumps(value), encoding="utf-8")
            return MODULE.parse_request(request)

    def cache_snapshot(self, root: Path, revision: str | None = None) -> Path:
        snapshot = (
            root
            / "models--SceneWorks--chroma1-base-mlx"
            / "snapshots"
            / (revision or self.REVISION)
        )
        (snapshot / "q4").mkdir(parents=True)
        return snapshot

    def flux_request(self) -> dict:
        return {
            "id": "flux1-schnell-q8",
            "repository": "SceneWorks/flux1-schnell-mlx",
            "revision": "bba3ae01dfd94089f173c05edd4e1a4c551f2599",
            "subdirectory": "q8",
            "allowPatterns": ["q8/*"],
        }

    def flux_snapshot(self, root: Path) -> Path:
        snapshot = (
            root
            / "models--SceneWorks--flux1-schnell-mlx"
            / "snapshots"
            / self.flux_request()["revision"]
        )
        (snapshot / "q8" / "transformer").mkdir(parents=True)
        (snapshot / "q8" / "model_index.json").write_bytes(b"{}")
        return snapshot

    def test_cache_hit_resolves_only_exact_allow_list_without_copy_or_network(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory).resolve()
            snapshot = self.cache_snapshot(cache)
            (snapshot / "q4" / "weights.safetensors").write_bytes(b"weights")
            (snapshot / "README.md").write_text("unselected", encoding="utf-8")
            with mock.patch.object(socket, "create_connection") as network:
                result = MODULE.resolve_cached_artifact(self.request(), cache)
            network.assert_not_called()
            self.assertEqual(Path(result["cacheRoot"]), cache)
            self.assertEqual(Path(result["snapshotRoot"]), snapshot)
            self.assertEqual(Path(result["selectedRoot"]), snapshot / "q4")
            self.assertEqual(result["matchedFiles"], ["weights.safetensors"])
            self.assertEqual(list((snapshot / "q4").iterdir()), [snapshot / "q4" / "weights.safetensors"])

    def test_missing_cache_refuses_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory).resolve()
            cache.mkdir(exist_ok=True)
            with mock.patch.object(socket, "create_connection") as network:
                with self.assertRaisesRegex(RuntimeError, "exact cached repository is missing"):
                    MODULE.resolve_cached_artifact(self.request(), cache)
            network.assert_not_called()

    def test_mutable_ref_or_wrong_revision_cannot_satisfy_exact_revision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory).resolve()
            wrong = "b" * 40
            snapshot = self.cache_snapshot(cache, wrong)
            (snapshot / "q4" / "weights.safetensors").write_bytes(b"wrong revision")
            refs = snapshot.parents[1] / "refs"
            refs.mkdir()
            (refs / "main").write_text(wrong, encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, f"exact cached revision {self.REVISION} is missing"):
                MODULE.resolve_cached_artifact(self.request(), cache)

    def test_empty_or_missing_allow_pattern_is_an_incomplete_cache_refusal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory).resolve()
            snapshot = self.cache_snapshot(cache)
            (snapshot / "q4" / "weights.safetensors").write_bytes(b"")
            with self.assertRaisesRegex(RuntimeError, "empty file"):
                MODULE.resolve_cached_artifact(self.request(), cache)
            (snapshot / "q4" / "weights.safetensors").write_bytes(b"weights")
            request = self.request()
            request["allowPatterns"] = ["q4/missing.safetensors"]
            with self.assertRaisesRegex(RuntimeError, "incomplete for allow-pattern"):
                MODULE.resolve_cached_artifact(request, cache)

    def test_incomplete_file_inside_selected_authority_is_never_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory).resolve()
            snapshot = self.cache_snapshot(cache)
            (snapshot / "q4" / "weights.safetensors.incomplete").write_bytes(b"partial")
            with self.assertRaisesRegex(RuntimeError, "untrusted incomplete file"):
                MODULE.audit_cached_artifact(self.request(), cache)

    def test_reparse_entry_or_cache_root_escape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            cache = Path(directory).resolve()
            snapshot = self.cache_snapshot(cache)
            outside_file = Path(outside).resolve() / "weights.safetensors"
            outside_file.write_bytes(b"outside")
            link = snapshot / "q4" / "weights.safetensors"
            try:
                os.symlink(outside_file, link)
            except OSError as error:
                self.skipTest(f"symlink/reparse fixture unavailable: {error}")
            with self.assertRaisesRegex(RuntimeError, "escaped the trusted cache root"):
                MODULE.resolve_cached_artifact(self.request(), cache)

    def test_huggingface_file_link_is_allowed_only_when_blob_stays_in_trusted_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory).resolve()
            snapshot = self.cache_snapshot(cache)
            blob = snapshot.parents[1] / "blobs" / ("b" * 64)
            blob.parent.mkdir()
            blob.write_bytes(b"trusted blob")
            link = snapshot / "q4" / "weights.safetensors"
            try:
                os.symlink(blob, link)
            except OSError as error:
                self.skipTest(f"symlink/reparse fixture unavailable: {error}")
            result = MODULE.resolve_cached_artifact(self.request(), cache)
            self.assertEqual(result["matchedFiles"], ["weights.safetensors"])
            self.assertEqual(result["reusedFiles"], [{
                "path": "weights.safetensors",
                "bytes": 12,
                "sha256": hashlib.sha256(b"trusted blob").hexdigest(),
            }])

    def test_staging_copies_hits_once_and_downloads_only_the_reviewed_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as staged:
            cache = Path(directory).resolve()
            staging = Path(staged).resolve()
            snapshot = self.flux_snapshot(cache)
            incomplete = snapshot.parents[1] / "blobs" / (("c" * 64) + ".incomplete")
            incomplete.parent.mkdir(exist_ok=True)
            incomplete.write_bytes(b"partial")
            calls = []
            payload = b"downloaded exact transformer"
            digest = hashlib.sha256(payload).hexdigest()

            audit = MODULE.audit_cached_artifact(self.flux_request(), cache)
            self.assertFalse(audit["complete"])
            self.assertEqual(
                audit["missingFiles"], ["q8/transformer/model.safetensors"]
            )

            def url(**kwargs):
                self.assertEqual(kwargs, {
                    "repo_id": "SceneWorks/flux1-schnell-mlx",
                    "filename": "q8/transformer/model.safetensors",
                    "revision": "bba3ae01dfd94089f173c05edd4e1a4c551f2599",
                })
                return "https://huggingface.co/exact"

            def metadata(request_url, token):
                self.assertEqual(request_url, "https://huggingface.co/exact")
                self.assertFalse(token)
                return types.SimpleNamespace(
                    commit_hash=self.flux_request()["revision"],
                    etag=digest,
                    size=len(payload),
                )

            def download(**kwargs):
                calls.append(kwargs)
                self.assertEqual(kwargs["repo_id"], "SceneWorks/flux1-schnell-mlx")
                self.assertEqual(kwargs["filename"], "q8/transformer/model.safetensors")
                self.assertEqual(kwargs["revision"], self.flux_request()["revision"])
                self.assertEqual(kwargs["repo_type"], "model")
                self.assertEqual(Path(kwargs["cache_dir"]), staging / ".hf-network-staging")
                self.assertFalse(kwargs["token"])
                self.assertFalse(kwargs["force_download"])
                self.assertFalse(kwargs["local_files_only"])
                target = Path(kwargs["local_dir"]) / kwargs["filename"]
                target.write_bytes(payload)
                return str(target)

            fake_hf = types.SimpleNamespace(
                __version__="0.36.0",
                hf_hub_url=url,
                get_hf_file_metadata=metadata,
                hf_hub_download=download,
            )
            with mock.patch.dict(sys.modules, {"huggingface_hub": fake_hf}):
                result = MODULE.stage_artifact(
                    self.flux_request(), cache, staging, True
                )
            self.assertEqual(len(calls), 1)
            self.assertEqual(
                result["downloadedFiles"],
                [{
                    "path": "transformer/model.safetensors",
                    "bytes": len(payload),
                    "sha256": digest,
                    "lfsSha256": digest,
                    "commitSha": self.flux_request()["revision"],
                }],
            )
            self.assertEqual(incomplete.read_bytes(), b"partial", "helper never selects partial blobs itself")
            self.assertFalse((snapshot / "q8/transformer/model.safetensors").exists())
            offline = MODULE.resolve_cached_artifact(self.flux_request(), staging)
            self.assertEqual(offline["downloadedFiles"], [])
            self.assertTrue(offline["complete"])

    def test_staging_never_downloads_or_overwrites_a_valid_cache_hit(self) -> None:
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as staged:
            cache = Path(directory).resolve()
            staging = Path(staged).resolve()
            snapshot = self.flux_snapshot(cache)
            target = snapshot / "q8" / "transformer" / "model.safetensors"
            target.write_bytes(b"already valid")
            called = mock.Mock(side_effect=AssertionError("network must not be called"))
            fake_hf = types.SimpleNamespace(
                __version__="0.36.0", hf_hub_url=called,
                get_hf_file_metadata=called, hf_hub_download=called,
            )
            with mock.patch.dict(sys.modules, {"huggingface_hub": fake_hf}):
                result = MODULE.stage_artifact(
                    self.flux_request(), cache, staging, False
                )
            called.assert_not_called()
            self.assertEqual(result["downloadedFiles"], [])
            self.assertEqual(target.read_bytes(), b"already valid")
            with self.assertRaisesRegex(RuntimeError, "refusing to overwrite staged cache file"):
                MODULE.stage_artifact(self.flux_request(), cache, staging, False)

    def test_network_flag_rejects_unapproved_authority(self) -> None:
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as staged:
            cache = Path(directory).resolve()
            staging = Path(staged).resolve()
            snapshot = self.cache_snapshot(cache)
            (snapshot / "q4" / "weights.safetensors").write_bytes(b"weights")
            with self.assertRaisesRegex(
                RuntimeError, "network staging requires exactly the sole reviewed missing filename"
            ):
                MODULE.stage_artifact(self.request(), cache, staging, True)

    def test_request_rejects_unreviewed_floating_or_escaping_fields(self) -> None:
        self.assertEqual(self.parse(self.request())["allowPatterns"], ["q4/*"])
        extra = self.request()
        extra["token"] = "secret"
        with self.assertRaisesRegex(ValueError, "fields must be exactly"):
            self.parse(extra)
        floating = self.request()
        floating["revision"] = "main"
        with self.assertRaisesRegex(ValueError, "40-hex"):
            self.parse(floating)
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
