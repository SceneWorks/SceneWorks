import importlib.util
import hashlib
import json
import os
import socket
import stat
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
    CURRENT_REPOSITORY = "SceneWorks/illustrious-xl-v1-mlx"
    CURRENT_REVISION = "778c3f02b7703b0c2755d0c0447592897193c6b5"

    def request(self) -> dict:
        return {
            "id": "chroma1-base-q4",
            "repository": "SceneWorks/chroma1-base-mlx",
            "revision": self.REVISION,
            "subdirectory": "q4",
            "allowPatterns": ["q4/*"],
            "expectedFiles": ["weights.safetensors"],
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
            "expectedFiles": ["model_index.json", "transformer/model.safetensors"],
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

    def current_request(self, inventory: dict) -> dict:
        return {
            "id": "illustrious-v1-q4",
            "repository": self.CURRENT_REPOSITORY,
            "revision": self.CURRENT_REVISION,
            "subdirectory": "q4",
            "allowPatterns": ["q4/*"],
            "expectedFiles": sorted(inventory),
        }

    def current_selected(self, root: Path) -> Path:
        return (
            root / "models--SceneWorks--illustrious-xl-v1-mlx" / "snapshots"
            / self.CURRENT_REVISION / "q4"
        )

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
            request["expectedFiles"] = ["missing.safetensors", "weights.safetensors"]
            with self.assertRaisesRegex(RuntimeError, "authoritative expected file is missing"):
                MODULE.resolve_cached_artifact(request, cache)

    def test_unreviewed_derivative_extras_are_ignored_and_never_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory).resolve()
            snapshot = self.cache_snapshot(cache)
            (snapshot / "q4" / "weights.safetensors").write_bytes(b"weights")
            sidecars = snapshot / "q4" / ".candle-device-format-v1"
            sidecars.mkdir()
            for index in range(495):
                (sidecars / f"derived-{index:03}.bin").write_bytes(b"derived")
            (snapshot / "q4" / "weights.safetensors.incomplete").write_bytes(b"partial")
            audit = MODULE.audit_cached_artifact(self.request(), cache)
            self.assertEqual(audit["matchedFiles"], ["weights.safetensors"])
            self.assertEqual(audit["selectedFiles"], ["weights.safetensors"])
            self.assertEqual(len(audit["reusedFiles"]), 1)

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
            with self.assertRaisesRegex(RuntimeError, "escaped trusted cache root"):
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
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as stored:
            cache = Path(directory).resolve()
            missing_store = Path(stored).resolve()
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
                self.assertEqual(Path(kwargs["cache_dir"]), missing_store / ".hf-network-staging")
                self.assertFalse(kwargs["token"])
                self.assertFalse(kwargs["force_download"])
                self.assertFalse(kwargs["local_files_only"])
                target = Path(kwargs["local_dir"]) / kwargs["filename"]
                metadata_dir = Path(kwargs["local_dir"]) / ".cache" / "huggingface"
                metadata_dir.mkdir(parents=True)
                (metadata_dir / "download-metadata").write_text("untrusted downloader state")
                target.write_bytes(payload)
                return str(target)

            fake_hf = types.SimpleNamespace(
                __version__="0.36.0",
                hf_hub_url=url,
                get_hf_file_metadata=metadata,
                hf_hub_download=download,
            )
            with mock.patch.dict(sys.modules, {"huggingface_hub": fake_hf}):
                downloaded = MODULE.download_reviewed_missing(
                    self.flux_request(), cache, missing_store
                )
            result = MODULE.stage_artifact(
                self.flux_request(), cache, missing_store, missing_store
            )
            self.assertEqual(len(calls), 1)
            self.assertFalse(
                (missing_store / "models--SceneWorks--flux1-schnell-mlx" / "snapshots"
                 / self.flux_request()["revision"] / ".cache").exists()
            )
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
            self.assertEqual(downloaded["downloadedFiles"][0]["path"], "transformer/model.safetensors")
            self.assertEqual(result["downloadedFiles"][0]["path"], "transformer/model.safetensors")
            self.assertEqual(
                {key: value for key, value in downloaded["downloadedFiles"][0].items() if key != "path"},
                {key: value for key, value in result["downloadedFiles"][0].items() if key != "path"},
            )
            offline = MODULE.resolve_cached_artifact(self.flux_request(), missing_store)
            self.assertEqual(offline["downloadedFiles"], [])
            self.assertTrue(offline["complete"])
            stored_target = (
                missing_store / "models--SceneWorks--flux1-schnell-mlx" / "snapshots"
                / self.flux_request()["revision"] / "q8/transformer/model.safetensors"
            )
            stored_target.write_bytes(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "missing-file store failed exact identity"):
                MODULE.stage_artifact(self.flux_request(), cache, missing_store, missing_store)

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
                    self.flux_request(), cache, staging, None
                )
            called.assert_not_called()
            self.assertEqual(result["downloadedFiles"], [])
            self.assertEqual(target.read_bytes(), b"already valid")
            repeated = MODULE.stage_artifact(self.flux_request(), cache, staging, None)
            self.assertEqual(repeated["reusedFiles"], result["reusedFiles"])
            staged_target = (
                staging / "models--SceneWorks--flux1-schnell-mlx" / "snapshots"
                / self.flux_request()["revision"] / "q8/transformer/model.safetensors"
            )
            staged_target.write_bytes(b"drift")
            with self.assertRaisesRegex(RuntimeError, "refusing to overwrite non-identical"):
                MODULE.stage_artifact(self.flux_request(), cache, staging, None)

    def test_network_flag_rejects_unapproved_authority(self) -> None:
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as staged:
            cache = Path(directory).resolve()
            staging = Path(staged).resolve()
            snapshot = self.cache_snapshot(cache)
            (snapshot / "q4" / "weights.safetensors").write_bytes(b"weights")
            with self.assertRaisesRegex(
                RuntimeError, "network fill requires an exact reviewed missing-file plan"
            ):
                MODULE.download_reviewed_missing(self.request(), cache, staging)

    def test_reviewed_current_snapshot_can_be_fully_absent_or_partially_hydrated(self) -> None:
        config = b"exact config"
        weights = b"exact weights"
        inventory = {
            "config.json": (len(config), hashlib.sha256(config).hexdigest()),
            "unet/model.safetensors": (len(weights), hashlib.sha256(weights).hexdigest()),
        }
        authority = {(self.CURRENT_REPOSITORY, self.CURRENT_REVISION): inventory}
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            MODULE.REVIEWED_HYDRATION_AUTHORITIES, authority, clear=True
        ):
            cache = Path(directory).resolve()
            absent = MODULE.audit_cached_artifact(self.current_request(inventory), cache)
            self.assertFalse(absent["complete"])
            self.assertEqual(absent["reusedFiles"], [])
            self.assertEqual(absent["missingFiles"], ["q4/config.json", "q4/unet/model.safetensors"])

            selected = self.current_selected(cache)
            selected.mkdir(parents=True)
            (selected / "config.json").write_bytes(config)
            partial = MODULE.audit_cached_artifact(self.current_request(inventory), cache)
            self.assertEqual([row["path"] for row in partial["reusedFiles"]], ["config.json"])
            self.assertEqual(partial["missingFiles"], ["q4/unet/model.safetensors"])

    def test_reviewed_current_snapshot_rejects_corrupt_unexpected_and_unsafe_entries(self) -> None:
        payload = b"exact"
        inventory = {"weights.safetensors": (len(payload), hashlib.sha256(payload).hexdigest())}
        authority = {(self.CURRENT_REPOSITORY, self.CURRENT_REVISION): inventory}
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside_directory, mock.patch.dict(
            MODULE.REVIEWED_HYDRATION_AUTHORITIES, authority, clear=True
        ):
            cache = Path(directory).resolve()
            selected = self.current_selected(cache)
            selected.mkdir(parents=True)
            target = selected / "weights.safetensors"
            target.write_bytes(b"drift")
            with self.assertRaisesRegex(RuntimeError, "source byte identity drifted"):
                MODULE.audit_cached_artifact(self.current_request(inventory), cache)
            target.write_bytes(payload)
            unexpected = selected / "unexpected.bin"
            unexpected.write_bytes(b"unreviewed")
            with self.assertRaisesRegex(RuntimeError, "unexpected file"):
                MODULE.audit_cached_artifact(self.current_request(inventory), cache)
            unexpected.unlink()
            target.unlink()
            outside = Path(outside_directory).resolve() / "outside.bin"
            outside.write_bytes(payload)
            try:
                os.symlink(outside, target)
            except OSError as error:
                self.skipTest(f"symlink/reparse fixture unavailable: {error}")
            with self.assertRaisesRegex(RuntimeError, "escaped trusted cache root"):
                MODULE.audit_cached_artifact(self.current_request(inventory), cache)

    def test_reviewed_current_partial_snapshot_downloads_only_missing_bytes_then_audits_stage(self) -> None:
        first, second = b"cached exact", b"download exact"
        inventory = {
            "config.json": (len(second), hashlib.sha256(second).hexdigest()),
            "unet/model.safetensors": (len(first), hashlib.sha256(first).hexdigest()),
        }
        authority = {(self.CURRENT_REPOSITORY, self.CURRENT_REVISION): inventory}
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as stored, mock.patch.dict(
            MODULE.REVIEWED_HYDRATION_AUTHORITIES, authority, clear=True
        ):
            cache = Path(directory).resolve()
            missing_store = Path(stored).resolve()
            selected = self.current_selected(cache)
            (selected / "unet").mkdir(parents=True)
            (selected / "unet" / "model.safetensors").write_bytes(first)

            def url(**kwargs):
                return f"https://huggingface.invalid/{kwargs['filename']}"

            def metadata(_url, token):
                self.assertFalse(token)
                return types.SimpleNamespace(
                    commit_hash=self.CURRENT_REVISION,
                    etag="c" * 40,
                    size=len(second),
                )

            def download(**kwargs):
                self.assertEqual(kwargs["filename"], "q4/config.json")
                target = Path(kwargs["local_dir"]) / kwargs["filename"]
                target.write_bytes(second)
                return str(target)

            fake_hf = types.SimpleNamespace(
                __version__="0.36.0", hf_hub_url=url,
                get_hf_file_metadata=metadata, hf_hub_download=download,
            )
            with mock.patch.dict(sys.modules, {"huggingface_hub": fake_hf}):
                downloaded = MODULE.download_reviewed_missing(
                    self.current_request(inventory), cache, missing_store
                )
            self.assertEqual([row["path"] for row in downloaded["downloadedFiles"]], ["config.json"])
            staged_result = MODULE.stage_artifact(
                self.current_request(inventory), cache, missing_store, missing_store
            )
            self.assertTrue(staged_result["complete"])
            self.assertEqual(staged_result["matchedFiles"], sorted(inventory))

    def test_download_accepts_only_the_same_resolved_ordinary_confined_file(self) -> None:
        payload = b"exact download"
        inventory = {"config.json": (len(payload), hashlib.sha256(payload).hexdigest())}
        authority = {(self.CURRENT_REPOSITORY, self.CURRENT_REVISION): inventory}

        for mode, expected_error in [
            ("normalized-alias", None),
            ("extended-drive", None),
            ("different", "resolved to a different file"),
            ("missing-destination", "expected destination is missing"),
            ("missing-returned", "returned path is missing"),
            ("escape", "returned path escaped the exact destination snapshot"),
            ("reparse", "expected destination is not an ordinary non-reparse file"),
        ]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory, \
                    tempfile.TemporaryDirectory() as stored, tempfile.TemporaryDirectory() as outside, \
                    mock.patch.dict(MODULE.REVIEWED_HYDRATION_AUTHORITIES, authority, clear=True):
                cache = Path(directory).resolve()
                missing_store = Path(stored).resolve()
                outside_file = Path(outside).resolve() / "outside.bin"

                def url(**kwargs):
                    return f"https://huggingface.invalid/{kwargs['filename']}"

                def metadata(_url, token):
                    self.assertFalse(token)
                    return types.SimpleNamespace(
                        commit_hash=self.CURRENT_REVISION,
                        etag="c" * 40,
                        size=len(payload),
                    )

                def download(**kwargs):
                    destination = Path(kwargs["local_dir"]) / kwargs["filename"]
                    if mode != "missing-destination":
                        destination.write_bytes(payload)
                    if mode == "normalized-alias":
                        alias = str(
                            destination.parent / ".." / destination.parent.name / destination.name
                        )
                        return alias.swapcase().replace("\\", "/") if os.name == "nt" else alias
                    if mode == "extended-drive":
                        return f"\\\\?\\{destination}"
                    if mode in {"different", "missing-destination"}:
                        different = destination.with_name("different.bin")
                        different.write_bytes(payload)
                        return str(different)
                    if mode == "missing-returned":
                        return str(destination.with_name("missing.bin"))
                    if mode == "escape":
                        outside_file.write_bytes(payload)
                        return str(outside_file)
                    return str(destination)

                fake_hf = types.SimpleNamespace(
                    __version__="0.36.0", hf_hub_url=url,
                    get_hf_file_metadata=metadata, hf_hub_download=download,
                )
                reparse = mock.patch.object(
                    MODULE,
                    "_is_reparse",
                    side_effect=lambda value: mode == "reparse" and stat.S_ISREG(value.st_mode),
                )
                with mock.patch.dict(sys.modules, {"huggingface_hub": fake_hf}), reparse:
                    if expected_error is not None:
                        with self.assertRaisesRegex(RuntimeError, expected_error):
                            MODULE.download_reviewed_missing(
                                self.current_request(inventory), cache, missing_store
                            )
                        continue
                    downloaded = MODULE.download_reviewed_missing(
                        self.current_request(inventory), cache, missing_store
                    )
                self.assertEqual(downloaded["downloadedFiles"], [{
                    "path": "config.json",
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "lfsSha256": hashlib.sha256(payload).hexdigest(),
                    "commitSha": self.CURRENT_REVISION,
                }])

    @unittest.skipUnless(os.name == "nt", "Windows extended path spellings are Windows-only")
    def test_windows_extended_prefix_normalizes_only_drive_and_unc_paths(self) -> None:
        self.assertEqual(
            MODULE._without_windows_extended_prefix(
                Path(r"\\?\D:\runner\snapshot\q4\model_index.json"), "fixture"
            ),
            Path(r"D:\runner\snapshot\q4\model_index.json"),
        )
        self.assertEqual(
            MODULE._without_windows_extended_prefix(
                Path(r"\\?\UNC\server\share\snapshot\q4\model_index.json"), "fixture"
            ),
            Path(r"\\server\share\snapshot\q4\model_index.json"),
        )
        with self.assertRaisesRegex(RuntimeError, "unsupported Windows device namespace"):
            MODULE._without_windows_extended_prefix(
                Path(r"\\?\GLOBALROOT\Device\HarddiskVolumeShadowCopy1\escape"), "fixture"
            )

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
