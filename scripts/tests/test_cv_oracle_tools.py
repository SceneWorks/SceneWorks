"""Closed wheel provisioning contracts only; never import learned runtimes."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv_oracle_tools as tools
import cv_provision as cv


class LockedOracleTools(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.output = Path(self.temp.name).resolve()
        self.manifest = cv.load_manifest()
        self.packages = tools.locked_packages(self.manifest)

    def tearDown(self):
        self.temp.cleanup()

    def report(self):
        return {"install": [{
            "metadata": {"name": package["name"], "version": package["version"]},
            "download_info": {"url": (self.output / "wheels" / package["filename"]).as_uri(),
                              "archive_info": {"hashes": {"sha256": package["sha256"]}}},
        } for package in self.packages]}

    def test_complete_observed_lock_includes_yaml_and_original_primary_versions(self):
        versions = {package["name"]: package["version"] for package in self.packages}
        self.assertEqual(self.manifest["oracle"]["lockProvenance"]["runId"], 41682964895)
        self.assertEqual(len(versions), 38)
        self.assertEqual(versions["pyyaml"], "6.0.3")
        self.assertEqual(versions["torch"], "2.8.0")
        self.assertEqual(versions["ultralytics"], "8.3.0")
        self.assertEqual(sum(package["size"] for package in self.packages), 205845717)
        self.assertEqual(self.manifest["oracle"]["platform"]["machine"], "arm64")

    def test_incomplete_extra_duplicate_or_wrong_platform_lock_refuses_before_download(self):
        for mutate in (
            lambda m: m["oracle"]["packages"].pop(),
            lambda m: m["oracle"]["packages"].append(copy.deepcopy(m["oracle"]["packages"][0])),
            lambda m: m["oracle"]["platform"].update(sysPlatform="linux"),
            lambda m: m["oracle"]["packages"][1].update(name=m["oracle"]["packages"][0]["name"]),
        ):
            invalid = copy.deepcopy(self.manifest)
            mutate(invalid)
            with patch.object(tools, "acquire", side_effect=AssertionError("no download")), \
                    patch.object(tools.subprocess, "run", side_effect=AssertionError("no installer")):
                with self.assertRaises(ValueError):
                    tools.install_tools(invalid, self.output, {"downloadBytes": 0, "assets": []})

    def test_unreviewed_urls_paths_hashes_sizes_and_versions_refuse(self):
        for values in (
            {"url": "https://attacker.invalid/numpy.whl"},
            {"url": self.packages[0]["url"] + "?latest=true"},
            {"url": self.packages[0]["url"].replace("files.pythonhosted.org", "files.pythonhosted.org:443")},
            {"filename": "../outside.whl"},
            {"filename": "different-1.0-py3-none-any.whl"},
            {"sha256": "0" * 63},
            {"size": True},
            {"size": 0},
            {"version": "latest"},
        ):
            invalid = copy.deepcopy(self.manifest)
            invalid["oracle"]["packages"][0].update(values)
            with self.assertRaises(ValueError):
                tools.locked_packages(invalid)

    def test_yaml_cannot_be_omitted_even_from_an_otherwise_consistent_inventory(self):
        invalid = copy.deepcopy(self.manifest)
        packages = invalid["oracle"]["packages"] = [p for p in invalid["oracle"]["packages"] if p["name"] != "pyyaml"]
        invalid["oracle"]["lockProvenance"].update(packageCount=len(packages), wheelBytes=sum(p["size"] for p in packages))
        with self.assertRaisesRegex(ValueError, "direct imports"):
            tools.locked_packages(invalid)

    def test_install_is_closed_offline_hash_required_and_environment_independent(self):
        commands, acquired = [], []
        def acquire(asset, destination, receipt, budget, **kwargs):
            acquired.append(asset)
            self.assertFalse(kwargs["use_cache"])
            self.assertLessEqual(sum(item["bytes"] for item in acquired), budget)
        def run(command, **kwargs):
            commands.append(command)
            if "install" in command:
                self.assertEqual(len(acquired), len(self.packages), "all pins verified before installer")
                for flag in ("--no-index", "--no-deps", "--require-hashes", "--only-binary=:all:", "--ignore-installed"):
                    self.assertIn(flag, command)
                for flag in ("--dry-run", "--index-url", "--extra-index-url", "--find-links"):
                    self.assertNotIn(flag, command)
                env = kwargs["env"]
                self.assertEqual(env["PIP_CONFIG_FILE"], os.devnull)
                self.assertEqual(env["PIP_NO_INDEX"], "1")
                self.assertNotIn("PIP_INDEX_URL", env)
                self.assertNotIn("PIP_EXTRA_INDEX_URL", env)
                self.assertNotIn("PIP_FIND_LINKS", env)
                requirements = Path(command[command.index("--requirement") + 1]).read_text().splitlines()
                self.assertEqual(len(requirements), len(self.packages))
                for package, line in zip(self.packages, requirements):
                    self.assertEqual(line, f"{package['name']} @ {(self.output / 'wheels' / package['filename']).as_uri()} --hash=sha256:{package['sha256']}")
                (self.output / "install.json").write_text(json.dumps(self.report()))
            self.assertIn("-I", command)
            self.assertIn("-B", command)
            self.assertTrue(kwargs["check"])
            self.assertGreater(kwargs["timeout"], 0)
        with patch.object(tools, "acquire", side_effect=acquire), \
                patch.object(tools.subprocess, "run", side_effect=run), \
                patch.dict(os.environ, {"PIP_INDEX_URL": "https://attacker.invalid/", "PIP_EXTRA_INDEX_URL": "https://attacker.invalid/extra", "PIP_FIND_LINKS": "/ambient"}):
            python = tools.install_tools(self.manifest, self.output, {"downloadBytes": 0, "assets": []})
        self.assertEqual(python, self.output / "venv/bin/python")
        self.assertEqual(len(commands), 4)  # venv; one offline install; check; freeze
        self.assertEqual(json.loads((self.output / "wheel-lock.json").read_text()), self.manifest["oracle"])

    def test_incomplete_extra_duplicate_or_changed_install_report_refuses(self):
        tools.verify_install_report(self.report(), self.packages, self.output / "wheels")
        for mutate in (
            lambda r: r["install"].pop(),
            lambda r: r["install"].append(copy.deepcopy(r["install"][0])),
            lambda r: r["install"][1].update(metadata=r["install"][0]["metadata"]),
            lambda r: r["install"][0]["metadata"].update(version="8.3.1"),
            lambda r: r["install"][0]["download_info"]["archive_info"]["hashes"].update(sha256="0" * 64),
            lambda r: r["install"][0]["download_info"].update(url="https://files.pythonhosted.org/unreviewed.whl"),
        ):
            invalid = self.report()
            mutate(invalid)
            with self.assertRaises(ValueError):
                tools.verify_install_report(invalid, self.packages, self.output / "wheels")

    def test_altered_installer_report_stops_before_check_freeze_or_execution(self):
        commands = []
        def run(command, **_kwargs):
            commands.append(command)
            if "install" in command:
                invalid = self.report()
                invalid["install"][0]["metadata"]["version"] = "unreviewed"
                (self.output / "install.json").write_text(json.dumps(invalid))
        with patch.object(tools, "acquire"), patch.object(tools.subprocess, "run", side_effect=run):
            with self.assertRaisesRegex(ValueError, "reviewed lock"):
                tools.install_tools(self.manifest, self.output, {"downloadBytes": 0, "assets": []})
        self.assertEqual(len(commands), 2)  # venv and install only; no subsequent tool/model step

    def test_corrupt_locked_download_never_reaches_installer(self):
        payload = b"bounded wheel bytes; not executable"
        first = self.packages[0]
        first.update(size=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        self.manifest["oracle"]["lockProvenance"]["wheelBytes"] = sum(p["size"] for p in self.packages)
        def download(asset, destination, receipt, budget, **_kwargs):
            response = io.BytesIO(payload[:-1])
            response.headers = {}
            cv.acquire(asset, destination, receipt, budget, use_cache=False, opener=lambda *_a, **_k: response)
        with patch.object(tools, "acquire", side_effect=download), \
                patch.object(tools.subprocess, "run", side_effect=AssertionError("no installer")):
            with self.assertRaisesRegex(ValueError, "identity"):
                tools.install_tools(self.manifest, self.output, {"downloadBytes": 0, "assets": []})
        self.assertFalse((self.output / "wheels" / first["filename"]).exists())
        self.assertFalse((self.output / "wheels" / (first["filename"] + ".partial")).exists())


if __name__ == "__main__":
    unittest.main()
