"""Unit tests for cache_inventory.py. Synthetic temporary directories ONLY; no real cache is read.

Run: python3.12 -I scripts/cache-inventory/test_cache_inventory.py
"""

import contextlib
import importlib.util
import io
import json
import os
import pathlib
import tempfile
import unittest
from unittest import mock

_HERE = pathlib.Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location("cache_inventory", _HERE / "cache_inventory.py")
ci = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ci)

IRIS = ci.IRIS_REPO
QWEN = ci.QWEN_REPO


def snapshot_tree(root):
    """(relative path, type, size, mtime_ns, link target) for everything under root."""
    rows = []
    for base, dirs, files in os.walk(root, followlinks=False):
        for name in sorted(dirs + files):
            full = os.path.join(base, name)
            st = os.lstat(full)
            link = os.readlink(full) if os.path.islink(full) else None
            rows.append((os.path.relpath(full, root), st.st_mode, st.st_size, st.st_mtime_ns, link))
    return sorted(rows)


class Fixture(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = pathlib.Path(os.path.realpath(self._tmp.name))
        self.runner_temp = self.base / "runner_temp"
        self.runner_temp.mkdir()
        self.root = self.base / "data" / "hf" / "hub"
        self.root.mkdir(parents=True)

    def seed_iris(self):
        blobs = self.root / IRIS / "blobs"
        snap = self.root / IRIS / "snapshots" / "rev"
        blobs.mkdir(parents=True)
        (snap / "depth").mkdir(parents=True)
        (blobs / "blob0").write_bytes(b"x" * 10)
        (blobs / "blob2").write_bytes(b"y" * 12)
        os.symlink(blobs / "blob0", snap / "config.yaml")
        os.symlink("../../blobs/blob2", snap / "depth" / "config.yaml")

    def env(self, **extra):
        env = {"RUNNER_TEMP": str(self.runner_temp), "HF_HUB_CACHE": str(self.root), "HOME": str(self.base)}
        env.update(extra)
        return env

    def entries(self, report):
        return {entry["path"]: entry for entry in report["entries"]}


class AllowlistTests(unittest.TestCase):
    def test_allowlist_is_finite_relative_and_fixture_scoped(self):
        self.assertEqual(len(ci.ALLOWLIST), 84)
        self.assertEqual(len(set(ci.ALLOWLIST)), 84)
        repos = {IRIS, QWEN, ci.MINIMAX_REPO, ci.ADAPTER_REPO, ci.SOLO_REPO, ci.PAIR_REPO}
        for rel in ci.ALLOWLIST:
            self.assertFalse(os.path.isabs(rel))
            self.assertNotIn("..", rel.split("/"))
            self.assertIn(rel.split("/")[0], repos, rel)
            # only literal fixture revision "rev" ever appears under snapshots/
            parts = rel.split("/")
            if len(parts) > 2 and parts[1] == "snapshots":
                self.assertEqual(parts[2], "rev", rel)
        self.assertEqual({rel.split("/")[0] for rel in ci.ALLOWLIST}, repos)
        self.assertEqual(
            ci.GROUP_COUNTS,
            {
                "f203_iris_qwen": 28,
                "f8c46_minimax_h3_shared_repo": 32,
                "f8c46_zimage_training_adapter": 8,
                "f8c46_exclusive_solo_model": 10,
                "f8c46_unscoped_whole_repo_pair": 6,
            },
        )
        # f203 behavior is preserved verbatim
        self.assertIn(f"{IRIS}/blobs/empty-prompt-blob", ci.ALLOWLIST)
        self.assertIn(f"{IRIS}/snapshots/rev/upscaler/empty_prompt.safetensors", ci.ALLOWLIST)
        self.assertNotIn(f"{IRIS}/blobs/blob7", ci.ALLOWLIST)
        self.assertEqual(list(ci.ALLOWLIST[:28]), list(ci.GROUPS[0][1]))

    def test_f8c46_fixture_names_match_the_source_seed_helpers(self):
        m = ci.MINIMAX_REPO
        for rel in (
            "blobs/q4-config", "blobs/q4dit", "blobs/q4dit-index", "blobs/q4refdit",
            "blobs/q4refdit-index", "blobs/bf16-config", "blobs/bf16refdit-index",
            "snapshots/rev/q4/transformer/config.json",
            "snapshots/rev/q4/transformer_ref/config.json",
            "snapshots/rev/bf16/transformer_ref/diffusion_pytorch_model-00001-of-00001.safetensors",
            "snapshots/rev/bf16/transformer/diffusion_pytorch_model.safetensors.index.json",
        ):
            self.assertIn(f"{m}/{rel}", ci.ALLOWLIST)
        a = ci.ADAPTER_REPO
        for rel in ("blobs/v1blob", "blobs/v2blob", "snapshots/rev/zimage_turbo_training_adapter_v1.safetensors"):
            self.assertIn(f"{a}/{rel}", ci.ALLOWLIST)
        self.assertIn(f"{ci.SOLO_REPO}/snapshots/rev/README.md", ci.ALLOWLIST)
        self.assertIn(f"{ci.SOLO_REPO}/blobs/readme", ci.ALLOWLIST)
        self.assertIn(f"{ci.PAIR_REPO}/blobs/dit", ci.ALLOWLIST)
        # q8 is not seeded by the four named tests
        self.assertFalse([p for p in ci.ALLOWLIST if "q8" in p])


class ResolverTests(unittest.TestCase):
    def resolve(self, **env):
        return ci.resolve_root(env, home="/Users/someone")

    def test_precedence_hub_cache_then_huggingface_hub_cache_then_hf_home(self):
        root, prov = self.resolve(HF_HUB_CACHE="/a/b/c", HUGGINGFACE_HUB_CACHE="/d/e/f", HF_HOME="/g/h/i")
        self.assertEqual((root, prov["chosen_variable"]), ("/a/b/c", "HF_HUB_CACHE"))
        root, prov = self.resolve(HUGGINGFACE_HUB_CACHE="/d/e/f", HF_HOME="/g/h/i")
        self.assertEqual((root, prov["chosen_variable"]), ("/d/e/f", "HUGGINGFACE_HUB_CACHE"))
        root, prov = self.resolve(HF_HOME="/g/h/i")
        self.assertEqual((root, prov["chosen_variable"]), ("/g/h/i/hub", "HF_HOME"))
        self.assertEqual(prov["status"], "RESOLVED")

    def test_blank_values_are_unset_and_trimmed(self):
        root, prov = self.resolve(HF_HUB_CACHE="   ", HUGGINGFACE_HUB_CACHE="", HF_HOME=" /g/h/i ")
        self.assertEqual(root, "/g/h/i/hub")
        self.assertFalse(prov["sources"][0]["set"])

    def test_no_variable_is_unknown_and_never_invents_a_root(self):
        root, prov = self.resolve(HOME="/Users/someone", PATH="/usr/bin")
        self.assertIsNone(root)
        self.assertEqual(prov["status"], "UNKNOWN")
        self.assertIsNone(prov["chosen_variable"])

    def test_overbroad_or_invalid_roots_are_rejected(self):
        for bad in ("/", "/Users", "/Users/someone", "relative/hub", "/a/../b/c/d", "/a/b\x00c/d", "/tmp"):
            root, prov = self.resolve(HF_HUB_CACHE=bad)
            self.assertIsNone(root, bad)
            self.assertEqual(prov["status"], "REJECTED", bad)

    def test_non_hub_basename_warns(self):
        _, prov = self.resolve(HF_HUB_CACHE="/a/b/cache")
        self.assertIn("warning", prov)


class InspectTests(Fixture):
    def test_present_and_absent_are_distinct_and_symlink_is_not_followed(self):
        self.seed_iris()
        report = ci.build_report(self.env(), home=str(self.base))
        self.assertEqual(report["root_status"], "RESOLVED")
        self.assertEqual(len(report["entries"]), 84)
        got = self.entries(report)
        self.assertEqual(got[f"{IRIS}/blobs/blob0"]["status"], "present")
        self.assertEqual(got[f"{IRIS}/blobs/blob0"]["size"], 10)
        self.assertEqual(got[f"{IRIS}/blobs/blob1"]["status"], "absent")
        link = got[f"{IRIS}/snapshots/rev/config.yaml"]
        self.assertEqual(link["type"], "symlink")
        self.assertTrue(link["target_lexically_within_root"])
        rel = got[f"{IRIS}/snapshots/rev/depth/config.yaml"]
        self.assertEqual(rel["readlink"], "../../blobs/blob2")
        self.assertEqual(got[QWEN]["status"], "absent")
        self.assertEqual(got[f"{QWEN}/blobs/encoder-config"]["status"], "absent")

    def test_dangling_link_is_reported_as_link_only(self):
        self.seed_iris()
        (self.root / IRIS / "blobs" / "blob0").unlink()
        got = self.entries(ci.build_report(self.env(), home=None))
        self.assertEqual(got[f"{IRIS}/snapshots/rev/config.yaml"]["status"], "present")
        self.assertEqual(got[f"{IRIS}/blobs/blob0"]["status"], "absent")

    def test_intermediate_symlink_is_not_followed(self):
        outside = self.base / "outside"
        (outside / "rev").mkdir(parents=True)
        (outside / "rev" / "config.yaml").write_bytes(b"secret")
        (self.root / IRIS).mkdir()
        os.symlink(outside, self.root / IRIS / "snapshots")
        got = self.entries(ci.build_report(self.env(), home=None))
        entry = got[f"{IRIS}/snapshots/rev/config.yaml"]
        self.assertEqual(entry["status"], "blocked_by_symlink")
        self.assertEqual(entry["stopped_at"], f"{IRIS}/snapshots")
        self.assertNotIn("size", entry)
        self.assertEqual(got[f"{IRIS}/snapshots"]["type"], "symlink")
        self.assertFalse(got[f"{IRIS}/snapshots"]["target_lexically_within_root"])

    def test_leaf_symlink_escaping_root_is_flagged_not_stat_ed(self):
        self.seed_iris()
        leaf = self.root / IRIS / "snapshots" / "rev" / "model.safetensors"
        os.symlink("../../../../../../etc/hosts", leaf)
        entry = self.entries(ci.build_report(self.env(), home=None))[f"{IRIS}/snapshots/rev/model.safetensors"]
        self.assertFalse(entry["target_lexically_within_root"])
        self.assertEqual(entry["type"], "symlink")

    def test_intermediate_non_directory_is_blocked(self):
        (self.root / IRIS).mkdir()
        (self.root / IRIS / "blobs").write_bytes(b"")
        entry = self.entries(ci.build_report(self.env(), home=None))[f"{IRIS}/blobs/blob0"]
        self.assertEqual(entry["status"], "blocked_not_directory")

    @unittest.skipIf(os.geteuid() == 0, "permission bits do not bind root")
    def test_permission_failure_is_inaccessible_not_absent(self):
        self.seed_iris()
        blobs = self.root / IRIS / "blobs"
        blobs.chmod(0)
        self.addCleanup(blobs.chmod, 0o755)
        entry = self.entries(ci.build_report(self.env(), home=None))[f"{IRIS}/blobs/blob0"]
        self.assertEqual(entry["status"], "inaccessible")
        self.assertEqual(entry["error"], "EACCES")

    def test_unexpected_oserror_is_unknown(self):
        real = os.lstat

        def flaky(path, *a, **k):
            if str(path).endswith("blob1"):
                raise OSError(5, "io")
            return real(path, *a, **k)

        self.seed_iris()
        with mock.patch.object(ci.os, "lstat", flaky):
            entry = self.entries(ci.build_report(self.env(), home=None))[f"{IRIS}/blobs/blob1"]
        self.assertEqual(entry["status"], "unknown")

    def test_root_absent_file_and_symlinked_root(self):
        gone = self.base / "nothing" / "here" / "hub"
        report = ci.build_report(self.env(HF_HUB_CACHE=str(gone)), home=None)
        self.assertEqual((report["root_status"], report["entries"]), ("absent", []))
        afile = self.base / "a" / "b" / "file"
        afile.parent.mkdir(parents=True)
        afile.write_bytes(b"")
        self.assertEqual(ci.build_report(self.env(HF_HUB_CACHE=str(afile)), home=None)["root_status"], "not_directory")
        self.seed_iris()
        alias = self.base / "alias" / "x" / "hub"
        alias.parent.mkdir(parents=True)
        os.symlink(self.root, alias)
        report = ci.build_report(self.env(HF_HUB_CACHE=str(alias)), home=None)
        self.assertEqual(report["root_status"], "RESOLVED")
        self.assertEqual(report["root"]["resolved_root"], str(self.root))
        self.assertEqual(self.entries(report)[f"{IRIS}/blobs/blob0"]["status"], "present")

    def test_symlinked_root_resolving_to_overbroad_is_rejected(self):
        alias = self.base / "alias" / "x" / "hub"
        alias.parent.mkdir(parents=True)
        os.symlink("/", alias)
        report = ci.build_report(self.env(HF_HUB_CACHE=str(alias)), home=None)
        self.assertEqual((report["root_status"], report["entries"]), ("REJECTED", []))

    def test_only_lstat_is_used_and_calls_are_bounded(self):
        self.seed_iris()
        calls = []
        real = os.lstat

        def counting(path, *a, **k):
            calls.append(str(path))
            return real(path, *a, **k)

        def forbidden(*a, **k):
            raise AssertionError("directory listing / reading is forbidden")

        with mock.patch.object(ci.os, "lstat", counting), mock.patch.object(
            os, "listdir", forbidden
        ), mock.patch.object(os, "scandir", forbidden), mock.patch.object(os, "walk", forbidden), mock.patch(
            "builtins.open", forbidden
        ):
            ci.build_report(self.env(), home=None)
        root = str(self.root)
        # realpath() of the root lstat()s only the root's own ancestors; everything else must sit on
        # an allowlisted path.
        self.assertLessEqual(len(calls), 20 + 84 * 5)
        allowed = {os.path.join(root, rel) for rel in ci.ALLOWLIST}
        for call in calls:
            if root.startswith(call + os.sep) or call == root:
                continue
            self.assertTrue(any(a == call or a.startswith(call + os.sep) for a in allowed), call)


class PrivacyTests(Fixture):
    SENTINELS = ("hf_sensitive_sentinel", "https://user:pw@example.invalid/x?token=abc", "ghp_SECRETTOKEN")

    def test_invalid_hf_values_are_never_echoed(self):
        for var in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
            for sentinel in self.SENTINELS:
                out = self.runner_temp / f"inv-{var}-{abs(hash(sentinel))}"
                before = snapshot_tree(self.base)
                env = {"RUNNER_TEMP": str(self.runner_temp), "HOME": str(self.base), var: sentinel}
                with contextlib.redirect_stdout(io.StringIO()) as stdout:
                    code = ci.main(["--output-dir", str(out)], env=env)
                text = (out / "cache-inventory.json").read_text(encoding="utf-8") + stdout.getvalue()
                self.assertEqual(code, 3)
                report = json.loads((out / "cache-inventory.json").read_text(encoding="utf-8"))
                self.assertEqual(report["root_status"], "REJECTED")
                self.assertEqual(report["resolver"]["chosen_variable"], var)
                self.assertTrue(report["resolver"]["reason"])
                for secret in self.SENTINELS:
                    self.assertNotIn(secret, text)
                self.assertNotIn("pw@", text)
                added = [r for r in snapshot_tree(self.base) if r not in before]
                self.assertTrue(all(r[0] == "runner_temp" or r[0].startswith("runner_temp/" + out.name) for r in added))

    def test_unset_and_nonfilesystem_values_never_leak_unrelated_env(self):
        env = self.env(GITHUB_TOKEN="ghp_SECRETTOKEN", HF_TOKEN="hf_SECRETTOKEN", HF_HOME="hf_sensitive_sentinel")
        env["HF_HUB_CACHE"] = ""
        text = json.dumps(ci.build_report(env, home=None))
        for secret in ("ghp_SECRETTOKEN", "hf_SECRETTOKEN", "hf_sensitive_sentinel"):
            self.assertNotIn(secret, text)

    def test_outside_root_leaf_link_text_is_redacted_and_never_followed(self):
        self.seed_iris()
        leaf = self.root / IRIS / "snapshots" / "rev" / "model.safetensors"
        os.symlink("https://user:pw@example.invalid/x?token=abc", leaf)
        outside = self.base / "outside_secret_dir"
        outside.mkdir()
        (outside / "f").write_bytes(b"s")
        os.symlink(outside / "f", self.root / IRIS / "snapshots" / "rev" / "depth" / "model.safetensors")
        before = snapshot_tree(self.base)
        report = ci.build_report(self.env(), home=None)
        text = json.dumps(report)
        self.assertNotIn("example.invalid", text)
        self.assertNotIn("outside_secret_dir", text)
        uri = self.entries(report)[f"{IRIS}/snapshots/rev/model.safetensors"]
        self.assertTrue(uri["readlink_redacted"])
        self.assertNotIn("readlink", uri)
        entry = self.entries(report)[f"{IRIS}/snapshots/rev/depth/model.safetensors"]
        self.assertFalse(entry["target_lexically_within_root"])
        for entry in (entry,):
            self.assertTrue(entry["readlink_redacted"])
            self.assertNotIn("readlink", entry)
        # in-root links keep their text; nothing anywhere was written
        self.assertEqual(self.entries(report)[f"{IRIS}/snapshots/rev/depth/config.yaml"]["readlink"], "../../blobs/blob2")
        self.assertEqual(snapshot_tree(self.base), before)

    def test_symlinked_root_target_text_is_redacted(self):
        self.seed_iris()
        alias = self.base / "alias" / "x" / "hub"
        alias.parent.mkdir(parents=True)
        os.symlink(self.root, alias)
        record = ci.build_report(self.env(HF_HUB_CACHE=str(alias)), home=None)["root"]
        self.assertTrue(record["readlink_redacted"])
        self.assertNotIn("readlink", record)

    def test_report_size_is_bounded(self):
        self.seed_iris()
        out = self.runner_temp / "inv-size"
        self.assertEqual(self.run_main_for(out), 0)
        self.assertLess((out / "cache-inventory.json").stat().st_size, 32 * 1024)

    def test_all_present_report_stays_under_cap_and_tallies_groups(self):
        paths = set(ci.ALLOWLIST)
        parents = {os.path.dirname(p) for p in paths}
        for rel in sorted(paths):
            full = self.root / rel
            if rel in parents:
                full.mkdir(parents=True, exist_ok=True)
        for rel in sorted(paths - parents):
            full = self.root / rel
            full.parent.mkdir(parents=True, exist_ok=True)
            if "/snapshots/rev/" in rel:
                os.symlink("../../blobs/x", full)
            else:
                full.write_bytes(b"z")
        out = self.runner_temp / "inv-full"
        self.assertEqual(self.run_main_for(out), 0)
        raw = (out / "cache-inventory.json").read_bytes()
        self.assertLess(len(raw), ci.MAX_REPORT_BYTES)
        report = json.loads(raw.decode("utf-8"))
        self.assertEqual(sum(g.get("present", 0) for g in report["groups"].values()), 84)
        self.assertEqual({e["group"] for e in report["entries"]}, set(ci.GROUP_COUNTS))

    def test_oversized_report_becomes_an_error_artifact(self):
        out = self.runner_temp / "inv-big"
        with mock.patch.object(ci, "MAX_REPORT_BYTES", 10):
            self.assertEqual(self.run_main_for(out), 1)
        self.assertEqual(json.loads((out / "cache-inventory.json").read_text())["root_status"], "ERROR")

    def run_main_for(self, out):
        with contextlib.redirect_stdout(io.StringIO()):
            return ci.main(["--output-dir", str(out)], env=self.env())


class MainTests(Fixture):
    def run_main(self, out, **env_extra):
        env = self.env(**env_extra)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = ci.main(["--output-dir", str(out)], env=env)
        return code

    def test_zero_cache_writes_and_report_is_valid_utf8_json(self):
        self.seed_iris()
        before = snapshot_tree(self.root)
        out = self.runner_temp / "inv-1"
        code = self.run_main(out, GITHUB_RUN_ID="7", GITHUB_SHA="a" * 40, AWS_SECRET_ACCESS_KEY="sekrit-token")
        self.assertEqual(code, 0)
        self.assertEqual(snapshot_tree(self.root), before)
        raw = (out / "cache-inventory.json").read_bytes()
        text = raw.decode("utf-8")
        report = json.loads(text)
        self.assertEqual(report["diagnosed_failed_run_id"], "39687007308")
        self.assertEqual(report["diagnostic"]["run_id"], "7")
        self.assertEqual(report["allowlist_count"], 84)
        self.assertNotIn("sekrit-token", text)
        self.assertNotIn("PATH", report["resolver"])
        self.assertEqual(sorted(p.name for p in out.iterdir()), ["cache-inventory.json"])

    def test_unknown_fallback_writes_report_without_touching_filesystem_roots(self):
        env = {"RUNNER_TEMP": str(self.runner_temp), "HOME": str(self.base)}
        out = self.runner_temp / "inv-unknown"
        with contextlib.redirect_stdout(io.StringIO()):
            code = ci.main(["--output-dir", str(out)], env=env)
        report = json.loads((out / "cache-inventory.json").read_text(encoding="utf-8"))
        self.assertEqual((code, report["root_status"], report["entries"]), (0, "UNKNOWN", []))
        self.assertIsNone(report["root"])

    def test_rejected_root_writes_report_and_exits_nonzero(self):
        out = self.runner_temp / "inv-rej"
        code = self.run_main(out, HF_HUB_CACHE="/Users")
        report = json.loads((out / "cache-inventory.json").read_text(encoding="utf-8"))
        self.assertEqual((code, report["root_status"]), (3, "REJECTED"))

    def test_internal_failure_still_writes_artifact(self):
        out = self.runner_temp / "inv-err"
        with mock.patch.object(ci, "build_report", side_effect=RuntimeError("boom /secret/path")):
            code = self.run_main(out)
        text = (out / "cache-inventory.json").read_text(encoding="utf-8")
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(text)["error_type"], "RuntimeError")
        self.assertNotIn("/secret/path", text)

    def test_output_dir_rules(self):
        existing = self.runner_temp / "exists"
        existing.mkdir()
        for bad in (self.base / "elsewhere", "relative-dir", self.runner_temp / ".." / "escape"):
            with self.assertRaises(SystemExit, msg=str(bad)):
                self.run_main(bad)
        with self.assertRaises(FileExistsError):
            self.run_main(existing)
        self.assertEqual(list(existing.iterdir()), [])
        with self.assertRaises(SystemExit):
            self.run_main(self.root / "inside-cache")
        with self.assertRaises(SystemExit):
            ci.main(["--output-dir", str(self.runner_temp / "x")], env={"HF_HUB_CACHE": str(self.root)})
        self.assertFalse((self.root / "inside-cache").exists())

    def test_runner_temp_inside_cache_is_refused(self):
        inside = self.root / "tmp"
        inside.mkdir()
        env = self.env(RUNNER_TEMP=str(inside))
        with self.assertRaises(SystemExit):
            ci.main(["--output-dir", str(inside / "inv")], env=env)
        self.assertEqual(list(inside.iterdir()), [])

    def test_unknown_arguments_are_rejected(self):
        with self.assertRaises(SystemExit):
            ci.main(["--output-dir", str(self.runner_temp / "n"), "--path", "/"], env=self.env())


if __name__ == "__main__":
    unittest.main(verbosity=2)
