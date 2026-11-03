"""Ordinary metadata/buffer contracts only: never import or initialize a model runtime."""
import hashlib
import importlib.util
import shlex
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

    def test_actual_workflow_path_initializers_run_without_models(self):
        # Execute only the embedded stdlib path initialization from each assembled workflow.
        for name in ("cv-provider-validation.yml", "cv-oracle-provision.yml"):
            workflow = (SCRIPTS.parent / ".github/workflows" / name).read_text()
            step = workflow.split("      - name: Select run-private provisioning paths\n", 1)[1].split("      - name:", 1)[0]
            self.assertIn("shell: node {0}", step)
            code = step.split("        run: |\n", 1)[1]
            code = "\n".join(line.removeprefix("          ") for line in code.splitlines())
            output = self.root / (name + ".env")
            env = dict(os.environ, RUNNER_TEMP=str(self.root / "runner space"),
                       GITHUB_RUN_ID="123", GITHUB_RUN_ATTEMPT="2", GITHUB_ENV=str(output))
            subprocess.run(["node", "-e", code], cwd=SCRIPTS.parent, env=env, check=True)
            values = dict(line.split("=", 1) for line in output.read_text().splitlines())
            self.assertTrue(values)
            for value in values.values():
                self.assertTrue(value.startswith(str(self.root / "runner space")))
                self.assertTrue(value.endswith("-123-2"))
                self.assertFalse(Path(value).exists())

    @unittest.skipUnless(sys.platform == "darwin", "native Mac host interpreter setup")
    def test_actual_mac_python_setup_is_run_private_and_validates_interpreter(self):
        env_file, path_file = self.root / "github env", self.root / "github path"
        env = dict(os.environ, RUNNER_TEMP=str(self.root), GITHUB_JOB="produce",
                   GITHUB_RUN_ID="123", GITHUB_RUN_ATTEMPT="2",
                   GITHUB_ENV=str(env_file), GITHUB_PATH=str(path_file))
        subprocess.run(["bash", str(SCRIPTS / "select-cv-python.sh")], env=env, check=True)
        shim = Path(path_file.read_text().strip())
        self.assertEqual(shim.parent, self.root)
        self.assertTrue((shim / "python").is_symlink())
        self.assertTrue((shim / "python3").is_symlink())
        selected = env_file.read_text().strip().split("=", 1)[1]
        self.assertEqual((shim / "python").resolve(), Path(selected).resolve())
        output = subprocess.check_output([str(shim / "python"), "-I", "-B", "-c", "import sys; print(sys.version_info[:2])"], text=True)
        self.assertEqual(output.strip(), "(3, 12)")
        venv = self.root / "private tools"
        subprocess.run([str(shim / "python"), "-I", "-B", "-m", "venv", "--without-pip", str(venv)], check=True)
        output = subprocess.check_output([str(venv / "bin/python"), "-I", "-B", "-c", "import sys; print(sys.version_info[:2]); assert sys.prefix != sys.base_prefix"], text=True)
        self.assertEqual(output.strip(), "(3, 12)")
        wrong = self.root / "wrong interpreter"
        wrong.mkdir()
        fake = wrong / "python3.12"
        fake.write_text("#!/bin/sh\nexit 19\n")
        fake.chmod(0o755)
        env.update(PATH=str(wrong) + os.pathsep + env["PATH"], GITHUB_RUN_ATTEMPT="3")
        rejected = subprocess.run(["bash", str(SCRIPTS / "select-cv-python.sh")], env=env)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertFalse((self.root / "cv-python-produce-123-3").exists())

    def test_actual_node_provision_step_refuses_before_assets(self):
        workflow = (SCRIPTS.parent / ".github/workflows/cv-provider-validation.yml").read_text()
        step = workflow.split("      - name: Provision pinned model fixtures and verified native runtime\n", 1)[1].split("      - uses:", 1)[0]
        self.assertIn("shell: node {0}", step)
        code = step.split("        run: |\n", 1)[1]
        entry = self.root / "step.js"
        entry.write_text("\n".join(line.removeprefix("          ") for line in code.splitlines()))
        env = dict(os.environ, CV_NATIVE_PYTHON=sys.executable, CV_BACKEND="mlx",
                   CV_SOURCE_SHA="a" * 40, CV_INFERENCE_SHA="b" * 40,
                   CV_PROVISION_ROOT=str(self.root / "assets"), CV_ORACLE_ARTIFACT=str(self.root / "oracle"))
        env.pop("GITHUB_ACTIONS", None)
        result = subprocess.run(["node", str(entry)], cwd=SCRIPTS.parent, env=env, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("explicit Actions dispatch", result.stderr)
        self.assertFalse((self.root / "assets").exists())

    def test_actual_cuda_workflow_arguments_use_standard_admission_before_any_probe(self):
        workflow = (SCRIPTS.parent / ".github/workflows/cv-provider-validation.yml").read_text()
        command = workflow.split("python -I scripts/ci/windows_cuda_execution.py ", 1)[1].split(" -- node scripts/cv-provider-validation.mjs)", 1)[0]
        args = shlex.split(command) + ["--", "node", "scripts/cv-provider-validation.mjs"]
        spec = importlib.util.spec_from_file_location("cv_cuda_admission", SCRIPTS / "ci/windows_cuda_execution.py")
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        # Capture the actual helper's argparse output, stopping before Windows setup/probes.
        captured = []
        parse = helper.argparse.ArgumentParser.parse_args
        class Parsed(Exception):
            pass
        def capture(parser, *_args, **_kwargs):
            captured.append(parse(parser, args))
            raise Parsed()
        with patch.object(helper.argparse.ArgumentParser, "parse_args", capture), \
             patch.object(helper, "prepared_components", side_effect=AssertionError("no GPU setup")), \
             patch.object(helper, "resolve_binding", side_effect=AssertionError("no GPU probe")):
            with self.assertRaises(Parsed):
                helper.main()
        selected = captured[0]
        self.assertEqual(selected.actor, "cv-provider-validation")
        self.assertEqual(selected.admission_policy, helper.STRICT_POLICY)
        self.assertEqual(selected.budget_minutes, 170)
        self.assertEqual((selected.physical_indices, selected.cuda_ordinals, selected.probe_visible), (None, "0", "1"))
        self.assertEqual(selected.command, ["--", "node", "scripts/cv-provider-validation.mjs"])
        env = dict(GITHUB_EVENT_NAME="workflow_dispatch", GITHUB_WORKFLOW_REF="SceneWorks/SceneWorks/.github/workflows/nax-kernel-validation.yml@refs/heads/proof",
                   SCENEWORKS_GPU_ID="1", CUDA_VISIBLE_DEVICES="1", NVIDIA_VISIBLE_DEVICES="1", CUDA_DEVICE_ORDER="PCI_BUS_ID")
        helper.require_outermost(env)
        helper.require_admission_policy(selected.admission_policy, selected.actor, env, selected.physical_indices, selected.cuda_ordinals, selected.probe_visible)
        with self.assertRaisesRegex(RuntimeError, "restricted to load-progress"):
            helper.require_admission_policy(helper.FUNCTIONAL_POLICY, selected.actor, env, None, "0", "1")
        with self.assertRaises(RuntimeError):
            helper.require_outermost({**env, helper.ACTIVE_KEY: "123"})
        self.assertIn('if "%CV_BUILD_ONLY%"=="true" (node scripts/cv-provider-validation.mjs)', workflow)

    def test_nested_manual_dispatch_admits_exact_source_and_all_pin_sites(self):
        source, pin = "a" * 40, "b" * 40
        for name in ("Cargo.toml", "crates/sceneworks-core/Cargo.toml", "crates/sceneworks-worker/Cargo.toml", "crates/sceneworks-memory-adapter/Cargo.toml"):
            file = self.root / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text('runtime = { git = "https://github.com/SceneWorks/inference", rev = "' + pin + '" }')
        (self.root / "Cargo.lock").write_text('source = "git+https://github.com/SceneWorks/inference?rev=' + pin + '#' + pin + '"')
        env = dict(GITHUB_ACTIONS="true", GITHUB_EVENT_NAME="workflow_dispatch", GITHUB_SHA=source,
                   GITHUB_WORKFLOW_REF="SceneWorks/SceneWorks/.github/workflows/nax-kernel-validation.yml@refs/heads/proof",
                   CV_MAC_GO="true")
        with patch.object(cv, "ROOT", self.root), patch.object(cv.subprocess, "check_output", side_effect=[source, ""]):
            cv.admit(source, pin, "mlx", env=env)
        (self.root / "Cargo.lock").write_text('source = "git+https://github.com/SceneWorks/inference?rev=' + pin + '#' + source + '"')
        with patch.object(cv, "ROOT", self.root), patch.object(cv.subprocess, "check_output", side_effect=[source, ""]):
            with self.assertRaisesRegex(ValueError, "resolved lock pins"):
                cv.admit(source, pin, "mlx", env=env)

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
            "cv_provision.py": ["--mode", "oracle-input", "--backend", "mlx", "--public-only"],
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

    def test_actual_mac_oracle_command_stages_public_metadata_fixture(self):
        workflow = (SCRIPTS.parent / ".github/workflows/cv-oracle-provision.yml").read_text()
        command = next(line.strip().removeprefix("run: ") for line in workflow.splitlines()
                       if "scripts/cv_provision.py --mode oracle-input" in line)
        output = self.root / "oracle-input"
        for key, value in {"CV_SOURCE_SHA": "a" * 40, "CV_INFERENCE_SHA": "b" * 40,
                           "CV_ORACLE_INPUT": str(output)}.items():
            command = command.replace("$" + key, value)
        argv = shlex.split(command)
        argv = argv[argv.index("scripts/cv_provision.py"):]
        manifest = {"models": [], "photo": self.asset, "budgets": {"assetDownloadBytes": 1000}}
        acquire = cv.acquire
        def fixture_acquire(*args, **kwargs):
            self.assertFalse(kwargs["use_cache"])
            return acquire(*args, **kwargs, opener=lambda *_a, **_k: Response(self.payload))
        with patch.object(cv, "admit"), patch.object(cv.sys, "argv", argv), \
             patch.object(cv.sys, "platform", "darwin"), patch.object(cv, "load_manifest", return_value=manifest), \
             patch.object(cv, "candidates", side_effect=AssertionError("no external cache access")), \
             patch.object(cv, "acquire", side_effect=fixture_acquire):
            cv.main()
        self.assertEqual((output / "fixtures" / self.asset["name"]).read_bytes(), self.payload)
        receipt = json.loads((output / "provisioning.json").read_text())
        self.assertEqual((receipt["status"], receipt["cachePolicy"]), ("passed", "public-only"))

    def test_public_only_never_inspects_host_cache_and_still_checks_bytes(self):
        with patch.object(cv, "candidates", side_effect=AssertionError("host cache must not be inspected")):
            cv.acquire(self.asset, self.root / "out", self.receipt, 1000, use_cache=False,
                       opener=lambda *_a, **_k: Response(self.payload, len(self.payload)))
            self.assertEqual((self.root / "out").read_bytes(), self.payload)
            self.assertEqual(self.receipt["assets"][0]["origin"], "download")
            with self.assertRaisesRegex(ValueError, "identity"):
                cv.acquire(self.asset, self.root / "bad", self.receipt, 1000, use_cache=False,
                           opener=lambda *_a, **_k: Response(b"x" * len(self.payload)))
            self.assertFalse((self.root / "bad").exists())
            with self.assertRaisesRegex(ValueError, "budget"):
                cv.acquire(self.asset, self.root / "over", self.receipt, 0, use_cache=False,
                           opener=lambda *_a, **_k: self.fail("budget must precede network"))

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
        # The primary tools are part of a complete locked transitive closure, including yaml.
        from cv_oracle_tools import locked_packages
        packages = locked_packages(manifest)
        self.assertTrue({"torch", "torchvision", "ultralytics", "numpy", "safetensors",
                         "opencv-python", "pillow", "pyyaml"} <= {p["name"] for p in packages})


if __name__ == "__main__":
    unittest.main()
