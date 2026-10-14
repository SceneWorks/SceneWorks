"""CPU-only refusal and identity tests for the separate Mac2 fixture route."""
import importlib.util
from argparse import Namespace
import fcntl
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch


source = Path(__file__).with_name("h3-veda-mlx-fixtures.py")
spec = importlib.util.spec_from_file_location("h3_veda_mlx_fixtures", source)
f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f)


class FixturePolicyTests(unittest.TestCase):
    def test_shared_python_binding_consumes_original_bootstrap_and_rejects_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / "scripts").mkdir()
            module = root / "scripts/h3-veda-owned-python.mjs"
            module.write_text("exact shared source")
            run = root / "owned"
            (run / "tools").mkdir(parents=True)
            executable = root / "python3.12"
            executable.write_bytes(b"actual owned interpreter fixture")
            attestation = run / "tools/python-3.12-attestation.json"
            value = {"context": {"schemaVersion": 1, "runDir": str(run), "sourceSha256": f.g.file_sha(module), "platform": "darwin", "arch": "arm64"}, "interpreter": {"path": str(executable), "sha256": f.g.file_sha(executable), "version": "3.12.13", "implementation": "cpython"}}
            original = json.dumps(value)
            attestation.write_text(original)
            env = {"S9_GUARD_PYTHON": str(executable), "S9_GUARD_PYTHON_SHA256": f.g.file_sha(executable), "S9_GUARD_PYTHON_ATTESTATION": str(attestation), "S9_GUARD_PYTHON_ATTESTATION_SHA256": f.g.file_sha(attestation)}
            with patch.dict(os.environ, env), patch.object(f.sys, "executable", str(executable)), patch.object(f.sys, "version_info", (3, 12, 13)):
                bound = f.python_binding(root, run)
                self.assertEqual(bound["bootstrapAttestation"]["originalJson"], original)
                self.assertEqual(bound["version"], [3, 12, 13])
                for key in ["S9_GUARD_PYTHON_SHA256", "S9_GUARD_PYTHON_ATTESTATION_SHA256"]:
                    with patch.dict(os.environ, {key: "0" * 64}):
                        with self.assertRaisesRegex(RuntimeError, "authority differs"):
                            f.python_binding(root, run)
                with patch.object(f.sys, "version_info", (3, 11, 9)):
                    with self.assertRaisesRegex(RuntimeError, "version differs"):
                        f.python_binding(root, run)
                with patch.object(f.sys, "executable", str(root / "foreign")):
                    with self.assertRaisesRegex(RuntimeError, "different interpreter"):
                        f.python_binding(root, run)
                module.write_text("source mutation")
                with self.assertRaisesRegex(RuntimeError, "context differs"):
                    f.python_binding(root, run)
                module.write_text("exact shared source")
                executable.write_bytes(b"interpreter mutation")
                with self.assertRaisesRegex(RuntimeError, "authority differs"):
                    f.python_binding(root, run)

    def test_rust_toolchain_installs_only_missing_exact_version_and_records_actual_tools(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); output = root / "output"; output.mkdir()
            tools = {name: root / name for name in ("rustc", "cargo")}
            for path in tools.values(): path.write_bytes(b"actual tool bytes")
            calls = []
            missing = False
            def command(argv, **kwargs):
                calls.append(argv)
                if argv == ["rustup", "toolchain", "list"]:
                    value = "stable-aarch64-apple-darwin (default)\n" if missing else "1.96.0-aarch64-apple-darwin\n"
                elif argv[1] == "which": value = str(tools[argv[-1]])
                else: value = "rustc 1.96.0 (ac68faa20 2026-05-25)" if argv[3] == "rustc" else "cargo 1.96.0 (fixture 2026-05-25)"
                return subprocess.CompletedProcess(argv, 0, value, "")
            with patch.object(f.subprocess, "run", side_effect=command), patch.object(f.subprocess, "Popen") as launch:
                facts = f.rust_toolchain(root, output, {})
                launch.assert_not_called(); self.assertFalse(facts["provision"]["required"])
                self.assertEqual(facts["tools"]["rustc"]["sha256"], f.g.file_sha(tools["rustc"]))
                missing = True; child = launch.return_value; child.pid = 123; child.wait.return_value = 0
                with patch.object(f.g, "pid_birth", return_value="owned-birth"), patch.object(f, "child_closure", return_value={"closed": True}):
                    facts = f.rust_toolchain(root, output, {})
                self.assertTrue(facts["provision"]["required"])
                self.assertEqual(launch.call_args.args[0], ["rustup", "toolchain", "install", "1.96.0", "--profile", "minimal"])
                self.assertEqual(child.wait.call_args.kwargs["timeout"], 600)
                self.assertEqual(facts["provision"]["birth"], "owned-birth")
                self.assertTrue(all("default" not in c and "override" not in c for c in calls))
                child.wait.return_value = 1
                with patch.object(f.g, "pid_birth", return_value="owned-birth"), patch.object(f, "child_closure", return_value={"closed": True}):
                    with self.assertRaisesRegex(RuntimeError, "provision failed"): f.rust_toolchain(root, output, {})
            with patch.object(f.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "1.97.1-aarch64-apple-darwin\n", "")), patch.object(f.subprocess, "Popen") as launch:
                child = launch.return_value; child.pid = 123; child.wait.return_value = 0
                with patch.object(f.g, "pid_birth", return_value="owned-birth"), patch.object(f, "child_closure", return_value={"closed": True}):
                    with self.assertRaisesRegex(RuntimeError, "version differs"): f.rust_toolchain(root, output, {})

    def test_fixed_actual_mac2_and_shared_bootstrap_workflow_preserve_registered_header(self):
        workflow = (source.parent.parent / ".github/workflows/h3-veda-mlx-fixtures.yml").read_bytes()
        self.assertEqual(len(workflow.split(b"permissions:", 1)[0]), 936)
        self.assertEqual(f.g.hashlib.sha256(workflow.split(b"permissions:", 1)[0]).hexdigest(), "fb26dfff779a53d96f7ad96bd6167b93363dd83fb586255f0064a0a1ad127503")
        text = workflow.decode()
        self.assertNotIn("actions/setup-python@", text)
        self.assertIn('node scripts/h3-veda-owned-python.mjs bootstrap --run-dir "$S9_RUN_DIR"', text)
        self.assertIn('"$S9_GUARD_PYTHON" scripts/h3-veda-mlx-fixtures.py', text)
        self.assertEqual(f.PHYSICAL_GPU_ID, "734A72AE-13DA-52A8-B0CF-F97E1D646124")
        self.assertEqual(f.METAL_DEVICE_NAME, "Apple M5 Max")


    def test_exact_five_named_device_functions_and_duration_bounds(self):
        self.assertEqual(len(f.FIXTURES), 5)
        self.assertEqual(len(set(f.FIXTURES)), 5)
        self.assertEqual(f.ADMISSION_SECONDS, 1800)
        self.assertEqual(f.CHILD_SECONDS, 3600)
        self.assertEqual(f.NATIVE_HEAD, "ba6ed0db452ecbed780ab8eeeac883cd9758ed3a")

    def test_exact_fixture_result_requires_one_pass_and_no_ignore(self):
        with tempfile.TemporaryDirectory() as temporary:
            out, err = Path(temporary) / "stdout", Path(temporary) / "stderr"
            err.write_text("")
            good = "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 388 filtered out; finished in 1.00s\n"
            out.write_text(good)
            self.assertTrue(f.fixture_outcome(out, err, 0))
            for log in (good * 2, good.replace("0 ignored", "1 ignored"), good.replace("1 passed", "0 passed"), "running 1 test\n"):
                out.write_text(log)
                self.assertFalse(f.fixture_outcome(out, err, 0))
            out.write_text(good)
            self.assertFalse(f.fixture_outcome(out, err, 101))

    def test_each_admission_needs_five_timed_same_device_samples_and_actor_proofs(self):
        with tempfile.TemporaryDirectory() as temporary:
            source_receipt = Path(temporary) / "phase.json"
            source_receipt.write_text("{}")
            identity = {"runId": 123, "runAttempt": 1, "runnerId": 5281, "jobId": 456, "sourceHead": "a" * 40, "jobName": "mac2", "workflowRef": f.WORKFLOW + "refs/tags/codex-test"}
            proof = {"schemaVersion": 1, **identity, "worker": {"pid": 321}, "ancestry": [{"pid": 11}, {"pid": 321}]}
            samples = [{"observedAt": 1100 + i * 500, "admitted": True, "driverAvailable": True, "metalDevice": "CPU_TEST_METAL", "platformUuid": "CPU_TEST_UUID", "actors": [], "memoryFreePercent": 33, "actionsAncestorProof": proof} for i in range(5)]
            phase = {"admitted": True, "physicalGpuId": "CPU_TEST_UUID", "queueObservedAt": 1000, "observedAt": 3200, "driverSamples": samples, "sourceReceipt": {"path": str(source_receipt), "sha256": f.g.file_sha(source_receipt)}}
            f.validate_phase(phase, "CPU_TEST_UUID", "CPU_TEST_METAL", identity)
            for changed in ({"observedAt": 1101}, {"platformUuid": "foreign"}, {"memoryFreePercent": 5}, {"actionsAncestorProof": None}):
                mutated = {**samples[1], **changed}
                phase["driverSamples"] = [samples[0], mutated, *samples[2:]]
                with self.assertRaises(RuntimeError):
                    f.validate_phase(phase, "CPU_TEST_UUID", "CPU_TEST_METAL", identity)
            phase["driverSamples"] = samples
            phase["queueObservedAt"] = 4000
            with self.assertRaisesRegex(RuntimeError, "queue observation order"):
                f.validate_phase(phase, "CPU_TEST_UUID", "CPU_TEST_METAL", identity)
            phase["queueObservedAt"] = 1000
            source_receipt.write_text("changed")
            with self.assertRaisesRegex(RuntimeError, "receipt bytes changed"):
                f.validate_phase(phase, "CPU_TEST_UUID", "CPU_TEST_METAL", identity)

    def test_metallib_and_binary_both_require_unique_pinned_build_messages(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)
            binary = target / "debug" / "deps" / "mlx_gen_minimax_h3-123"
            shader_dir = target / "debug" / "build" / "pmetal" / "out"
            binary.parent.mkdir(parents=True)
            (shader_dir / "build" / "lib").mkdir(parents=True)
            binary.write_bytes(b"actual binary")
            shader = shader_dir / "build" / "lib" / "mlx.metallib"
            shader.write_bytes(b"actual metallib")
            package = "git+https://github.com/michaeltrefry/mlx-rs?rev=" + "a" * 40 + "#pmetal-mlx-sys@1.2.3"
            lines = [json.dumps({"reason": "compiler-artifact", "target": {"kind": ["lib"], "name": "mlx_gen_minimax_h3"}, "executable": str(binary)}), json.dumps({"reason": "build-script-executed", "package_id": package, "out_dir": str(shader_dir)})]
            bound, metal, provenance = f.parse_build(lines, target, package)
            self.assertEqual(bound["bytes"], 13)
            self.assertEqual(metal["bytes"], 15)
            self.assertEqual(provenance["packageId"], package)
            for bad in (lines[:1], lines[1:], lines + lines[1:], lines + lines[:1]):
                with self.assertRaisesRegex(RuntimeError, "one exact"):
                    f.parse_build(bad, target, package)
            shader.unlink()
            with self.assertRaises(FileNotFoundError):
                f.parse_build(lines, target, package)

    def test_locked_metallib_package_rejects_unpinned_or_duplicate_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            lock = Path(temporary) / "Cargo.lock"
            block = 'version = 4\n\n[[package]]\nname = "pmetal-mlx-sys"\nversion = "1.2.3"\nsource = "git+https://github.com/michaeltrefry/mlx-rs?rev=' + 'a' * 40 + '#' + 'a' * 40 + '"\n'
            lock.write_text(block)
            self.assertIn("#pmetal-mlx-sys@1.2.3", f.package_identity(lock))
            lock.write_text(block.replace("?rev=" + 'a' * 40, "?branch=main"))
            with self.assertRaisesRegex(RuntimeError, "not exact pinned"):
                f.package_identity(lock)
            lock.write_text(block + block[block.index("[[package]]"):])
            with self.assertRaisesRegex(RuntimeError, "ambiguous"):
                f.package_identity(lock)

    def test_symlinked_source_artifact_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "real").write_bytes(b"bytes")
            (root / "link").symlink_to(root / "real")
            with self.assertRaisesRegex(RuntimeError, "unsafe"):
                f.file_binding(root / "link")

    def test_source_head_and_clean_checkout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            (root / "test").write_text("frozen\n")
            subprocess.run(["git", "-C", str(root), "add", "test"], check=True)
            subprocess.run(["git", "-C", str(root), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "test"], check=True)
            head = f.shell(["git", "rev-parse", "HEAD"], root)
            self.assertEqual(f.exact_source(root, head)["head"], head)
            with self.assertRaisesRegex(RuntimeError, "source head"):
                f.exact_source(root, "0" * 40)
            (root / "test").write_text("mutated\n")
            with self.assertRaisesRegex(RuntimeError, "dirty"):
                f.exact_source(root, head)

    def test_actions_owner_rejects_foreign_and_terminal_jobs(self):
        p = f.FixturePhysical.__new__(f.FixturePhysical)
        p.m = {"sourceFreeze": {"sceneworks": {"head": "a" * 40}}}
        env = {"GITHUB_ACTIONS": "true", "GH_TOKEN": "fake-test-token", "GITHUB_REPOSITORY": "SceneWorks/SceneWorks", "GITHUB_EVENT_NAME": "workflow_dispatch", "GITHUB_SHA": "a" * 40, "GITHUB_WORKFLOW_REF": f.WORKFLOW + "refs/tags/codex-fixtures-test", "GITHUB_JOB": "mac2", "RUNNER_NAME": f.RUNNER, "GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "1"}
        run = {"id": 123, "run_attempt": 1, "event": "workflow_dispatch", "head_sha": "a" * 40, "status": "in_progress", "conclusion": None, "name": f.WORKFLOW_NAME}
        job = {"id": 456, "name": "mac2", "runner_name": f.RUNNER, "runner_id": f.RUNNER_ID, "run_id": 123, "run_attempt": 1, "head_sha": "a" * 40, "status": "in_progress", "conclusion": None}
        raw = {"orgRunners": [{"runners": [{"id": f.RUNNER_ID, "name": f.RUNNER, "status": "online", "busy": True}]}]}
        def command(argv, timeout):
            return json.dumps(run if "/jobs?" not in argv[-1] else [{"jobs": [job]}])
        with patch.dict(os.environ, env, clear=True), patch.object(f.g, "command", side_effect=command):
            self.assertEqual(p.actions_owner(raw), f.RUNNER_ID)
            self.assertEqual(p.actions_identity["jobId"], 456)
            for key, value in (("GITHUB_WORKFLOW_REF", "SceneWorks/SceneWorks/.github/workflows/h3-veda-terminal.yml@refs/heads/main"), ("GITHUB_SHA", "b" * 40), ("RUNNER_NAME", "nax-macos")):
                with patch.dict(os.environ, {key: value}):
                    with self.assertRaises(RuntimeError):
                        p.actions_owner(raw)
            for changes in ({"status": "completed"}, {"head_sha": "b" * 40}, {"runner_id": 999}, {"name": "other"}):
                original = job.copy()
                job.update(changes)
                with self.assertRaises(RuntimeError):
                    p.actions_owner(raw)
                job.clear(); job.update(original)
            run["conclusion"] = "success"
            with self.assertRaises(RuntimeError):
                p.actions_owner(raw)

    def test_uncertain_child_closure_never_releases_locks(self):
        class Physical:
            def __init__(self): self.released = False
            def owned(self): return True
            def close(self): self.released = True
        physical = Physical()
        with self.assertRaisesRegex(RuntimeError, "closure absent"):
            f.safe_close(physical, False)
        self.assertFalse(physical.released)
        f.safe_close(physical, True)
        self.assertTrue(physical.released)

    def test_pid_birth_reuse_refuses_signal(self):
        class Child:
            pid = 12345
            def poll(self): return None
        with patch.object(f.g, "pid_birth", return_value="replacement"), patch.object(f.os, "killpg") as kill:
            with self.assertRaisesRegex(RuntimeError, "birth changed"):
                f.child_closure(Child(), "original", 9999999)
            kill.assert_not_called()

    def test_actual_owned_cpu_process_group_drains_without_signaling_foreign_child(self):
        owned = subprocess.Popen(["/bin/sleep", "30"], start_new_session=True)
        foreign = subprocess.Popen(["/bin/sleep", "30"], start_new_session=True)
        try:
            birth = f.g.pid_birth(owned.pid)
            self.assertIsNotNone(birth)
            closure = f.child_closure(owned, birth, time.monotonic() + 10)
            self.assertTrue(closure["reaped"] and closure["descendantsDrained"])
            self.assertFalse(f.group_alive(owned.pid))
            self.assertIsNone(foreign.poll())
        finally:
            if owned.poll() is None:
                owned.terminate(); owned.wait(timeout=5)
            if foreign.poll() is None:
                foreign.terminate(); foreign.wait(timeout=5)

    @unittest.skipUnless(hasattr(os, "fork"), "Mac/POSIX lock inheritance proof")
    def test_independent_watchdog_retains_both_actual_cpu_flocks_until_clean_closure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = [root / "global-one.lock", root / "global-two.lock"]
            handles = [path.open("a+b") for path in paths]
            for handle in handles:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            parent_birth = f.g.pid_birth(os.getpid())
            self.assertIsNotNone(parent_birth)
            with patch.object(f.g, "canonical_mac_locks", return_value=paths):
                watcher_pid = os.fork()
                if watcher_pid == 0:
                    try:
                        os._exit(f.watchdog_locks(root, os.getppid(), parent_birth, [h.fileno() for h in handles], .01))
                    except BaseException:
                        os._exit(2)
                try:
                    time.sleep(.05)
                    for handle in handles:
                        handle.close()
                    with paths[0].open("a+b") as challenger:
                        with self.assertRaises(BlockingIOError):
                            fcntl.flock(challenger, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    f.g.atomic(root / "watchdog-stop.json", {"childTreeClosed": True, "physicalLocksClosedByController": True})
                    child, status = os.waitpid(watcher_pid, 0)
                    self.assertEqual(child, watcher_pid)
                    self.assertEqual(os.waitstatus_to_exitcode(status), 0)
                    self.assertTrue((root / "watchdog-closure.json").exists())
                    with paths[0].open("a+b") as challenger:
                        fcntl.flock(challenger, fcntl.LOCK_EX | fcntl.LOCK_NB)
                finally:
                    for handle in handles:
                        if not handle.closed:
                            handle.close()
                    try:
                        os.kill(watcher_pid, 9)
                    except ProcessLookupError:
                        pass

    def test_watchdog_controller_loss_drains_only_exact_child_birth(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = [root / "one.lock", root / "two.lock"]
            handles = [path.open("a+b") for path in paths]
            for handle in handles:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            f.g.atomic(root / "running.json", {"active": {"pid": 98765, "birth": "owned-birth"}})
            calls = [None, "owned-birth", None]
            with patch.object(f.g, "canonical_mac_locks", return_value=paths), patch.object(f.g, "pid_birth", side_effect=lambda _pid: calls.pop(0) if calls else None), patch.object(f.os, "getpgid", return_value=98765), patch.object(f.os, "killpg") as kill, patch.object(f, "group_alive", return_value=False):
                try:
                    self.assertEqual(f.watchdog_locks(root, 123, "parent-birth", [h.fileno() for h in handles], .001), 0)
                    kill.assert_called_once_with(98765, f.signal.SIGTERM)
                    proof = json.loads((root / "watchdog-closure.json").read_text())
                    self.assertEqual(proof["reason"], "controller_lost_owned_child_drained")
                    self.assertEqual(proof["childBirth"], "owned-birth")
                finally:
                    for handle in handles:
                        handle.close()

    def test_watchdog_controller_loss_without_device_launch_releases_only_after_durable_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = [root / "one.lock", root / "two.lock"]
            handles = [path.open("a+b") for path in paths]
            for handle in handles:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            f.g.atomic(root / "running.json", {"fixtures": [], "accepted": False})
            with patch.object(f.g, "canonical_mac_locks", return_value=paths), patch.object(f.g, "pid_birth", return_value=None), patch.object(f.os, "killpg") as kill:
                try:
                    self.assertEqual(f.watchdog_locks(root, 123, "parent-birth", [h.fileno() for h in handles], .001), 0)
                    kill.assert_not_called()
                    self.assertEqual(json.loads((root / "watchdog-closure.json").read_text())["reason"], "controller_lost_before_or_after_closed_device_child")
                finally:
                    for handle in handles:
                        handle.close()

    def test_watchdog_ambiguous_launch_intent_retains_locks(self):
        class StopAfterUnresolved(Exception):
            pass
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = [root / "one.lock", root / "two.lock"]
            handles = [path.open("a+b") for path in paths]
            for handle in handles:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            f.g.atomic(root / "running.json", {"launchIntent": {"fixture": f.FIXTURES[0], "state": "pre_spawning_device_child"}})
            with patch.object(f.g, "canonical_mac_locks", return_value=paths), patch.object(f.g, "pid_birth", return_value=None), patch.object(f.os, "killpg") as kill, patch.object(f.time, "sleep", side_effect=StopAfterUnresolved):
                try:
                    with self.assertRaises(StopAfterUnresolved):
                        f.watchdog_locks(root, 123, "parent-birth", [h.fileno() for h in handles], .001)
                    kill.assert_not_called()
                    self.assertTrue((root / "watchdog-unresolved.json").exists())
                finally:
                    for handle in handles:
                        handle.close()

    def test_watchdog_changed_child_birth_never_signals_foreign_process(self):
        class StopAfterUnresolved(Exception):
            pass
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = [root / "one.lock", root / "two.lock"]
            handles = [path.open("a+b") for path in paths]
            for handle in handles:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            f.g.atomic(root / "running.json", {"active": {"pid": 98765, "birth": "original"}})
            with patch.object(f.g, "canonical_mac_locks", return_value=paths), patch.object(f.g, "pid_birth", side_effect=lambda pid: None if pid == 123 else "replacement"), patch.object(f.os, "killpg") as kill, patch.object(f.time, "sleep", side_effect=StopAfterUnresolved):
                try:
                    with self.assertRaises(StopAfterUnresolved):
                        f.watchdog_locks(root, 123, "parent-birth", [h.fileno() for h in handles], .001)
                    kill.assert_not_called()
                    self.assertEqual(json.loads((root / "watchdog-unresolved.json").read_text())["reason"], "orphan_identity_or_closure_unproven")
                finally:
                    for handle in handles:
                        handle.close()

    def test_watchdog_cli_clean_closure_exits_zero(self):
        argv = [str(source), "--watchdog", "--output", "/tmp/cpu-only", "--parent-pid", "123", "--parent-birth", "birth", "--lock-fds", "3,4"]
        with patch.object(f.sys, "argv", argv), patch.object(f.signal, "signal"), patch.object(f, "watchdog_locks", return_value=0):
            with self.assertRaises(SystemExit) as exit_result:
                f.main()
            self.assertEqual(exit_result.exception.code, 0)

    def test_workflow_header_and_native_route_are_fixed(self):
        workflow = (source.parent.parent / ".github/workflows/h3-veda-mlx-fixtures.yml").read_text()
        self.assertIn("runs-on: [self-hosted, macOS, ARM64, rw-mage]", workflow)
        self.assertIn("timeout-minutes: 75", workflow)
        self.assertIn("if: always()", workflow)
        self.assertIn("scripts/h3-veda-mlx-fixtures.py", workflow)
        self.assertIn(f.NATIVE_HEAD, workflow)
        self.assertNotIn("workflow_run:", workflow)

    def staging_fixture(self, root, sha="a" * 40):
        sw, inf = root / "sceneworks", root / "inference"
        sw.mkdir(); inf.mkdir()
        run = root / ".h3-veda-epic-25140" / sha / f.RUNNER
        args = Namespace(sceneworks=str(sw), inference=str(inf), run_dir=str(run), sceneworks_sha=sha, inference_sha=f.NATIVE_HEAD, expected_runner=f.RUNNER, physical_gpu_id=f.PHYSICAL_GPU_ID, metal_device_name=f.METAL_DEVICE_NAME)
        env = {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": "SceneWorks/SceneWorks", "GITHUB_EVENT_NAME": "workflow_dispatch", "GITHUB_WORKSPACE": str(root), "GITHUB_SHA": sha, "GITHUB_WORKFLOW_REF": f.WORKFLOW + "refs/heads/cpu-test", "GITHUB_JOB": "mac2", "RUNNER_NAME": f.RUNNER}
        return args, env, sw, inf, run

    def test_actual_failed_mac2_stage_path_is_source_bound_and_not_mac1_epic(self):
        workspace = Path("/Users/MTrefry/actions-runner-nax/_work/SceneWorks/SceneWorks")
        sha = "00d1626c1af72266432725d7bf6c0cf07b332e60"
        run = workspace / ".h3-veda-epic-25140" / sha / f.RUNNER
        args = Namespace(run_dir=str(run), sceneworks_sha=sha, expected_runner=f.RUNNER)
        env = {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": "SceneWorks/SceneWorks", "GITHUB_EVENT_NAME": "workflow_dispatch", "GITHUB_WORKSPACE": str(workspace), "GITHUB_SHA": sha, "GITHUB_WORKFLOW_REF": f.WORKFLOW + "refs/heads/codex/s9-mac2-mlx-fixtures-00d1626c1", "GITHUB_JOB": "mac2", "RUNNER_NAME": f.RUNNER}
        self.assertFalse(run.is_relative_to(f.g.EPIC))
        with patch.dict(os.environ, env):
            self.assertEqual(f.actions_stage_root(args, workspace / "sceneworks", workspace / "inference")["runDir"], str(run))

    def test_actions_stage_rejects_arbitrary_stale_cross_runner_and_source_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); args, env, sw, inf, run = self.staging_fixture(root)
            with patch.dict(os.environ, env):
                self.assertEqual(f.actions_stage_root(args, sw, inf)["runDir"], str(run))
                for key, value in [("GITHUB_ACTIONS", "false"), ("GITHUB_WORKSPACE", str(root / "foreign")), ("GITHUB_SHA", "b" * 40), ("GITHUB_JOB", "foreign"), ("RUNNER_NAME", "nax-macos"), ("GITHUB_WORKFLOW_REF", "foreign")]:
                    with patch.dict(os.environ, {key: value}):
                        with self.assertRaises(RuntimeError): f.actions_stage_root(args, sw, inf)
                for bad in [f.g.EPIC / "arbitrary", root / "arbitrary", run / "child", run.parent / "nax-macos", run.parent.parent / ("b" * 40) / f.RUNNER]:
                    args.run_dir = str(bad)
                    with self.assertRaises(RuntimeError): f.actions_stage_root(args, sw, inf)
                args.run_dir = str(run)
                with self.assertRaisesRegex(RuntimeError, "checkout roots"): f.actions_stage_root(args, root / "foreign-source", inf)

    def test_unapproved_native_head_refuses_before_build_or_physical_claim(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); args, env, sw, inf, run_dir = self.staging_fixture(root)
            args.inference_sha = "b" * 40
            with patch.dict(os.environ, env), patch.object(f.subprocess, "Popen") as launch:
                self.assertEqual(f.run(args), 1); launch.assert_not_called()
            receipt = json.loads((run_dir / "mlx-five-fixtures" / "receipt.json").read_text())
            self.assertFalse(receipt["accepted"]); self.assertIn("unapproved native source", receipt["error"]); self.assertEqual(receipt["fixtures"], [])

    def test_wrong_mac2_uuid_or_metal_name_refuses_before_build_or_lock_claim(self):
        for uuid, metal in [("foreign UUID", f.METAL_DEVICE_NAME), (f.PHYSICAL_GPU_ID, "foreign Metal")]:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve(); args, env, sw, inf, run_dir = self.staging_fixture(root)
                args.physical_gpu_id, args.metal_device_name = uuid, metal
                with patch.dict(os.environ, env), patch.object(f.subprocess, "Popen") as launch, patch.object(f.FixturePhysical, "claim") as claim:
                    self.assertEqual(f.run(args), 1); launch.assert_not_called(); claim.assert_not_called()
                self.assertIn("unapproved Mac2 physical identity", json.loads((run_dir / "mlx-five-fixtures/receipt.json").read_text())["error"])

    def test_symlinked_evidence_parent_refuses_before_any_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); args, env, sw, inf, run = self.staging_fixture(root)
            actual = root / "actual"; actual.mkdir(); (root / ".h3-veda-epic-25140").symlink_to(actual, target_is_directory=True)
            with patch.dict(os.environ, env):
                with self.assertRaisesRegex(RuntimeError, "non-symlink|symlink ancestor"): f.run(args)
            self.assertFalse((actual / "mlx-five-fixtures").exists())


if __name__ == "__main__":
    unittest.main()
