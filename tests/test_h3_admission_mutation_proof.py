"""Driver mechanics ONLY: in-process fake Cargo and disposable files, never Rust proof.

unittest discovery works locally; the existing non-e2e/parity pytest job also discovers these.
No Cargo, models, caches, network, GPU, subprocesses or API ports are used by these tests.
"""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.ci import h3_admission_mutation_proof as proof

ROOT = Path(__file__).resolve().parents[1]
REAL_SOURCE = (ROOT / proof.SOURCE).read_bytes()
REAL_TEST_SOURCE = (ROOT / proof.TEST_SOURCE).read_bytes()
PIN = "1" * 40
HEAD = "2" * 40
# Extra future mechanisms must also run; no frozen corpus/test population count.
NAMES = sorted(proof.REQUIRED | {proof.PREFIX + "future_mechanism"})


def success(names):
    return proof.Result(0, f"running {len(names)} tests\n" + "".join(
        f"test {name} ... ok\n" for name in sorted(names)) +
        f"test result: ok. {len(names)} passed; 0 failed; 0 ignored; 0 measured; 400 filtered out;\n")


def assertion_failure(plan):
    return proof.Result(101, f"running 1 test\ntest {plan.test} ... FAILED\n\n"
        f"thread '{plan.test}' (123) panicked at crates/sceneworks-worker/src/video_admission/tests.rs:1:2:\n"
        f"assertion `left == right` failed: {plan.assertion}\nleft: 1\nright: 2\n"
        "test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 400 filtered out;\n"
        "error: test failed, to rerun pass `-p sceneworks-worker --lib`\n")


class FakeCargo:
    """Not an executable; injection forces unit_fake receipts even on complete success."""
    def __init__(self, root, mode="pass", names=None):
        self.root = root
        self.mode = mode
        self.names = NAMES if names is None else names
        self.plans = {plan.name: plan for plan in proof.mutations(REAL_SOURCE.decode())}
        self.calls = []
        self.seen_mutations = []
        self.baselines = []

    def __call__(self, command, cwd, env, log):
        self.calls.append((command, log.stem, dict(env)))
        if cwd != self.root:
            raise AssertionError("driver escaped fixed checkout")
        if command[:3] == ["git", "rev-parse", "HEAD"]:
            value = "3" * 40 if self.mode == "head-drift" and log.stem.endswith("-checkpoint") else HEAD
            return proof.Result(0, value + "\n")
        if command[:2] == ["git", "diff"]:
            return proof.Result(1 if self.mode == "dirty" else 0, "")
        if "--version" in command or "-vV" in command:
            return proof.Result(0, "unit_fake toolchain; NOT a real Cargo/rustc invocation\n")
        if "--list" in command:
            return proof.Result(0, "".join(f"{name}: test\n" for name in self.names))
        current = (cwd / proof.SOURCE).read_bytes()
        if "--exact" not in command:
            if current != REAL_SOURCE:
                raise AssertionError("baseline did not receive exact restored source")
            self.baselines.append(log.stem)
            if self.mode == "bad-restored" and log.stem.endswith("-restored"):
                return proof.Result(1, "setup failed before test execution\n")
            if self.mode == "bad-baseline":
                return success([])
            if self.mode == "input-drift" and log.stem.endswith("-restored"):
                with (cwd / proof.TEST_SOURCE).open("ab") as stream:
                    stream.write(b"\n// unit_fake outside input edit\n")
            return success(self.names)
        plan = self.plans[log.stem]
        if current != plan.source.encode():
            raise AssertionError("mutation did not reach the actual fixed production file")
        self.seen_mutations.append(plan.name)
        if self.mode == "interrupt":
            raise KeyboardInterrupt("unit_fake interrupted after source mutation")
        if self.mode == "external-edit":
            (cwd / proof.SOURCE).write_bytes(b"// another writer's source\n")
            raise KeyboardInterrupt("unit_fake concurrent writer")
        result = assertion_failure(plan)
        if self.mode == "compiler":
            return proof.Result(101, "error[E0308]: mismatched types\n" + result.text)
        if self.mode == "linker":
            return proof.Result(101, "error: linking with `cc` failed: exit status: 1\n" + result.text)
        if self.mode == "setup":
            return proof.Result(101, "failed to run custom build command\n" + result.text)
        if self.mode == "zero":
            return success([])
        if self.mode == "survived":
            return success([plan.test])
        if self.mode == "wrong-assertion":
            return proof.Result(101, result.text.replace(plan.assertion, "unrelated runtime panic"))
        if self.mode == "no-panic":
            return proof.Result(101, result.text.replace("panicked at", "printed an error at"))
        if self.mode == "wrong-test":
            return proof.Result(101, result.text.replace(plan.test, proof.PREFIX + "unrelated"))
        return result


class H3AdmissionMutationProofTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="h3-proof-python-unit-")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name).resolve()
        self.root = self.base / "checkout"
        self.root.mkdir()
        for relative, data in [(proof.SOURCE, REAL_SOURCE), (proof.TEST_SOURCE, REAL_TEST_SOURCE),
                               (proof.SCRIPT, (ROOT / proof.SCRIPT).read_bytes())]:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        (self.root / "Cargo.lock").write_text(
            'version = 4\n[[package]]\nname = "sceneworks-gen-core"\nversion = "0.1.0"\n'
            f'source = "git+https://github.com/SceneWorks/inference?rev={PIN}#{PIN}"\n')
        self.output = self.base / "artifacts"

    def execute(self, mode="pass", names=None):
        fake = FakeCargo(self.root, mode, names)
        # Any accidental real process launch is a test error, not a fallback.
        with mock.patch.object(proof, "subprocess_runner", side_effect=AssertionError("NO real subprocess permitted")), \
             contextlib.redirect_stdout(io.StringIO()):
            receipt = proof.run_proof(self.root, self.output, runner=fake)
        return receipt, fake

    def test_complete_fake_run_is_explicitly_not_actual_accounting_proof(self):
        receipt, fake = self.execute()
        self.assertEqual(receipt["status"], "unit_fake_passed")
        self.assertEqual(receipt["proof_kind"], "unit_fake")
        self.assertFalse(receipt["actual_accounting_proof"])
        self.assertEqual(receipt["source_commit"], HEAD)
        self.assertEqual(receipt["inference_pin"], PIN)
        self.assertEqual(receipt["source_sha256"], proof.sha(REAL_SOURCE))
        self.assertEqual(receipt["restored_source_sha256"], proof.sha(REAL_SOURCE))
        self.assertEqual((self.root / proof.SOURCE).read_bytes(), REAL_SOURCE)
        self.assertEqual(set(receipt["discovered_tests"]), set(NAMES))
        self.assertEqual(fake.seen_mutations, list(fake.plans))
        self.assertEqual(len(fake.baselines), 1 + len(fake.plans))
        self.assertTrue(all(item["status"] == "killed_by_assertion" and item["restored_baseline"] == "green" for item in receipt["mutations"]))
        self.assertEqual(json.loads((self.output / "receipt.json").read_text()), receipt)
        self.assertTrue((self.output / "events.jsonl").read_text())
        for command, label, env in fake.calls:
            self.assertTrue((self.output / (label + ".log")).read_text().startswith("UNIT_FAKE"))
            self.assertEqual(env["CARGO_NET_OFFLINE"], "true")
            self.assertEqual(env["CARGO_BUILD_JOBS"], "2")
            self.assertEqual(env["CARGO_PROFILE_DEV_STRIP"], "false")
            self.assertEqual(env["CARGO_PROFILE_TEST_STRIP"], "false")
            self.assertEqual(env["CARGO_PROFILE_DEV_BUILD_OVERRIDE_STRIP"], "none")
            self.assertEqual(env["CARGO_PROFILE_TEST_BUILD_OVERRIDE_STRIP"], "none")
            for key in ["HF_HOME", "SCENEWORKS_DATA_DIR", "TMPDIR"]:
                self.assertTrue(Path(env[key]).is_relative_to(self.output))
            if "test" in command:
                self.assertEqual(command[:3], ["/usr/bin/nice", "-n", "10"])
                self.assertIn("--locked", command)
                self.assertIn("sceneworks-worker", command)
                self.assertIn("--lib", command)
        self.assertTrue(all("duration_seconds" in entry and "exit_code" in entry for entry in receipt["commands"]))

    def test_each_mutation_changes_one_known_production_scope(self):
        plans = proof.mutations(REAL_SOURCE.decode())
        self.assertEqual(len({plan.name for plan in plans}), len(plans))
        self.assertEqual(len({plan.source for plan in plans}), len(plans))
        self.assertTrue(all(plan.source.encode() != REAL_SOURCE for plan in plans))
        guards = {plan.name: plan.source for plan in plans}
        self.assertIn(".filter(|_| selector.backend() == MemoryBackend::Mlx)", guards["mlx-only-guard"])
        self.assertIn(").map(|_| phase_peaks)", guards["drop-native-bound"])
        self.assertIn("dit_window.map(|_| 2)", guards["wrong-selected-window"])

    def test_compiler_linker_setup_zero_survivor_and_unrelated_panics_never_get_credit(self):
        for mode in ["compiler", "linker", "setup", "zero", "survived", "wrong-assertion", "no-panic", "wrong-test"]:
            with self.subTest(mode=mode):
                self.output = self.base / ("artifacts-" + mode)
                receipt, fake = self.execute(mode)
                self.assertEqual(receipt["status"], "unit_fake_failed")
                self.assertFalse(receipt["actual_accounting_proof"])
                self.assertEqual(len(fake.seen_mutations), 1)
                self.assertEqual(receipt["mutations"][0]["status"], "invalid_or_survived")
                self.assertEqual(receipt["mutations"][0]["restored_baseline"], "green")
                self.assertEqual((self.root / proof.SOURCE).read_bytes(), REAL_SOURCE)

    def test_discovery_is_dynamic_but_empty_duplicates_and_missing_mechanisms_refuse(self):
        for label, names in [("empty", []), ("missing", [proof.STAGES]), ("duplicate", NAMES + [NAMES[0]])]:
            with self.subTest(label=label):
                self.output = self.base / ("artifacts-" + label)
                receipt, fake = self.execute(names=names)
                self.assertEqual(receipt["status"], "unit_fake_failed")
                self.assertFalse(fake.seen_mutations)
                self.assertFalse(fake.baselines)
                self.assertEqual((self.root / proof.SOURCE).read_bytes(), REAL_SOURCE)

    def test_bad_initial_baseline_never_mutates(self):
        receipt, fake = self.execute("bad-baseline")
        self.assertEqual(receipt["status"], "unit_fake_failed")
        self.assertFalse(fake.seen_mutations)
        self.assertEqual((self.root / proof.SOURCE).read_bytes(), REAL_SOURCE)

    def test_post_restore_baseline_failure_invalidates_entire_receipt(self):
        receipt, fake = self.execute("bad-restored")
        self.assertEqual(receipt["status"], "unit_fake_failed")
        self.assertFalse(receipt["actual_accounting_proof"])
        self.assertEqual(len(fake.seen_mutations), 1)
        self.assertEqual((self.root / proof.SOURCE).read_bytes(), REAL_SOURCE)

    def test_interrupt_restores_exact_source_and_preserves_partial_receipt(self):
        receipt, fake = self.execute("interrupt")
        self.assertEqual(receipt["status"], "unit_fake_interrupted")
        self.assertFalse(receipt["actual_accounting_proof"])
        self.assertEqual((self.root / proof.SOURCE).read_bytes(), REAL_SOURCE)
        self.assertEqual(len(fake.seen_mutations), 1)
        self.assertEqual(fake.baselines, ["baseline"])
        self.assertEqual(json.loads((self.output / "receipt.json").read_text())["status"], "unit_fake_interrupted")
        self.assertEqual((self.output / "original-video_admission.rs").read_bytes(), REAL_SOURCE)

    def test_restore_does_not_clobber_another_writer(self):
        receipt, _ = self.execute("external-edit")
        self.assertEqual(receipt["status"], "unit_fake_restore_conflict")
        self.assertEqual((self.root / proof.SOURCE).read_bytes(), b"// another writer's source\n")
        self.assertEqual((self.output / "original-video_admission.rs").read_bytes(), REAL_SOURCE)
        self.assertFalse(receipt["actual_accounting_proof"])

    def test_unknown_source_anchor_refuses_before_any_command(self):
        original = REAL_SOURCE.replace(b"fn video_floor_weights_bytes(", b"fn changed_native_api(")
        (self.root / proof.SOURCE).write_bytes(original)
        receipt, fake = self.execute()
        self.assertEqual(receipt["status"], "unit_fake_failed")
        self.assertFalse(fake.calls)
        self.assertEqual((self.root / proof.SOURCE).read_bytes(), original)
        self.assertEqual(receipt["source_sha256"], proof.sha(original))

    def test_dirty_checkpoint_refuses_before_cargo(self):
        receipt, fake = self.execute("dirty")
        self.assertEqual(receipt["status"], "unit_fake_failed")
        self.assertFalse(any("test" in argv for argv, _, _ in fake.calls))
        self.assertEqual((self.root / proof.SOURCE).read_bytes(), REAL_SOURCE)

    def test_mixed_or_unknown_pin_refuses_before_commands(self):
        with (self.root / "Cargo.lock").open("a") as stream:
            stream.write('[[package]]\nname="other-inference"\nversion="0.1.0"\n'
                         f'source="git+https://github.com/SceneWorks/inference?rev={HEAD}#{HEAD}"\n')
        receipt, fake = self.execute()
        self.assertEqual(receipt["status"], "unit_fake_failed")
        self.assertFalse(fake.calls)
        self.assertIn("one exact inference commit", receipt["error"])

    def test_output_scope_and_cfg_cannot_be_arbitrary_code_arguments(self):
        fake = FakeCargo(self.root)
        with self.assertRaises(proof.ProofError):
            proof.run_proof(self.root, self.root / "artifacts", runner=fake)
        with self.assertRaises(proof.ProofError):
            proof.run_proof(self.root, self.output, "default; cargo run", runner=fake)
        self.assertFalse(fake.calls)
        self.assertEqual((self.root / proof.SOURCE).read_bytes(), REAL_SOURCE)

    def test_existing_output_is_not_overwritten(self):
        self.output.mkdir()
        sentinel = self.output / "user-data"
        sentinel.write_text("keep")
        with self.assertRaises(FileExistsError):
            proof.run_proof(self.root, self.output, runner=FakeCargo(self.root))
        self.assertEqual(sentinel.read_text(), "keep")

    def test_changed_checkpoint_before_mutation_refuses_without_edit(self):
        receipt, fake = self.execute("head-drift")
        self.assertEqual(receipt["status"], "unit_fake_failed")
        self.assertFalse(fake.seen_mutations)
        self.assertIn("checkpoint/source changed", receipt["error"])
        self.assertEqual((self.root / proof.SOURCE).read_bytes(), REAL_SOURCE)

    def test_other_proof_inputs_cannot_drift_during_restored_baseline(self):
        receipt, fake = self.execute("input-drift")
        self.assertEqual(receipt["status"], "unit_fake_failed")
        self.assertFalse(receipt["actual_accounting_proof"])
        self.assertEqual(len(fake.seen_mutations), 1)
        self.assertIn("identity changed", receipt["error"])
        self.assertEqual((self.root / proof.SOURCE).read_bytes(), REAL_SOURCE)
        self.assertTrue((self.root / proof.TEST_SOURCE).read_bytes().endswith(b"// unit_fake outside input edit\n"))

    def test_falsey_injected_runner_never_falls_back_to_real_processes(self):
        class FalseyFake(FakeCargo):
            def __bool__(self):
                return False
        fake = FalseyFake(self.root)
        with mock.patch.object(proof, "subprocess_runner", side_effect=AssertionError("NO real subprocess permitted")), \
             contextlib.redirect_stdout(io.StringIO()):
            receipt = proof.run_proof(self.root, self.output, runner=fake)
        self.assertEqual(receipt["status"], "unit_fake_passed")
        self.assertFalse(receipt["actual_accounting_proof"])
        self.assertTrue(fake.calls)

    def test_summary_or_status_text_alone_is_not_an_assertion_kill(self):
        plan = proof.mutations(REAL_SOURCE.decode())[0]
        valid = assertion_failure(plan)
        proof.killed(valid, plan)
        for text in [valid.text.replace("1 failed", "0 failed"), valid.text.replace("running 1 test", "running 0 tests"),
                     valid.text.replace(" ... FAILED", " ... ignored"), valid.text.replace(plan.assertion, "other panic")]:
            with self.subTest(text=text[:100]), self.assertRaises(proof.ProofError):
                proof.killed(proof.Result(101, text), plan)


if __name__ == "__main__":
    unittest.main()
