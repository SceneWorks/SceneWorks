#!/usr/bin/env python3
"""One explicit H3 admission CPU proof, never an ordinary PR/main mutation gate.

Run only in an authorized, isolated ephemeral checkout after pin integration. The hosting
route owns OS-level cache/network containment; this driver uses disposable data/HF/temp paths,
offline Cargo, two build jobs and fixed worker/lib tests. No models or API servers are run.
Injected runners are for Python unit tests ONLY and produce unmistakably unit_fake receipts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path("crates/sceneworks-worker/src/video_admission.rs")
TEST_SOURCE = Path("crates/sceneworks-worker/src/video_admission/tests.rs")
SCRIPT = Path("scripts/ci/h3_admission_mutation_proof.py")
PREFIX = "video_admission::tests::native_quant_"
STAGES = PREFIX + "separate_stages_fit_resident_and_staged_on_both_backends"
GEOMETRY = PREFIX + "geometry_precision_and_schedule_reach_the_production_selector"
WINDOW = PREFIX + "windowed_dit_and_conditioner_reach_rung_four_without_full_component_floor"
TASKS = PREFIX + "unselected_task_changes_do_not_move_the_selected_request_budget"
SCOPES = PREFIX + "declared_window_scopes_reach_selector_and_reject_unsupported_choices"
REQUIRED = frozenset({STAGES, GEOMETRY, WINDOW, TASKS, SCOPES})
SUMMARY = re.compile(r"test result: (ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored; (\d+) measured; (\d+) filtered out;")
STATUSES = re.compile(r"^test (" + re.escape(PREFIX) + r"\w+) \.\.\. (ok|FAILED|ignored)\s*$", re.M)
RUNNING = re.compile(r"^running (\d+) tests?\s*$", re.M)
COMPILE_ERRORS = re.compile(r"error\[E\d+\]|could not compile|linking with .* failed|linker command failed|failed to run custom build command|No such file or directory", re.I)


class ProofError(RuntimeError):
    pass


@dataclass(frozen=True)
class Result:
    code: int
    text: str


@dataclass(frozen=True)
class Mutation:
    name: str
    test: str
    assertion: str
    source: str


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1 or old == new:
        raise ProofError(f"source anchor must match exactly once: {old[:100]!r}")
    return text.replace(old, new, 1)


def in_function(text: str, start: str, end: str, old: str, new: str) -> str:
    if text.count(start) != 1 or text.count(end) != 1:
        raise ProofError(f"unknown source function boundary: {start}")
    first, last = text.index(start), text.index(end, text.index(start))
    return text[:first] + once(text[first:last], old, new) + text[last:]


def mutations(source: str) -> list[Mutation]:
    guard = ".and_then(|phase| phase.native_quant.as_ref())\n    {"
    scope = ("fn video_floor_weights_bytes(", "\n/// The historical activation floor,")
    stage_message = "separate native stages fit; their legacy sum does not"
    result = []
    for name, condition in [
        ("btr-only-guard", "contract.engages_selection(&selection, MemoryStrategy::BoundedTransformerResidency)"),
        ("mlx-only-guard", "selector.backend() == MemoryBackend::Mlx"),
    ]:
        replacement = guard.replace("\n    {", f"\n        .filter(|_| {condition})\n    {{")
        changed = in_function(source, *scope, guard, replacement)
        result.append(Mutation(name, STAGES, stage_message, changed))
    old = """            let phase_peaks = match native_quant_phase_floor(
                self.contract,
                &selection,
                self.native_quant_request,
                geometry,
                phase_peaks,
            ) {"""
    result.append(Mutation("drop-native-bound", STAGES,
        "actual cold constructor exceeds the ordinary stage+activation maximum and must remain priced",
        once(source, old, old[:-2] + ".map(|_| phase_peaks) {")))
    result.append(Mutation("wrong-selected-window", WINDOW,
        "ordinary raw floor consumes correlated native and TE residency without scratch",
        in_function(source, *scope, "native.resident_peak_bytes_windowed(rows, dit_window)?",
                    "native.resident_peak_bytes_windowed(rows, dit_window.map(|_| 2))?")))
    old = """                selection,
                predicted_peak_bytes,
                calibration_fingerprint,
            );
            synthesized.push(("""
    new = old.replace("                predicted_peak_bytes,", "                crate::memory_strategy::floor_admitted_peak_bytes(backend, predicted_peak_bytes, unmodeled_activation_bytes),")
    result.append(Mutation("double-policy-margin", GEOMETRY,
        "admitted geometry must replace the borrowed placeholder shape", once(source, old, new)))
    old = "    let shape = request.shape_at(geometry);"
    new = """    let mut native_profile = native.clone();
    for task in &mut native_profile.tasks {
        for projection in &mut task.projections {
            projection.activation_width = 2;
        }
    }
    let native = &native_profile;
    let shape = request.shape_at(geometry);"""
    result.append(Mutation("wrong-source-precision", GEOMETRY,
        "selected native source precision is not coerced to request tier",
        in_function(source, "fn native_quant_phase_floor(", "\n/// The weights+headroom floor", old, new)))
    return result


def listed_tests(result: Result) -> set[str]:
    if result.code != 0 or COMPILE_ERRORS.search(result.text):
        raise ProofError("test discovery did not compile/run successfully")
    names = re.findall(r"^(" + re.escape(PREFIX) + r"\w+): test\s*$", result.text, re.M)
    if not names or len(names) != len(set(names)) or not REQUIRED.issubset(names):
        raise ProofError("compiled discovery is empty, duplicated, or missing required H3 mechanisms")
    return set(names)


def green(result: Result, names: set[str]) -> None:
    summaries, running = SUMMARY.findall(result.text), RUNNING.findall(result.text)
    statuses = STATUSES.findall(result.text)
    if (result.code != 0 or COMPILE_ERRORS.search(result.text) or len(summaries) != 1
        or running != [str(len(names))] or len(statuses) != len(names)
        or dict(statuses) != {name: "ok" for name in names}
        or summaries[0][:5] != ("ok", str(len(names)), "0", "0", "0")):
        raise ProofError("full discovered baseline did not execute and pass every H3 test")


def killed(result: Result, mutation: Mutation) -> None:
    summaries = SUMMARY.findall(result.text)
    panic = re.search(r"thread ['\"]" + re.escape(mutation.test) + r"['\"][^\n]*panicked at", result.text)
    if (result.code == 0 or COMPILE_ERRORS.search(result.text)
        or RUNNING.findall(result.text) != ["1"]
        or STATUSES.findall(result.text) != [(mutation.test, "FAILED")]
        or len(summaries) != 1 or summaries[0][:5] != ("FAILED", "0", "1", "0", "0")
        or not panic or mutation.assertion not in result.text):
        raise ProofError(f"{mutation.name}: no compiled, executed named-test assertion failure (compile/setup/zero-test red is INVALID)")


def atomic_write(path: Path, data: bytes) -> None:
    mode = path.stat().st_mode if path.exists() else 0o644
    fd, temporary = tempfile.mkstemp(prefix=".h3-proof-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode & 0o777)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def subprocess_runner(command: list[str], cwd: Path, env: dict[str, str], log: Path) -> Result:
    """Fixed argv only; terminate only the process group this invocation creates."""
    output = []
    with log.open("w") as stream, subprocess.Popen(command, cwd=cwd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            start_new_session=(os.name == "posix")) as child:
        try:
            for line in child.stdout:
                stream.write(line)
                stream.flush()
                print(line, end="", flush=True)
                output.append(line)
            return Result(child.wait(), "".join(output))
        except BaseException:
            if child.poll() is None:
                if os.name == "posix":
                    os.killpg(child.pid, signal.SIGTERM)
                else:
                    child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    if os.name == "posix":
                        os.killpg(child.pid, signal.SIGKILL)
                    else:
                        child.kill()
                    child.wait()
            raise


def run_proof(root: Path, output: Path, cfg: str = "default", *, runner=None) -> dict:
    """An injected runner ALWAYS labels the entire receipt unit_fake, including failures."""
    root, output = root.resolve(), output.resolve()
    if cfg not in {"default", "candle"} or output == root or root in output.parents:
        raise ProofError("fixed cfg required; output must be a new directory outside the checkout")
    output.mkdir(parents=True, exist_ok=False)
    fake = runner is not None
    invoke = subprocess_runner if runner is None else runner
    receipt = {"schema": 1, "proof_kind": "unit_fake" if fake else "actual_cargo",
               "actual_accounting_proof": False, "cfg": cfg, "platform": sys.platform,
               "status": "running", "commands": [], "mutations": []}
    source_path = root / SOURCE
    original = None
    active_hash = None
    old_handlers = {}
    env = os.environ.copy()
    cargo = Path.home() / ".cargo/bin/cargo"
    rustc = Path.home() / ".cargo/bin/rustc"
    env.update({"PATH": str(cargo.parent) + os.pathsep + env.get("PATH", ""),
                "CARGO_BUILD_JOBS": "2", "CARGO_NET_OFFLINE": "true", "CARGO_TERM_COLOR": "never",
                "CARGO_PROFILE_DEV_DEBUG": "0", "CARGO_PROFILE_TEST_DEBUG": "0",
                "CARGO_PROFILE_DEV_STRIP": "false", "CARGO_PROFILE_TEST_STRIP": "false",
                "CARGO_PROFILE_DEV_BUILD_OVERRIDE_STRIP": "none", "CARGO_PROFILE_TEST_BUILD_OVERRIDE_STRIP": "none"})
    for key, directory in [("HF_HOME", "hf"), ("SCENEWORKS_DATA_DIR", "data"), ("TMPDIR", "tmp")]:
        (output / directory).mkdir()
        env[key] = str(output / directory)
    profile = output / "cargo-profile.toml"
    profile.write_text('[profile.dev.package."*"]\nstrip="none"\n[profile.test.package."*"]\nstrip="none"\n')
    receipt["build_environment"] = {key: env.get(key) for key in (
        "CARGO_BUILD_JOBS", "CARGO_NET_OFFLINE", "CARGO_BUILD_TARGET", "CARGO_TARGET_DIR",
        "RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "RUSTUP_TOOLCHAIN", "CARGO_PROFILE_DEV_DEBUG",
        "CARGO_PROFILE_TEST_DEBUG", "CARGO_PROFILE_DEV_STRIP", "CARGO_PROFILE_TEST_STRIP",
        "CARGO_PROFILE_DEV_BUILD_OVERRIDE_STRIP", "CARGO_PROFILE_TEST_BUILD_OVERRIDE_STRIP")}

    def persist():
        atomic_write(output / "receipt.json", (json.dumps(receipt, indent=2) + "\n").encode())

    def event(kind, **fields):
        record = {"event": kind, "time": time.time(), **fields}
        with (output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        persist()

    def command(label, argv):
        entry = {"label": label, "argv": argv, "cwd": str(root), "log": label + ".log"}
        receipt["commands"].append(entry)
        started = time.monotonic()
        event("command_start", **entry)
        try:
            result = invoke(argv, root, env, output / entry["log"])
            if fake:
                (output / entry["log"]).write_text("UNIT_FAKE — NOT RUST ACCOUNTING PROOF\n" + result.text)
            entry["exit_code"] = result.code
            return result
        finally:
            entry["duration_seconds"] = time.monotonic() - started
            event("command_end", **entry)

    def restore():
        nonlocal active_hash
        if original is None:
            return
        current = sha(source_path.read_bytes())
        if current != sha(original):
            if current != active_hash:
                raise ProofError("restore conflict: source changed outside this driver; original preserved in artifact")
            atomic_write(source_path, original)
        if sha(source_path.read_bytes()) != sha(original):
            raise ProofError("exact source restoration failed")
        active_hash = None
        receipt["restored_source_sha256"] = sha(original)
        event("source_restored", sha256=sha(original))

    def interrupted(signum, _frame):
        raise InterruptedError(f"signal {signum}")

    def cargo_command(name, *, listing=False, exact=False):
        argv = ["/usr/bin/nice", "-n", "10", str(cargo), "--config", str(profile),
                "test", "--locked", "-p", "sceneworks-worker", "--lib"]
        if cfg == "candle":
            argv += ["--features", "backend-candle"]
        argv += [name, "--"]
        return argv + (["--list"] if listing else (["--exact"] if exact else []) +
                       ["--format", "pretty", "--test-threads=1" if exact else "--test-threads=2"])

    try:
        if not fake and (os.name != "posix" or not cargo.is_file() or not rustc.is_file()):
            raise ProofError("authorized POSIX CPU checkout with rustup Cargo/rustc required")
        for sig in (signal.SIGINT, signal.SIGTERM):
            old_handlers[sig] = signal.signal(sig, interrupted)
        if source_path.is_symlink() or not source_path.is_file():
            raise ProofError("fixed production source must be a regular nonsymlink file")
        original = source_path.read_bytes()
        receipt["source_sha256"] = sha(original)
        (output / "original-video_admission.rs").write_bytes(original)
        plans = mutations(original.decode())  # Validate ALL unique anchors before any Cargo/mutation.
        hashes = {str(path): sha((root / path).read_bytes()) for path in (TEST_SOURCE, Path("Cargo.lock"), SCRIPT)}
        receipt.update({"source_sha256": sha(original), "input_sha256": hashes})
        tests_text = (root / TEST_SOURCE).read_text()
        for plan in plans:
            if plan.assertion not in tests_text or "fn " + plan.test.split("::")[-1] + "(" not in tests_text:
                raise ProofError("required source assertion/test changed: " + plan.name)
        lock = tomllib.loads((root / "Cargo.lock").read_text())
        pins = {p["source"].rsplit("#", 1)[-1] for p in lock.get("package", [])
                if p.get("source", "").startswith("git+https://github.com/SceneWorks/inference?")}
        if len(pins) != 1 or not re.fullmatch(r"[a-f0-9]{40}", next(iter(pins))):
            raise ProofError("Cargo.lock must resolve one exact inference commit")
        receipt["inference_pin"] = next(iter(pins))
        head = command("head", ["git", "rev-parse", "HEAD"])
        if head.code or not re.fullmatch(r"[a-f0-9]{40}", head.text.strip()):
            raise ProofError("cannot identify source checkpoint")
        receipt["source_commit"] = head.text.strip()
        clean = command("clean", ["git", "diff", "--exit-code", "HEAD", "--", str(SOURCE), str(TEST_SOURCE), "Cargo.lock", str(SCRIPT)])
        if clean.code:
            raise ProofError("proof inputs differ from committed checkpoint")
        for label, executable, flag in [("cargo-version", cargo, "--version"), ("rustc-version", rustc, "-vV")]:
            version = command(label, [str(executable), flag])
            if version.code:
                raise ProofError("cannot identify toolchain")
            receipt[label] = version.text
        names = listed_tests(command("list", cargo_command(PREFIX, listing=True)))
        receipt["discovered_tests"] = sorted(names)
        green(command("baseline", cargo_command(PREFIX)), names)
        for index, plan in enumerate(plans, 1):
            checkpoint = command(f"{index}-checkpoint", ["git", "rev-parse", "HEAD"])
            if checkpoint.code or checkpoint.text.strip() != receipt["source_commit"] or source_path.read_bytes() != original:
                raise ProofError("checkpoint/source changed before mutation")
            if any(sha((root / path).read_bytes()) != expected for path, expected in hashes.items()):
                raise ProofError("proof input changed during execution")
            record = {"name": plan.name, "test": plan.test, "status": "running"}
            receipt["mutations"].append(record)
            changed = plan.source.encode()
            active_hash = sha(changed)
            record["mutated_source_sha256"] = active_hash
            import difflib
            (output / (plan.name + ".patch")).write_text("".join(difflib.unified_diff(
                original.decode().splitlines(True), plan.source.splitlines(True), fromfile=str(SOURCE), tofile=str(SOURCE))))
            failure = None
            try:
                atomic_write(source_path, changed)
                event("mutation_applied", **record)
                killed(command(plan.name, cargo_command(plan.test, exact=True)), plan)
                record["status"] = "killed_by_assertion"
            except ProofError as error:
                record["status"] = "invalid_or_survived"
                failure = error
            finally:
                restore()
            green(command(plan.name + "-restored", cargo_command(PREFIX)), names)
            if source_path.read_bytes() != original or any(
                sha((root / path).read_bytes()) != expected for path, expected in hashes.items()
            ):
                raise ProofError("source/test/pin/driver identity changed during restored baseline")
            record["restored_baseline"] = "green"
            event("mutation_complete", **record)
            if failure:
                raise failure
        receipt["status"] = "unit_fake_passed" if fake else "passed"
        receipt["actual_accounting_proof"] = not fake
    except BaseException as error:
        stopped = isinstance(error, (KeyboardInterrupt, InterruptedError))
        receipt["status"] = ("unit_fake_" if fake else "") + ("interrupted" if stopped else "failed")
        receipt["error"] = f"{type(error).__name__}: {error}"
        if receipt["mutations"] and receipt["mutations"][-1]["status"] == "running":
            receipt["mutations"][-1]["status"] = "interrupted" if stopped else "aborted"
    finally:
        try:
            restore()
        except BaseException as error:
            receipt["status"] = ("unit_fake_" if fake else "") + "restore_conflict"
            receipt["actual_accounting_proof"] = False
            receipt["restore_error"] = str(error)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        persist()
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path, help="new artifact directory outside checkout")
    parser.add_argument("--cfg", choices=("default", "candle"), default="default")
    args = parser.parse_args()
    try:
        receipt = run_proof(ROOT, args.output_dir, args.cfg)
    except (ProofError, OSError) as error:
        print(f"setup refused: {error}", file=sys.stderr)
        return 2
    return 0 if receipt["status"] == "passed" and receipt["actual_accounting_proof"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
