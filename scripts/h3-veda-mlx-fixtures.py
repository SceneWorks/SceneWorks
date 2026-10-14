"""Run the five frozen native MLX device fixtures on the admitted Mac2 lane.

This is a separate Actions-owned device gate. It cannot submit product jobs.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time
import tomllib


NATIVE_HEAD = "ba6ed0db452ecbed780ab8eeeac883cd9758ed3a"
WORKFLOW = "SceneWorks/SceneWorks/.github/workflows/h3-veda-mlx-fixtures.yml@"
WORKFLOW_NAME = "H3 Veda Mac2 native MLX fixtures"
RUNNER = "nax-macos-2"
RUNNER_ID = 5281
PHYSICAL_GPU_ID = "734A72AE-13DA-52A8-B0CF-F97E1D646124"
METAL_DEVICE_NAME = "Apple M5 Max"
FIXTURES = (
    "veda::frozen_tests::frozen_upstream_device_scores_select_exact_masks",
    "veda_fixture::native_generator_learned_sparse_off_and_failure_paths",
    "veda_fixture::selected_kernel_actual_device_smoke",
    "veda_fixture::native_veda_controls_preserve_carriers_and_capture_published_schedules",
    "veda::frozen_tests::full_self_test_lifetime_fits_declared_allocator_bound",
)
PASS = re.compile(r"test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; [0-9]+ filtered out")
ADMISSION_SECONDS = 1800
CHILD_SECONDS = 3600


def guardian_module():
    path = Path(__file__).with_name("h3-veda-guardian.py")
    spec = importlib.util.spec_from_file_location("h3_veda_fixture_guardian", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


g = guardian_module()


def check(condition, reason):
    if not condition:
        raise RuntimeError("H3 MLX fixture refusal: " + reason)


def shell(argv, cwd=None, timeout=30):
    return subprocess.check_output(argv, cwd=cwd, text=True, stderr=subprocess.STDOUT, timeout=timeout).strip()


def file_binding(path, allow_empty=False):
    raw = Path(path)
    info = raw.lstat()
    check(stat.S_ISREG(info.st_mode) and (info.st_size > 0 or allow_empty), "bound file missing/unsafe")
    path = raw.resolve(strict=True)
    return {"path": str(path), "bytes": info.st_size, "sha256": g.file_sha(path)}


def python_binding(root, run_dir):
    """Consume original shared bootstrap bytes, not a version-only PATH probe."""
    e = os.environ
    path = e.get("S9_GUARD_PYTHON", "")
    attestation = e.get("S9_GUARD_PYTHON_ATTESTATION", "")
    check(Path(path).is_absolute() and Path(attestation) == Path(run_dir) / "tools" / "python-3.12-attestation.json", "owned Python paths differ")
    interpreter = file_binding(path)
    original = file_binding(attestation)
    check(re.fullmatch(r"[0-9a-f]{64}", e.get("S9_GUARD_PYTHON_SHA256", "")) and interpreter["sha256"] == e["S9_GUARD_PYTHON_SHA256"], "Python executable byte authority differs")
    check(re.fullmatch(r"[0-9a-f]{64}", e.get("S9_GUARD_PYTHON_ATTESTATION_SHA256", "")) and original["sha256"] == e["S9_GUARD_PYTHON_ATTESTATION_SHA256"], "Python attestation byte authority differs")
    raw = Path(attestation).read_text(encoding="utf-8")
    value = json.loads(raw)
    bootstrap = file_binding(Path(root) / "scripts/h3-veda-owned-python.mjs")
    context = value.get("context", {})
    check(context.get("schemaVersion") == 1 and context.get("runDir") == str(run_dir) and context.get("platform") == "darwin" and context.get("arch") == "arm64" and context.get("sourceSha256") == bootstrap["sha256"], "Python source/host/root context differs")
    check(value["interpreter"].get("implementation") == "cpython" and value["interpreter"].get("version") == ".".join(map(str, sys.version_info[:3])), "Python actual full version differs")
    binding = {**interpreter, "version": list(sys.version_info[:3]), "bootstrapAttestation": {**original, "originalJson": raw}}
    manifest = {"runners": {"mlx": {"guardian": {"pythonPath": path, "pythonBinding": binding}}}, "sourceFreeze": {"files": {"scripts/h3-veda-owned-python.mjs": bootstrap["sha256"]}}}
    g.validate_python_binding(manifest, "mlx")
    return binding


def exact_source(root, head):
    root = Path(root).resolve(strict=True)
    check(re.fullmatch(r"[0-9a-f]{40}", head or ""), "source SHA malformed")
    check(shell(["git", "rev-parse", "HEAD"], root) == head, "source head changed")
    check(not shell(["git", "status", "--porcelain=v1", "--untracked-files=all"], root), "source checkout dirty")
    return {"head": head, "tree": shell(["git", "rev-parse", "HEAD^{tree}"], root)}


def package_identity(lock_path):
    packages = [p for p in tomllib.loads(Path(lock_path).read_text(encoding="utf-8"))["package"] if p["name"] == "pmetal-mlx-sys"]
    check(len(packages) == 1, "pmetal-mlx-sys lock identity ambiguous")
    p = packages[0]
    match = re.fullmatch(r"git\+https://github\.com/michaeltrefry/mlx-rs\?rev=([0-9a-f]{40})#\1", p.get("source", ""))
    check(match is not None, "pmetal-mlx-sys source is not exact pinned fork")
    return f"git+https://github.com/michaeltrefry/mlx-rs?rev={match[1]}#pmetal-mlx-sys@{p['version']}"


def parse_build(messages, target_dir, expected_package):
    binaries, shader_dirs = [], []
    for line in messages:
        try:
            m = json.loads(line)
        except json.JSONDecodeError:
            continue
        if m.get("reason") == "compiler-artifact" and m.get("target", {}).get("kind") == ["lib"] and m.get("target", {}).get("name") == "mlx_gen_minimax_h3" and m.get("executable"):
            binaries.append(m["executable"])
        if m.get("reason") == "build-script-executed" and m.get("package_id") == expected_package:
            shader_dirs.append(m["out_dir"])
    check(len(binaries) == len(shader_dirs) == 1, "one exact test binary and pinned metallib build output required")
    target = Path(target_dir).resolve(strict=True)
    binary = Path(binaries[0]).resolve(strict=True)
    out = Path(shader_dirs[0]).resolve(strict=True)
    check(binary.is_relative_to(target) and out.is_relative_to(target), "build output escaped owned target")
    shader = out / "build" / "lib" / "mlx.metallib"
    return file_binding(binary), file_binding(shader), {"packageId": expected_package, "outDir": str(out)}


class FixturePhysical(g.Physical):
    """The campaign policy with only the live fixture workflow tuple substituted."""

    def actions_owner(self, raw):
        e = os.environ
        check(e.get("GITHUB_ACTIONS") == "true" and e.get("GH_TOKEN") and e.get("GITHUB_REPOSITORY") == "SceneWorks/SceneWorks" and e.get("GITHUB_EVENT_NAME") == "workflow_dispatch", "Actions owner environment incomplete")
        check(e.get("GITHUB_SHA") == self.m["sourceFreeze"]["sceneworks"]["head"] and e.get("GITHUB_WORKFLOW_REF", "").startswith(WORKFLOW), "Actions workflow/source differs")
        check(e.get("GITHUB_JOB") == "mac2" and e.get("RUNNER_NAME") == RUNNER, "Actions job/runner differs")
        run_id, attempt = e.get("GITHUB_RUN_ID", ""), e.get("GITHUB_RUN_ATTEMPT", "")
        check(run_id.isdecimal() and attempt.isdecimal(), "Actions run/attempt missing")
        run = json.loads(g.command(["gh", "api", f"repos/SceneWorks/SceneWorks/actions/runs/{run_id}"], 15))
        jobs = json.loads(g.command(["gh", "api", "--paginate", "--slurp", f"repos/SceneWorks/SceneWorks/actions/runs/{run_id}/jobs?per_page=100"], 15))
        check(run.get("id") == int(run_id) and run.get("run_attempt") == int(attempt) and run.get("event") == "workflow_dispatch" and run.get("head_sha") == e["GITHUB_SHA"] and run.get("status") == "in_progress" and run.get("conclusion") is None and run.get("name") == WORKFLOW_NAME, "live Actions run identity differs")
        matches = [j for page in jobs for j in page.get("jobs", []) if j.get("name") == "mac2" and j.get("runner_name") == RUNNER and j.get("run_id") == int(run_id) and j.get("run_attempt") == int(attempt) and j.get("head_sha") == e["GITHUB_SHA"] and j.get("status") == "in_progress" and j.get("conclusion") is None]
        check(len(matches) == 1 and isinstance(matches[0].get("runner_id"), int) and matches[0]["runner_id"] == RUNNER_ID and isinstance(matches[0].get("id"), int), "live Actions job/runner ambiguous")
        runners = [r for page in raw["orgRunners"] for r in page["runners"] if r.get("id") == RUNNER_ID and r.get("name") == RUNNER]
        check(len(runners) == 1 and runners[0]["status"] == "online" and runners[0]["busy"] is True, "live Actions runner identity differs")
        self.actions_identity = {"runId": int(run_id), "runAttempt": int(attempt), "runnerId": RUNNER_ID, "jobId": matches[0]["id"], "sourceHead": e["GITHUB_SHA"], "jobName": "mac2", "workflowRef": e["GITHUB_WORKFLOW_REF"]}
        return RUNNER_ID


def fixture_outcome(stdout, stderr, rc):
    text = Path(stdout).read_text(encoding="utf-8", errors="replace") + "\n" + Path(stderr).read_text(encoding="utf-8", errors="replace")
    return rc == 0 and len(PASS.findall(text)) == 1 and len(re.findall(r"test result:", text)) == 1


def validate_phase(phase, physical_uuid, metal_name, actions_identity):
    samples = phase.get("driverSamples")
    check(phase.get("admitted") is True and phase.get("physicalGpuId") == physical_uuid and isinstance(samples, list) and len(samples) == 5, "five admitted physical samples/UUID missing")
    queue_at, phase_at = phase.get("queueObservedAt"), phase.get("observedAt")
    check(isinstance(queue_at, int) and isinstance(phase_at, int) and queue_at <= phase_at, "queue observation order changed")
    previous = None
    for sample in samples:
        observed = sample.get("observedAt")
        proof = sample.get("actionsAncestorProof")
        check(sample.get("admitted") is True and sample.get("driverAvailable") is True and sample.get("metalDevice") == metal_name and sample.get("platformUuid") == physical_uuid and sample.get("actors") == [] and isinstance(sample.get("memoryFreePercent"), int) and sample["memoryFreePercent"] >= 10, "physical driver/sample identity changed")
        check(isinstance(observed, int) and queue_at <= observed <= phase_at and (previous is None or observed - previous >= 450), "physical samples not independently timed")
        check(isinstance(proof, dict) and proof.get("schemaVersion") == 1 and all(proof.get(key) == value for key, value in actions_identity.items()) and isinstance(proof.get("worker"), dict) and proof["worker"].get("pid") in [entry.get("pid") for entry in proof.get("ancestry", [])], "per-sample live Actions ancestor proof missing/changed")
        previous = observed
    source = phase.get("sourceReceipt", {})
    check(isinstance(source.get("path"), str) and re.fullmatch(r"[0-9a-f]{64}", source.get("sha256", "")), "physical source receipt binding missing")
    check(source.get("sha256") == g.file_sha(source.get("path")), "physical source receipt bytes changed")


def group_alive(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def child_closure(child, birth, deadline):
    """Reap only the exact child process group; refusal retains lock ownership."""
    now = g.pid_birth(child.pid)
    check(now is None or now == birth, "child PID birth changed")
    if child.poll() is None:
        check(now == birth and os.getpgid(child.pid) == child.pid, "owned child process group changed")
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=max(.1, min(10, deadline - time.monotonic())))
        except subprocess.TimeoutExpired:
            check(g.pid_birth(child.pid) == birth and os.getpgid(child.pid) == child.pid, "child PID/group changed before forced drain")
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=max(.1, min(10, deadline - time.monotonic())))
    check(child.poll() is not None and not group_alive(child.pid), "owned child descendants not verifiably drained")
    return {"pid": child.pid, "birth": birth, "reaped": True, "descendantsDrained": True, "exitCode": child.returncode}


def safe_close(physical, closed):
    check(closed, "child process tree closure absent; physical locks retained")
    check(physical.owned(), "physical lock identity changed before release")
    physical.close()


def watchdog_locks(output, parent_pid, parent_birth, lock_fds, interval=.25):
    """Keep the identical flock descriptions held if Actions kills the controller.

    An orphan with no positive child closure waits for owner reconciliation. It
    never treats a cancellation, timeout, or missing process as lock release.
    """
    output = Path(output)
    check(len(lock_fds) == len(g.MAC_LOCKS) == 2, "watchdog requires both canonical lock descriptors")
    for fd, lock_path in zip(lock_fds, g.MAC_LOCKS):
        check(not lock_path.is_symlink(), "watchdog canonical lock path changed")
        info, path = os.fstat(fd), lock_path.stat()
        check((info.st_dev, info.st_ino) == (path.st_dev, path.st_ino), "watchdog lock descriptor/path changed")
    while True:
        stop = output / "watchdog-stop.json"
        if stop.exists():
            proof = json.loads(stop.read_text(encoding="utf-8"))
            check(proof.get("childTreeClosed") is True and proof.get("physicalLocksClosedByController") is True, "watchdog stop lacks complete closure")
            g.atomic(output / "watchdog-closure.json", {"schemaVersion": 1, "reason": "controller_clean_closure", "parentPid": parent_pid, "parentBirth": parent_birth, "stoppedAt": int(time.time() * 1000), "lockPaths": [str(p) for p in g.MAC_LOCKS]})
            return 0
        if g.pid_birth(parent_pid) != parent_birth:
            active_file = output / "running.json"
            active = json.loads(active_file.read_text(encoding="utf-8")) if active_file.exists() else {}
            if active_file.exists() and "active" not in active and "launchIntent" not in active:
                g.atomic(output / "watchdog-closure.json", {"schemaVersion": 1, "reason": "controller_lost_before_or_after_closed_device_child", "parentPid": parent_pid, "parentBirth": parent_birth, "childTreeClosed": True, "stoppedAt": int(time.time() * 1000), "lockPaths": [str(p) for p in g.MAC_LOCKS]})
                return 0
            child = active.get("active", {})
            pid, birth = child.get("pid"), child.get("birth")
            if isinstance(pid, int) and birth is not None and g.pid_birth(pid) == birth and os.getpgid(pid) == pid:
                os.killpg(pid, signal.SIGTERM)
                deadline = time.monotonic() + 10
                while g.pid_birth(pid) == birth and time.monotonic() < deadline:
                    time.sleep(.1)
                if g.pid_birth(pid) == birth:
                    check(os.getpgid(pid) == pid, "orphan child group changed before forced drain")
                    os.killpg(pid, signal.SIGKILL)
                deadline = time.monotonic() + 10
                while group_alive(pid) and time.monotonic() < deadline:
                    time.sleep(.1)
                if not group_alive(pid):
                    g.atomic(output / "watchdog-closure.json", {"schemaVersion": 1, "reason": "controller_lost_owned_child_drained", "parentPid": parent_pid, "parentBirth": parent_birth, "childPid": pid, "childBirth": birth, "childTreeClosed": True, "stoppedAt": int(time.time() * 1000), "lockPaths": [str(p) for p in g.MAC_LOCKS]})
                    return 0
            # A missing or changed PID is not proof that an arbitrary child
            # tree has ended. Retain the locks rather than signal a foreign PID.
            g.atomic(output / "watchdog-unresolved.json", {"schemaVersion": 1, "parentPid": parent_pid, "parentBirth": parent_birth, "activePid": pid, "activeBirth": birth, "reason": "orphan_identity_or_closure_unproven", "lockPaths": [str(p) for p in g.MAC_LOCKS]})
            while True:
                time.sleep(30)
        time.sleep(interval)


def rust_toolchain(native, output, env):
    listing = subprocess.run(["rustup", "toolchain", "list"], cwd=native, env=env, capture_output=True, text=True, timeout=20, check=True)
    installed = any(line.split()[0] == "1.96.0" or line.split()[0].startswith("1.96.0-") for line in listing.stdout.splitlines() if line.strip())
    provision = {"required": not installed, "installedListBefore": listing.stdout}
    if not installed:
        command = ["rustup", "toolchain", "install", "1.96.0", "--profile", "minimal"]
        stdout, stderr = output / "toolchain-install.stdout.log", output / "toolchain-install.stderr.log"
        with stdout.open("wb") as out, stderr.open("wb") as err:
            child = subprocess.Popen(command, cwd=native, env=env, stdout=out, stderr=err, start_new_session=True)
            birth = g.pid_birth(child.pid)
            try:
                check(birth is not None, "cannot prove owned Rust provision PID birth")
                rc = child.wait(timeout=600)
            finally:
                if birth is not None:
                    closure = child_closure(child, birth, time.monotonic() + 10)
                else:
                    check(child.poll() is not None, "unidentified Rust provision still running")
        provision.update(argv=command, pid=child.pid, birth=birth, rc=rc, closure=closure, stdout=file_binding(stdout, allow_empty=True), stderr=file_binding(stderr, allow_empty=True))
        g.atomic(output / "toolchain-install-receipt.json", provision)
        check(rc == 0, "pinned Rust toolchain provision failed")
    result = {"provision": provision, "tools": {}}
    for tool, expected in [("rustc", "rustc 1.96.0 (ac68faa20 2026-05-25)"), ("cargo", "cargo 1.96.0")]:
        version = subprocess.run(["rustup", "run", "1.96.0", tool, "--version"], cwd=native, env=env, capture_output=True, text=True, timeout=20, check=True).stdout.strip()
        check(version == expected if tool == "rustc" else version.startswith(expected + " ("), "pinned Rust tool version differs")
        actual = subprocess.run(["rustup", "which", "--toolchain", "1.96.0", tool], cwd=native, env=env, capture_output=True, text=True, timeout=20, check=True).stdout.strip()
        result["tools"][tool] = {"version": version, **file_binding(Path(actual).resolve(strict=True))}
    g.atomic(output / "toolchain-receipt.json", result)
    return result


def run(args):
    root = Path(args.sceneworks).resolve(strict=True)
    native = Path(args.inference).resolve(strict=True)
    raw_run_dir = Path(args.run_dir)
    check(raw_run_dir.is_absolute() and raw_run_dir.resolve() == raw_run_dir and raw_run_dir.is_relative_to(g.EPIC) and raw_run_dir != g.EPIC, "owned evidence directory must be an absolute non-symlink child of the epic")
    target = raw_run_dir / "cargo-target"
    output = raw_run_dir / "mlx-five-fixtures"
    check(output != root and not output.is_relative_to(root) and not output.is_relative_to(native), "output must be external to source")
    check(not output.exists(), "existing fixture output requires reconciliation; refusing overwrite")
    output.mkdir(parents=True)
    receipt = {"schemaVersion": 1, "kind": "h3_veda_mlx_five_actual_device", "accepted": False, "fixtures": [], "startedAt": int(time.time() * 1000), "admissionLeaseSeconds": ADMISSION_SECONDS, "ownedChildMaxSeconds": CHILD_SECONDS, "physicalLocks": [str(p) for p in g.MAC_LOCKS], "runDir": str(raw_run_dir)}
    g.atomic(output / "running.json", receipt)
    physical = None
    watchdog = None
    closed = True
    stopping = False

    def on_signal(signum, frame):
        nonlocal stopping
        if stopping:
            return
        stopping = True
        raise InterruptedError(f"controller received signal {signum}")

    old = {s: signal.signal(s, on_signal) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        check(args.inference_sha == NATIVE_HEAD and args.expected_runner == RUNNER, "unapproved native source/runner")
        check(args.physical_gpu_id == PHYSICAL_GPU_ID and args.metal_device_name == METAL_DEVICE_NAME, "unapproved Mac2 physical identity")
        check(callable(getattr(g.Physical, "actions_actor_proof", None)), "guardian lacks exact live Actions actor proof")
        check(os.environ.get("RUNNER_NAME") == RUNNER and os.environ.get("GITHUB_SHA") == args.sceneworks_sha, "actual runner/Actions source differs")
        receipt["sources"] = {"sceneworks": exact_source(root, args.sceneworks_sha), "inference": exact_source(native, NATIVE_HEAD)}
        receipt["sourceFiles"] = {"supervisor": file_binding(__file__), "guardian": file_binding(Path(__file__).with_name("h3-veda-guardian.py")), "workflow": file_binding(root / ".github/workflows/h3-veda-mlx-fixtures.yml"), "inferenceLock": file_binding(native / "Cargo.lock")}
        receipt["python"] = python_binding(root, raw_run_dir)
        receipt["sourceFiles"]["pythonBootstrap"] = file_binding(root / "scripts/h3-veda-owned-python.mjs")
        check(not stopping, "controller interrupted before build")
        target.mkdir(parents=True, exist_ok=True)
        build_out, build_err = output / "build.jsonl", output / "build.stderr.log"
        env = os.environ.copy()
        for key in ("PMETAL_METALLIB_PATH", "PMETAL_MLX_PREBUILT_DIR", "GH_TOKEN", "GITHUB_TOKEN", "INFERENCE_READ_TOKEN"):
            env.pop(key, None)
        env.update(CARGO_TARGET_DIR=str(target), CARGO_BUILD_JOBS="2", RUSTUP_TOOLCHAIN="1.96.0")
        receipt["rustToolchain"] = rust_toolchain(native, output, env)
        command = ["rustup", "run", "1.96.0", "cargo", "test", "--locked", "-p", "mlx-gen-minimax-h3", "--lib", "--no-run", "--message-format=json"]
        with build_out.open("wb") as out, build_err.open("wb") as err:
            build = subprocess.Popen(command, cwd=native, env=env, stdout=out, stderr=err, start_new_session=True)
            birth = g.pid_birth(build.pid)
            try:
                check(birth is not None, "cannot prove owned CPU build PID birth")
                rc = build.wait(timeout=CHILD_SECONDS)
            finally:
                if birth is not None:
                    closure = child_closure(build, birth, time.monotonic() + 10)
                else:
                    # No PID identity means no safe signal target. Preserve a
                    # failed receipt; CPU build cannot become device admission.
                    check(build.poll() is not None, "unidentified build still running")
        receipt["build"] = {"argv": command, "rc": rc, "closure": closure, "stdout": file_binding(build_out), "stderr": file_binding(build_err)}
        g.atomic(output / "running.json", receipt)
        check(rc == 0 and not stopping, "exact CPU build failed/interrupted")
        package = package_identity(native / "Cargo.lock")
        binary, shader, provenance = parse_build(build_out.read_text(encoding="utf-8").splitlines(), target, package)
        receipt["binary"], receipt["metallib"], receipt["metallibProvenance"] = binary, shader, provenance
        receipt["sources"]["inference"] = exact_source(native, NATIVE_HEAD)
        receipt["sources"]["sceneworks"] = exact_source(root, args.sceneworks_sha)
        check(not stopping, "controller interrupted before admission")
        manifest = {"runners": {"mlx": {"physicalGpuId": args.physical_gpu_id}}, "sourceFreeze": {"sceneworks": {"head": args.sceneworks_sha}, "resourceContract": {"mlx": {"physicalGpuId": args.physical_gpu_id, "metalDeviceName": args.metal_device_name, "canonicalLocks": [str(p) for p in g.MAC_LOCKS], "hostRunnerNames": [RUNNER]}}}}
        physical = FixturePhysical(manifest, "mlx", output)
        physical.claim()
        parent_birth = g.pid_birth(os.getpid())
        check(parent_birth is not None, "controller PID birth cannot be attested")
        lock_fds = tuple(handle.fileno() for handle in physical.locks.handles)
        watchdog_env = {key: value for key, value in os.environ.items() if key not in ("GH_TOKEN", "GITHUB_TOKEN", "INFERENCE_READ_TOKEN")}
        with (output / "watchdog.stderr.log").open("wb") as watchdog_err:
            watchdog = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--watchdog", "--output", str(output), "--parent-pid", str(os.getpid()), "--parent-birth", parent_birth, "--lock-fds", ",".join(map(str, lock_fds))], env=watchdog_env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=watchdog_err, start_new_session=True, pass_fds=lock_fds)
        check(g.pid_birth(watchdog.pid) is not None and watchdog.poll() is None, "independent physical lock watchdog failed to start")
        receipt["lockWatchdog"] = {"pid": watchdog.pid, "birth": g.pid_birth(watchdog.pid), "lockPaths": [str(p) for p in g.MAC_LOCKS]}
        lease_start = time.monotonic()
        owned_start = time.monotonic()
        for index, fixture in enumerate(FIXTURES, 1):
            check(not stopping and time.monotonic() - lease_start < ADMISSION_SECONDS and time.monotonic() - owned_start < CHILD_SECONDS, "admission or child deadline expired/interrupted")
            check(physical.owned(), "global physical locks lost")
            check(exact_source(native, NATIVE_HEAD) == receipt["sources"]["inference"] and exact_source(root, args.sceneworks_sha) == receipt["sources"]["sceneworks"], "source identity changed")
            check(python_binding(root, raw_run_dir) == receipt["python"], "Python binding changed during fixture lifetime")
            check(file_binding(binary["path"]) == binary and file_binding(shader["path"]) == shader, "binary/metallib bytes changed")
            for tool in receipt["rustToolchain"]["tools"].values():
                check(file_binding(tool["path"]) == {key: tool[key] for key in ("path", "bytes", "sha256")}, "Rust toolchain bytes changed")
            phase = physical.probe(f"mlx-fixture-{index}", {os.getpid()})
            validate_phase(phase, args.physical_gpu_id, args.metal_device_name, physical.actions_identity)
            receipt["actionsIdentity"] = physical.actions_identity
            check(time.monotonic() - lease_start < ADMISSION_SECONDS, "admission expired after physical probe")
            stdout, stderr = output / f"fixture-{index}.stdout.log", output / f"fixture-{index}.stderr.log"
            argv = [binary["path"], fixture, "--ignored", "--exact", "--nocapture"]
            test_env = env.copy()
            test_env["PMETAL_METALLIB_PATH"] = shader["path"]
            phase_record = {"name": fixture, "argv": argv, "admission": phase, "startedAt": int(time.time() * 1000), "binary": binary, "metallib": shader}
            g.atomic(output / "running.json", {**receipt, "launchIntent": {"fixture": fixture, "index": index, "state": "pre_spawning_device_child"}})
            with stdout.open("wb") as out, stderr.open("wb") as err:
                # Inherited lock descriptors keep the global lock held if this
                # controller is killed while the exact test process is alive.
                child = subprocess.Popen(argv, cwd=native, env=test_env, stdout=out, stderr=err, start_new_session=True, pass_fds=tuple(f.fileno() for f in physical.locks.handles))
                birth = g.pid_birth(child.pid)
                closed = False
                phase_record.update(pid=child.pid, birth=birth)
                g.atomic(output / "running.json", {**receipt, "active": phase_record})
                try:
                    check(birth is not None, "owned device test PID birth unprovable")
                    rc = child.wait(timeout=max(.1, CHILD_SECONDS - (time.monotonic() - owned_start)))
                finally:
                    if birth is not None:
                        closure = child_closure(child, birth, time.monotonic() + 10)
                        closed = True
            phase_record.update(rc=rc, closure=closure, stdout=file_binding(stdout), stderr=file_binding(stderr), endedAt=int(time.time() * 1000))
            phase_record["admissionLeaseElapsedSeconds"] = time.monotonic() - lease_start
            phase_record["ownedChildElapsedSeconds"] = time.monotonic() - owned_start
            check(python_binding(root, raw_run_dir) == receipt["python"], "Python binding changed during fixture lifetime")
            check(file_binding(binary["path"]) == binary and file_binding(shader["path"]) == shader, "binary/metallib bytes changed after child")
            phase_record["accepted"] = fixture_outcome(stdout, stderr, rc)
            receipt["fixtures"].append(phase_record)
            g.atomic(output / f"fixture-{index}-receipt.json", phase_record)
            g.atomic(output / "running.json", receipt)
            check(phase_record["accepted"] and not stopping, f"actual fixture {index} failed/interrupted")
        receipt["accepted"] = len(receipt["fixtures"]) == len(FIXTURES)
        receipt["sourcesAfter"] = {"sceneworks": exact_source(root, args.sceneworks_sha), "inference": exact_source(native, NATIVE_HEAD)}
        receipt["binaryAfter"] = file_binding(binary["path"])
        receipt["metallibAfter"] = file_binding(shader["path"])
        check(receipt["sourcesAfter"] == receipt["sources"] and receipt["binaryAfter"] == binary and receipt["metallibAfter"] == shader, "final source/binary/metallib identity changed")
        receipt["endedAt"] = int(time.time() * 1000)
    except BaseException as exc:
        receipt["accepted"] = False
        receipt["error"] = str(exc)
        receipt["endedAt"] = int(time.time() * 1000)
    finally:
        try:
            if physical is not None:
                safe_close(physical, closed)
                receipt["locksReleasedAfterClosure"] = True
                if watchdog is not None:
                    g.atomic(output / "watchdog-stop.json", {"childTreeClosed": True, "physicalLocksClosedByController": True})
                    check(watchdog.wait(timeout=10) == 0, "watchdog did not verify clean lock closure")
                    receipt["watchdogClosure"] = file_binding(output / "watchdog-closure.json")
        except BaseException as exc:
            receipt["accepted"] = False
            receipt["closureError"] = str(exc)
            receipt["watchdogHoldingLocks"] = watchdog is not None and watchdog.poll() is None
        g.atomic(output / "receipt.json", receipt)
        for signum, handler in old.items():
            signal.signal(signum, handler)
    return 0 if receipt["accepted"] else 1


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--watchdog":
        parser = argparse.ArgumentParser()
        parser.add_argument("--watchdog", action="store_true")
        parser.add_argument("--output", required=True)
        parser.add_argument("--parent-pid", required=True, type=int)
        parser.add_argument("--parent-birth", required=True)
        parser.add_argument("--lock-fds", required=True)
        args = parser.parse_args()
        signal.signal(signal.SIGTERM, lambda *_: None)
        signal.signal(signal.SIGINT, lambda *_: None)
        try:
            result = watchdog_locks(args.output, args.parent_pid, args.parent_birth, [int(x) for x in args.lock_fds.split(",")])
        except BaseException as exc:
            # Exception paths must not release inherited physical locks.
            g.atomic(Path(args.output) / "watchdog-unresolved.json", {"schemaVersion": 1, "reason": "watchdog_exception", "error": str(exc), "parentPid": args.parent_pid, "lockPaths": [str(p) for p in g.MAC_LOCKS]})
            while True:
                time.sleep(30)
        sys.exit(result)
    parser = argparse.ArgumentParser()
    for name in ("sceneworks", "inference", "sceneworks_sha", "inference_sha", "expected_runner", "physical_gpu_id", "metal_device_name", "run_dir"):
        parser.add_argument("--" + name.replace("_", "-"), required=True)
    sys.exit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
