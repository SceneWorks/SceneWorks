#!/usr/bin/env python3
"""Inspect an extracted AppImage and optionally compare its CUDA driver with the host.

Run on Linux after `SceneWorks_*.AppImage --appimage-extract`. No model weights,
package installation, desktop session, or changes to the AppDir are required.
Exit 1 means the AppDir contains host NVIDIA driver libraries; 2 means a probe
failed or was inconclusive. GPU probes are opt-in and run the packaged sidecar's
one-shot SCENEWORKS_GPU_CHECK, never the server or a generation job.
Only use --probe with a release known to implement the CUDA check (0.8.1's mode
only checked Metal and returned success without exercising CUDA).
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile


def run(argv, *, env=None, timeout=30):
    try:
        result = subprocess.run(
            [str(arg) for arg in argv], env=env, capture_output=True,
            text=True, errors="replace", timeout=timeout, check=False,
        )
        return {"exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"exit_code": None, "error": str(error)}


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def loader_paths(appdir):
    # Same directory order as linuxdeploy's AppRun, with existing paths only.
    return [
        str(appdir / relative)
        for relative in (
            "usr/lib", "usr/lib/i386-linux-gnu", "usr/lib/x86_64-linux-gnu",
            "usr/lib32", "usr/lib64", "lib", "lib/i386-linux-gnu",
            "lib/x86_64-linux-gnu", "lib32", "lib64",
        )
        if (appdir / relative).is_dir()
    ]


def probe(api, paths, timeout, scratch, preload=None):
    env = os.environ.copy()
    inherited = env.get("LD_LIBRARY_PATH")
    env["LD_LIBRARY_PATH"] = ":".join(paths + ([inherited] if inherited else []))
    env["SCENEWORKS_GPU_CHECK"] = "1"
    for kind in ("DATA", "CONFIG", "CACHE", "STATE"):
        directory = scratch / kind.lower()
        directory.mkdir(parents=True, exist_ok=True)
        env[f"XDG_{kind}_HOME"] = str(directory)
    if preload:
        env["LD_PRELOAD"] = str(preload)
    return {
        "ldd": run(["ldd", api], env=env, timeout=timeout),
        "gpu_check": run([api], env=env, timeout=timeout),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("appdir", type=Path, help="extracted squashfs-root directory")
    parser.add_argument("--artifact", type=Path, help="also record the original AppImage SHA-256")
    parser.add_argument("--probe", action="store_true", help="run the actual packaged GPU check")
    parser.add_argument("--host-driver-dir", type=Path,
                        help="compare with this directory's host libcuda.so.1 first (e.g. /usr/lib/wsl/lib)")
    parser.add_argument("--inject-803", action="store_true",
                        help="with --probe and --host-driver-dir, compile a cuInit fault injector using cc")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--output", type=Path, help="write the JSON evidence here too")
    args = parser.parse_args()
    appdir = args.appdir.resolve()
    api = appdir / "usr/bin/sceneworks-api"
    if not api.is_file():
        parser.error(f"packaged sidecar not found: {api}")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.host_driver_dir and not args.probe:
        parser.error("--host-driver-dir requires --probe")
    if args.inject_803 and not (args.probe and args.host_driver_dir):
        parser.error("--inject-803 requires --probe and --host-driver-dir")
    if args.probe and sys.platform != "linux":
        parser.error("--probe requires Linux")
    if args.host_driver_dir:
        args.host_driver_dir = args.host_driver_dir.resolve()
        if not (args.host_driver_dir / "libcuda.so.1").is_file():
            parser.error("--host-driver-dir must contain the host's libcuda.so.1")
    forbidden = []
    for path in sorted(appdir.rglob("*")):
        if path.name.startswith(("libcuda.so", "libnvidia-ml.so")):
            forbidden.append({
                "path": path.relative_to(appdir).as_posix(),
                "sha256": file_sha256(path) if path.is_file() else None,
            })
    report = {
        "platform": platform.platform(),
        "appdir": str(appdir),
        "api_sha256": file_sha256(api),
        "bundled_host_driver_libraries": forbidden,
        "packaging_passed": not forbidden,
        "scope": "AppImage packaging and one-shot CUDA probe; not a Bazzite desktop or model-generation test",
    }
    if args.artifact:
        report["artifact"] = {"name": args.artifact.name, "sha256": file_sha256(args.artifact)}
    if args.probe:
        report["nvidia_smi"] = run([
            "nvidia-smi", "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader,nounits",
        ], timeout=args.timeout)
        paths = loader_paths(appdir)
        with tempfile.TemporaryDirectory(prefix="sceneworks-appimage-probe-") as directory:
            scratch = Path(directory)
            report["bundled_driver"] = probe(api, paths, args.timeout, scratch / "bundled")
            if args.host_driver_dir:
                host_paths = [str(args.host_driver_dir)] + paths
                report["host_driver_first"] = probe(api, host_paths, args.timeout, scratch / "host")
                if args.inject_803:
                    source = scratch / "cuda803.c"
                    source.write_text("int cuInit(unsigned int flags) { (void)flags; return 803; }\n")
                    library = scratch / "cuda803.so"
                    compiled = run(["cc", "-shared", "-fPIC", source, "-o", library], timeout=args.timeout)
                    report["fault_injector_build"] = compiled
                    if compiled["exit_code"] == 0:
                        report["injected_803"] = probe(
                            api, host_paths, args.timeout, scratch / "injected", preload=library,
                        )
                        check = report["injected_803"]["gpu_check"]
                        text = check.get("stdout", "") + check.get("stderr", "")
                        report["injected_803_detected"] = (
                            check["exit_code"] not in (None, 0)
                            and "CUDA_ERROR_SYSTEM_DRIVER_MISMATCH" in text
                            and "reboot" in text.lower()
                        )
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if forbidden:
        return 1
    for key in ("bundled_driver", "host_driver_first"):
        if key in report and report[key]["gpu_check"]["exit_code"] != 0:
            return 2
    if args.inject_803 and not report.get("injected_803_detected"):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
