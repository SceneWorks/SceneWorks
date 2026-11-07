#!/usr/bin/env python3
"""Install the manual CPU oracle's pinned tools in a bounded, run-private venv."""
import argparse
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import urllib.parse

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cv_provision import MANIFEST, ROOT, acquire, admit, digest, load_manifest


def package_key(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def locked_packages(manifest):
    """The reviewed source is the authority for every wheel, never a live resolver."""
    oracle = manifest["oracle"]
    if oracle["platform"] != {"sysPlatform": "darwin", "machine": "arm64",
                              "implementation": "cpython", "python": "3.12"}:
        raise ValueError("native Mac ARM64 CPython 3.12 wheel lock required")
    packages = oracle["packages"]
    if not packages or len(packages) > 128 or len(packages) != oracle["lockProvenance"]["packageCount"]:
        raise ValueError("oracle wheel lock inventory mismatch")
    names, files = set(), set()
    for package in packages:
        name, version, filename = (package[key] for key in ("name", "version", "filename"))
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or not re.fullmatch(r"[A-Za-z0-9_.+-]+", version):
            raise ValueError("invalid locked package identity")
        key = package_key(name)
        if key in names or filename in files:
            raise ValueError("duplicate locked package or wheel")
        names.add(key)
        files.add(filename)
        if (not re.fullmatch(r"[A-Za-z0-9_.+-]+\.whl", filename)
                or package_key(filename.split("-", 1)[0]) != key
                or filename.split("-")[1] != version):
            raise ValueError("locked wheel filename differs from package identity")
        url = urllib.parse.urlsplit(package["url"])
        if (url.scheme != "https" or url.netloc != "files.pythonhosted.org"
                or url.query or url.fragment or Path(urllib.parse.unquote(url.path)).name != filename):
            raise ValueError("locked wheels require exact public PyPI HTTPS URLs")
        if (type(package["size"]) is not int or package["size"] <= 0
                or not re.fullmatch(r"[a-f0-9]{64}", package["sha256"])):
            raise ValueError("locked wheel size and SHA-256 required")
    # Includes yaml, imported explicitly by the producer as well as by Ultralytics.
    if not {"ultralytics", "torch", "torchvision", "numpy", "safetensors",
            "opencv-python", "pillow", "pyyaml"} <= names:
        raise ValueError("oracle's direct imports must be in the wheel lock")
    total = sum(package["size"] for package in packages)
    if total != oracle["lockProvenance"]["wheelBytes"] or total > manifest["budgets"]["oracleWheelBytes"]:
        raise ValueError("oracle wheel lock exceeds or differs from its byte budget")
    return packages


def verify_install_report(report, packages, wheels):
    expected = {package_key(package["name"]): package for package in packages}
    installed = report["install"]
    if len(installed) != len(expected):
        raise ValueError("installed oracle closure differs from wheel lock")
    seen = set()
    for item in installed:
        key = package_key(item["metadata"]["name"])
        package = expected.get(key)
        if (key in seen or package is None or item["metadata"]["version"] != package["version"]
                or item["download_info"]["archive_info"]["hashes"]["sha256"] != package["sha256"]
                or item["download_info"]["url"] != (wheels / package["filename"]).as_uri()):
            raise ValueError("installed oracle wheel differs from reviewed lock")
        seen.add(key)


def install_tools(manifest, output, receipt):
    packages = locked_packages(manifest)
    wheels = output / "wheels"
    # Fetch and verify the entire closed inventory before invoking any package installer.
    for package in packages:
        asset = {"name": package["filename"], "url": package["url"],
                 "bytes": package["size"], "sha256": package["sha256"]}
        acquire(asset, wheels / asset["name"], receipt, manifest["budgets"]["oracleWheelBytes"], use_cache=False)
    (output / "wheel-lock.json").write_text(json.dumps(manifest["oracle"], indent=2) + "\n")
    requirements = output / "requirements.lock"
    requirements.write_text("".join(
        f"{package['name']} @ {(wheels / package['filename']).as_uri()} --hash=sha256:{package['sha256']}\n"
        for package in packages))
    subprocess.run([sys.executable, "-I", "-B", "-m", "venv", str(output / "venv")], check=True, timeout=60)
    python = output / "venv/bin/python"
    pip_env = {key: value for key, value in os.environ.items() if not key.startswith("PIP_")}
    pip_env.update(PIP_CONFIG_FILE=os.devnull, PIP_CACHE_DIR=str(output / "pip-cache"),
                   PIP_NO_INDEX="1", PYTHONNOUSERSITE="1")
    pip = [str(python), "-I", "-B", "-m", "pip", "--disable-pip-version-check", "--no-cache-dir",
           "--timeout", "30", "--retries", "1"]
    # Direct verified wheel paths, no dependency resolver, no index, no sdist/build hooks.
    subprocess.run([*pip, "install", "--no-index", "--no-deps", "--require-hashes", "--ignore-installed",
                    "--only-binary=:all:", "--report", str(output / "install.json"),
                    "--requirement", str(requirements)], check=True, timeout=300, env=pip_env)
    verify_install_report(json.loads((output / "install.json").read_text()), packages, wheels)
    subprocess.run([*pip, "check"], check=True, timeout=60, env=pip_env)
    with (output / "freeze.txt").open("w") as stream:
        subprocess.run([*pip, "freeze", "--all"], check=True, stdout=stream, timeout=60, env=pip_env)
    return python


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--inference-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    admit(args.source_sha, args.inference_sha, "mlx")
    if (sys.platform != "darwin" or platform.machine() != "arm64"
            or sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 12)):
        raise ValueError("native Mac ARM64 CPython 3.12 oracle tools required")
    output = args.output.resolve()
    if output == ROOT or ROOT in output.parents:
        raise ValueError("oracle tools must be outside checkout")
    output.mkdir(parents=True, exist_ok=False)
    manifest = load_manifest()
    receipt = {"status": "installing", "sourceSha": args.source_sha, "inferenceSha": args.inference_sha,
               "manifestSha256": digest(MANIFEST), "downloadBytes": 0, "assets": []}
    try:
        python = install_tools(manifest, output, receipt)
        with open(os.environ["GITHUB_ENV"], "a", encoding="utf-8") as env:
            env.write(f"CV_ORACLE_PYTHON={python}\n")
        receipt["status"] = "passed"
    except Exception as error:
        receipt.update(status="failed", error=str(error))
        raise
    finally:
        (output / "tools.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
