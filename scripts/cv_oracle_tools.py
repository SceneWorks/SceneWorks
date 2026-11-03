#!/usr/bin/env python3
"""Install the manual CPU oracle's pinned tools in a bounded, run-private venv."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import urllib.parse
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cv_provision import ROOT, acquire, admit, digest, load_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--inference-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    admit(args.source_sha, args.inference_sha, "mlx")
    if sys.platform != "darwin" or sys.version_info[:2] != (3, 12):
        raise ValueError("native Mac CPython 3.12 oracle tools required")
    output = args.output.resolve()
    if output == ROOT or ROOT in output.parents:
        raise ValueError("oracle tools must be outside checkout")
    output.mkdir(parents=True, exist_ok=False)
    manifest = load_manifest()
    receipt = {"status": "installing", "sourceSha": args.source_sha, "inferenceSha": args.inference_sha, "downloadBytes": 0, "assets": []}
    budget = manifest["budgets"]["oracleWheelBytes"]
    wheels = output / "wheels"
    try:
        subprocess.run([sys.executable, "-m", "venv", str(output / "venv")], check=True)
        python = output / "venv/bin/python"
        pip_env = {key: value for key, value in os.environ.items() if not key.startswith("PIP_")}
        pip_env.update(PIP_CONFIG_FILE=os.devnull, PIP_CACHE_DIR=str(output / "pip-cache"),
                       PIP_INDEX_URL="https://pypi.org/simple", PIP_EXTRA_INDEX_URL="")
        pip = [str(python), "-m", "pip", "--disable-pip-version-check", "--timeout", "30", "--retries", "1"]
        for package in manifest["oracle"]["packages"]:
            asset = {"name": package["filename"], "url": package["url"], "bytes": package["size"], "sha256": package["sha256"]}
            acquire(asset, wheels / asset["name"], receipt, budget)
        requirements = [f"{package['name']}=={package['version']}" for package in manifest["oracle"]["packages"]]
        # PyPI supplies wheel metadata for resolution; all substantial primary wheels are
        # already pinned locally. Resolve once, then download every selected dependency with
        # explicit length/hash bounds before an entirely offline installation.
        subprocess.run([*pip, "install", "--dry-run", "--ignore-installed", "--only-binary=:all:", "--no-cache-dir",
                        "--index-url", "https://pypi.org/simple", "--find-links", str(wheels),
                        "--report", str(output / "resolution.json"), *requirements], check=True, timeout=600, env=pip_env)
        plan = json.loads((output / "resolution.json").read_text())["install"]
        if not plan or len(plan) > 128:
            raise ValueError("oracle dependency plan exceeds its bound")
        exact = []
        for item in plan:
            name, version = item["metadata"]["name"], item["metadata"]["version"]
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or not re.fullmatch(r"[A-Za-z0-9_.+-]+", version):
                raise ValueError("invalid resolved package identity")
            exact.append(f"{name}=={version}")
            info = item["download_info"]
            url = info["url"]
            parsed = urllib.parse.urlsplit(url)
            filename = Path(urllib.parse.unquote(parsed.path)).name
            expected = info["archive_info"]["hashes"]["sha256"]
            if not filename.endswith(".whl"):
                raise ValueError("oracle tools must use binary wheels")
            if (wheels / filename).is_file():
                if digest(wheels / filename) != expected:
                    raise ValueError("resolved wheel differs from pinned wheel")
                continue
            if parsed.scheme != "https" or parsed.hostname != "files.pythonhosted.org":
                raise ValueError("oracle dependencies must come from public PyPI wheels")
            with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=30) as response:
                size = int(response.headers["Content-Length"])
            acquire({"name": filename, "url": url, "bytes": size, "sha256": expected}, wheels / filename, receipt, budget)
        subprocess.run([*pip, "install", "--no-index", "--no-deps", "--find-links", str(wheels),
                        "--report", str(output / "install.json"), *exact], check=True, timeout=300, env=pip_env)
        subprocess.run([*pip, "check"], check=True, timeout=60, env=pip_env)
        with (output / "freeze.txt").open("w") as stream:
            subprocess.run([*pip, "freeze", "--all"], check=True, stdout=stream, timeout=60, env=pip_env)
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
