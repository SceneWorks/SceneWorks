#!/usr/bin/env python3
"""Bounded, digest-checked fixture provisioning for manual CV proofs; stdlib only."""
import argparse
import hashlib
import json
import mmap
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "scripts/cv-fixture-manifest.json"
ORACLES = ("refs.safetensors", "refs_ext.safetensors", "ref_people.json")


def digest(file):
    with Path(file).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_manifest():
    return json.loads(MANIFEST.read_text())


def admit(source, inference, backend, *, env=None):
    env = os.environ if env is None else env
    if env.get("GITHUB_ACTIONS") != "true" or env.get("GITHUB_EVENT_NAME") != "workflow_dispatch":
        raise ValueError("CV asset provisioning and oracle execution require an explicit Actions dispatch")
    if not all(re.fullmatch(r"[a-f0-9]{40}", sha) for sha in (source, inference)):
        raise ValueError("exact source and inference SHAs required")
    if env.get("GITHUB_SHA") != source:
        raise ValueError("dispatch SHA must equal selected source")
    if backend not in ("mlx", "cuda"):
        raise ValueError("unknown backend")
    if backend == "mlx" and env.get("CV_MAC_GO") != "true":
        raise ValueError("explicit Mac execution authorization required")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() != source:
        raise ValueError("checkout must match selected source")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("clean source required")
    pins = []
    for file in ("Cargo.toml", "crates/sceneworks-worker/Cargo.toml", "crates/sceneworks-memory-adapter/Cargo.toml"):
        pins += re.findall(r'git\s*=\s*"https://github.com/SceneWorks/inference"\s*,\s*rev\s*=\s*"([a-f0-9]+)"', (ROOT / file).read_text())
    locked = re.findall(r'git\+https://github.com/SceneWorks/inference\?rev=([a-f0-9]+)#([a-f0-9]+)', (ROOT / "Cargo.lock").read_text())
    if not pins or set(pins) != {inference} or not locked or any(set(pair) != {inference} for pair in locked):
        raise ValueError("all consumer manifests and resolved lock pins must match")


def verify_asset(file, asset):
    if not file.is_file() or file.stat().st_size != asset["bytes"] or digest(file) != asset["sha256"]:
        raise ValueError(f"fixture identity mismatch: {asset['name']}")


def candidates(asset, extra=None):
    if extra:
        yield Path(extra) / asset["name"]
    if "repo" not in asset:
        return
    roots = []
    for key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if os.environ.get(key):
            roots.append(Path(os.environ[key]))
    if os.environ.get("HF_HOME"):
        roots.append(Path(os.environ["HF_HOME"]) / "hub")
    roots.append(Path.home() / ".cache/huggingface/hub")
    if sys.platform == "win32":
        roots.append(Path("E:/huggingface/hub"))
    for root in dict.fromkeys(roots):
        yield root / ("models--" + asset["repo"].replace("/", "--")) / "snapshots" / asset["revision"] / asset["name"]


def acquire(asset, destination, receipt, budget, *, extra=None, opener=urllib.request.urlopen):
    """Never mutate cache candidates; stage a regular file and verify bytes before publication."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ValueError("fresh staging destination required")
    partial = destination.with_name(destination.name + ".partial")
    origin = "download"
    cached = next((file for file in candidates(asset, extra) if file.is_file()), None)
    try:
        if cached:
            verify_asset(cached, asset)
            with cached.open("rb") as source, partial.open("xb") as out:
                shutil.copyfileobj(source, out, 1024 * 1024)
            origin = str(cached)
        else:
            if receipt["downloadBytes"] + asset["bytes"] > budget:
                raise ValueError("manual fixture download budget exceeded")
            if not asset["url"].startswith("https://"):
                raise ValueError("HTTPS asset URL required")
            with opener(asset["url"], timeout=60) as response, partial.open("xb") as out:
                header = response.headers.get("Content-Length")
                if header is not None and int(header) != asset["bytes"]:
                    raise ValueError("download length differs from pinned manifest")
                count = 0
                while chunk := response.read(min(1024 * 1024, asset["bytes"] + 1 - count)):
                    count += len(chunk)
                    receipt["downloadBytes"] += len(chunk)
                    if count > asset["bytes"]:
                        raise ValueError("download exceeds pinned size")
                    out.write(chunk)
        verify_asset(partial, asset)
        partial.rename(destination)
        receipt["assets"].append({"file": asset["name"], "bytes": asset["bytes"], "sha256": asset["sha256"], "origin": origin, "url": asset["url"]})
    finally:
        partial.unlink(missing_ok=True)


def varint(buf, offset, end):
    value = shift = 0
    while offset < end:
        byte = buf[offset]
        offset += 1
        value |= (byte & 127) << shift
        if not byte & 128:
            return value, offset
        shift += 7
        if shift >= 70:
            raise ValueError("invalid protobuf varint")
    raise ValueError("truncated protobuf varint")


def fields(buf, start, end):
    offset = start
    while offset < end:
        key, offset = varint(buf, offset, end)
        field, wire = key >> 3, key & 7
        if wire == 0:
            value, offset = varint(buf, offset, end)
            yield field, wire, value, value
        elif wire in (1, 5):
            size = 8 if wire == 1 else 4
            yield field, wire, offset, offset + size
            offset += size
        elif wire == 2:
            size, offset = varint(buf, offset, end)
            if offset + size > end:
                raise ValueError("truncated protobuf message")
            yield field, wire, offset, offset + size
            offset += size
        else:
            raise ValueError("unsupported protobuf wire type")
    if offset != end:
        raise ValueError("invalid protobuf boundary")


def inspect_onnx(file):
    """Metadata-only ONNX schema walk; these pinned self-contained exports need no sidecars."""
    children = {"model": {7: "graph", 20: "training", 25: "function"},
                "graph": {1: "node", 5: "tensor", 15: "sparse"}, "node": {5: "attribute"},
                "attribute": {5: "tensor", 6: "graph", 10: "tensor", 11: "graph", 22: "sparse", 23: "sparse"},
                "sparse": {1: "tensor", 2: "tensor"}, "training": {1: "graph", 2: "graph"}, "function": {7: "node"}}
    tensors = 0
    with file.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as buf:
        def walk(kind, start, end):
            nonlocal tensors
            if kind == "tensor":
                tensors += 1
            for num, wire, begin, finish in fields(buf, start, end):
                if kind == "tensor" and (num == 13 or (num == 14 and wire == 0 and begin == 1)):
                    raise ValueError("pinned ONNX unexpectedly requires external tensor data")
                if child := children.get(kind, {}).get(num):
                    if wire != 2:
                        raise ValueError("invalid ONNX metadata field")
                    walk(child, begin, finish)
        walk("model", 0, len(buf))
    if tensors == 0:
        raise ValueError("ONNX graph contains no inspected tensors")
    return {"file": file.name, "tensors": tensors, "externalData": False}


def extract_runtime(wheel, destination, kind, backend, limit):
    expected = {"onnxruntime.dll", "onnxruntime_providers_cuda.dll", "onnxruntime_providers_shared.dll"}
    outputs = []
    with zipfile.ZipFile(wheel) as archive:
        entries = [entry for entry in archive.infolist() if not entry.is_dir() and
                   ((backend == "mlx" and Path(entry.filename).name.startswith("libonnxruntime") and entry.filename.endswith(".dylib")) or
                    (backend == "cuda" and entry.filename.endswith(".dll") and (kind == "cuda" or Path(entry.filename).name in expected)))]
        if not entries or sum(entry.file_size for entry in entries) > limit:
            raise ValueError("missing runtime libraries or extraction budget exceeded")
        names = [Path(entry.filename).name for entry in entries]
        if len(names) != len(set(names)):
            raise ValueError("duplicate runtime library basename")
        if backend == "cuda" and kind == "ort" and set(names) != expected:
            raise ValueError("ORT wheel lacks required CUDA provider libraries")
        if backend == "mlx" and len(entries) != 1:
            raise ValueError("expected one CoreML runtime dylib")
        destination.mkdir(parents=True, exist_ok=True)
        for entry in entries:
            name = "libonnxruntime.dylib" if backend == "mlx" else Path(entry.filename).name
            file = destination / name
            with archive.open(entry) as source, file.open("xb") as out:
                shutil.copyfileobj(source, out, 1024 * 1024)
            if backend == "mlx":
                file.chmod(0o755)
            outputs.append({"file": str(file), "bytes": file.stat().st_size, "sha256": digest(file)})
    return outputs


def validate_oracles(directory, source, inference, manifest):
    receipt = json.loads((directory / "oracle.json").read_text())
    weight = next(item for item in manifest["models"] if item["name"] == "yolo11m_fused_mlx.safetensors")
    if (receipt.get("status") != "passed" or receipt.get("sourceSha") != source or receipt.get("inferenceSha") != inference
            or receipt.get("weightsSha256") != weight["sha256"] or receipt.get("photoSha256") != manifest["photo"]["sha256"]
            or receipt.get("upstreamCommit") != manifest["oracle"]["upstreamCommit"] or receipt.get("device") != "cpu"
            or receipt.get("tensorCount") != 225 or receipt.get("personCount") != 4):
        raise ValueError("independent oracle provenance is incomplete or mismatched")
    if set(receipt.get("outputs", {})) != set(ORACLES):
        raise ValueError("oracle output inventory is incomplete")
    if sum((directory / name).stat().st_size for name in ORACLES) > manifest["budgets"]["oracleOutputBytes"]:
        raise ValueError("oracle artifact budget exceeded")
    for name in ORACLES:
        file = directory / name
        if file.is_symlink() or digest(file) != receipt["outputs"][name]["sha256"]:
            raise ValueError("oracle artifact digest mismatch")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("oracle-input", "proof"), required=True)
    parser.add_argument("--backend", choices=("mlx", "cuda"), required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--inference-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--oracle-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    admit(args.source_sha, args.inference_sha, args.backend)
    if (sys.platform, args.backend) not in (("darwin", "mlx"), ("win32", "cuda")):
        raise ValueError("native selected platform required")
    output = args.output.resolve()
    if ROOT == output or ROOT in output.parents:
        raise ValueError("provisioning output must be outside the checkout")
    output.mkdir(parents=True, exist_ok=False)
    fixture = output / "fixtures"
    fixture.mkdir()
    manifest = load_manifest()
    receipt = {"status": "provisioning", "sourceSha": args.source_sha, "inferenceSha": args.inference_sha,
               "manifestSha256": digest(MANIFEST), "downloadBytes": 0, "assets": [], "onnx": [], "runtime": []}
    try:
        selected = [item for item in manifest["models"] if args.backend in item["backends"] and
                    (args.mode == "proof" or item["name"] == "yolo11m_fused_mlx.safetensors")]
        for asset in [*selected, manifest["photo"]]:
            acquire(asset, fixture / asset["name"], receipt, manifest["budgets"]["assetDownloadBytes"], extra=args.cache_dir)
            if asset["name"].endswith(".onnx"):
                receipt["onnx"].append(inspect_onnx(fixture / asset["name"]))
        if args.mode == "proof":
            if args.backend == "mlx":
                if args.oracle_dir is None:
                    raise ValueError("independent oracle artifact required")
                receipt["oracle"] = validate_oracles(args.oracle_dir, args.source_sha, args.inference_sha, manifest)
                for name in (*ORACLES, "oracle.json"):
                    shutil.copyfile(args.oracle_dir / name, fixture / name)
            remaining = manifest["budgets"]["extractedRuntimeBytes"]
            for asset in manifest["runtime"][args.backend]:
                wheel = output / "downloads" / asset["name"]
                acquire(asset, wheel, receipt, manifest["budgets"]["assetDownloadBytes"])
                files = extract_runtime(wheel, output / "runtime" / asset["kind"], asset["kind"], args.backend, remaining)
                remaining -= sum(item["bytes"] for item in files)
                receipt["runtime"] += files
                wheel.unlink()
            dylib = output / "runtime/ort" / ("libonnxruntime.dylib" if args.backend == "mlx" else "onnxruntime.dll")
            receipt["ortDylib"] = str(dylib)
            with open(os.environ["GITHUB_ENV"], "a", encoding="utf-8") as env:
                env.write(f"CV_FIXTURE_DIR={fixture}\nORT_DYLIB_PATH={dylib}\nCV_PROVISION_RECEIPT={output / 'provisioning.json'}\n")
                if args.backend == "cuda":
                    env.write(f"SCENEWORKS_ORT_CUDA_DIR={output / 'runtime/cuda'}\nSCENEWORKS_ORT_CUDNN_DIR={output / 'runtime/cuda'}\n")
        receipt["status"] = "passed"
    except Exception as error:
        receipt["status"] = "failed"
        receipt["error"] = str(error)
        raise
    finally:
        (output / "provisioning.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
